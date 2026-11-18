"""Backend selection at real process boundaries, with actual read/math observations."""
import importlib
import json
import os
from unittest.mock import patch

import numpy as np
import pytest
from bed_reader import has_gpu_support, to_bed
from pysnptools.snpreader import Bed, SnpData
from pysnptools.util import get_array_module
from pysnptools.util.mapreduce1.runner import LocalMultiProc

from fastlmm.association import single_snp
from fastlmm.inference.lmm_cov import LMM


def _record(path, kind, backend, dtype=None):
    expected = os.environ["ARRAY_MODULE"]
    assert backend == expected, (kind, backend, expected)
    with open(f"{path}.{os.getpid()}.jsonl", "a") as stream:
        stream.write(json.dumps({"kind": kind, "backend": backend, "precision": os.environ.get("ARRAY_DTYPE"), "actual_dtype": None if dtype is None else str(dtype), "pid": os.getpid()}) + "\n")


class _TracingBed(Bed):
    def __init__(self, path, trace):
        super().__init__(path, count_A1=True)
        self.trace = str(trace)

    def _read_with_standardizing(self, *args, **kwargs):
        from bed_reader import bed_reader as native
        from fastlmm.inference.lmm_cov import LMM

        trace = self.trace
        if not getattr(LMM.nLLeval, "_environment_trace", False):
            original_evaluate = LMM.nLLeval

            def evaluate(model, *arguments, **keywords):
                _record(trace, "evaluate", model._xp.__name__, model.Y.dtype)
                if keywords.get("snps") is not None:
                    _record(trace, "test_snps", get_array_module(keywords["snps"]).__name__, keywords["snps"].dtype)
                return original_evaluate(model, *arguments, **keywords)

            evaluate._environment_trace = True
            LMM.nLLeval = evaluate
        backend = os.environ["ARRAY_MODULE"]
        if kwargs.get("xp") is not None:
            from pysnptools.util import array_module
            assert array_module(kwargs["xp"]).__name__ == backend
        if backend == "cupy":
            import cupy as cp
            if not getattr(cp.linalg.eigh, "_environment_trace", False):
                original_eigh = cp.linalg.eigh

                def eigh(values, *arguments, **keywords):
                    _record(trace, "eigh", get_array_module(values).__name__, values.dtype)
                    return original_eigh(values, *arguments, **keywords)

                eigh._environment_trace = True
                cp.linalg.eigh = eigh
        with patch.object(native, "read_gpu", wraps=native.read_gpu) as gpu_read:
            result = super()._read_with_standardizing(*args, **kwargs)
        data = result[0] if isinstance(result, tuple) else result
        _record(trace, "read", get_array_module(data.val).__name__, data.val.dtype)
        if backend == "cupy":
            assert gpu_read.call_count > 0
            _record(trace, "native_gpu_read", backend)
        else:
            assert gpu_read.call_count == 0
        return result


def _worker_environment(task_index):
    return {"ARRAY_MODULE": "cupy" if task_index % 2 == 0 else "numpy"}


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("phenotype_count", [1, 3])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mixed_workers_select_environment_and_restore(tmp_path, monkeypatch, outer, phenotype_count, dtype):
    if not has_gpu_support():
        pytest.skip("requires bed-reader built with the cubecl-cuda Cargo feature")
    cp = pytest.importorskip("cupy")
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(31)
    path = tmp_path / "tiny.bed"
    to_bed(path, rng.integers(0, 3, size=(40, 24), dtype=np.int8),
           properties={"chromosome": np.repeat([1, 2], 12)})
    trace = tmp_path / "backend"
    reader = _TracingBed(path, trace)
    pheno = SnpData(iid=reader.iid, sid=[f"y{index}" for index in range(phenotype_count)],
                    val=rng.normal(size=(40, phenotype_count)), xp=np)
    expected = single_snp(Bed(path, count_A1=True), pheno,
                          leave_out_one_chrom=outer, h2=0.3, force_full_rank=True, xp=np, dtype=dtype)
    module = importlib.import_module("fastlmm.association.single_snp")

    def small_blocks(first, second, *args):
        first.block_size = 4
        return first, second, 4

    runner = LocalMultiProc(2, just_one_process=False, taskindex_to_environ=_worker_environment)
    with patch.object(LMM, "nLLeval", LMM.nLLeval), \
         patch.object(cp.linalg, "eigh", cp.linalg.eigh), \
         patch.object(module, "_set_block_size", small_blocks), \
         patch.dict("os.environ", {"ARRAY_MODULE": "before-call", "ARRAY_DTYPE": "before-precision"}):
        actual = single_snp(reader, pheno, leave_out_one_chrom=outer,
                            h2=0.3, force_full_rank=True, xp=cp, runner=runner, dtype=dtype)
        assert os.environ["ARRAY_MODULE"] == "before-call"
        assert os.environ["ARRAY_DTYPE"] == "before-precision"
    records = [json.loads(line) for file in tmp_path.glob("backend.*.jsonl") for line in file.read_text().splitlines()]
    workers = [record for record in records if record["pid"] != os.getpid()]
    assert {record["precision"] for record in workers} == {dtype}
    assert {record["actual_dtype"] for record in workers if record["kind"] in ("evaluate", "test_snps")} == {dtype}
    assert {record["backend"] for record in workers if record["kind"] == "evaluate"} == {"numpy", "cupy"}
    assert {record["backend"] for record in workers if record["kind"] == "read"} == {"numpy", "cupy"}
    assert any(record["kind"] == "native_gpu_read" for record in workers)
    assert any(record["kind"] == "eigh" for record in records)
    columns = ["sid_index"] + (["Pheno"] if phenotype_count > 1 else [])
    actual = actual.sort_values(columns)
    expected = expected.sort_values(columns)
    # Use the established single_snp backend bounds: float32 kernel formation,
    # decomposition and rotation contribute independent roundoff.
    tolerance = 1.5e-4 if dtype == "float32" else 2e-8
    np.testing.assert_allclose(np.log10(actual.PValue), np.log10(expected.PValue), rtol=0, atol=tolerance)


def test_environment_restored_after_failure():
    with patch.dict("os.environ", {"ARRAY_MODULE": "before-call"}):
        with pytest.raises(AssertionError, match="test_snps must be given"):
            single_snp(None, None, xp=np)
        assert os.environ["ARRAY_MODULE"] == "before-call"


def test_concurrent_workflows_restore_environment():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, RLock
    import pandas as pd

    module = importlib.import_module("fastlmm.association.single_snp")
    first_entered = Event()
    second_attempted = Event()
    second_entered = Event()
    release_first = Event()
    lock = RLock()

    class ObservedLock:
        def __enter__(self):
            if first_entered.is_set():
                second_attempted.set()
            lock.acquire()

        def __exit__(self, *exception):
            lock.release()

    def internal(**kwargs):
        if not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(10)
        else:
            second_entered.set()
        assert os.environ["ARRAY_MODULE"] == "numpy"
        return pd.DataFrame({"sid_index": np.arange(kwargs["test_snps"].sid_count)})

    data = SnpData(iid=[["a", "a"], ["b", "b"], ["c", "c"]],
                   sid=["s"], val=np.array([[0.0], [1.0], [2.0]]), xp=np)
    with patch.dict("os.environ", {"ARRAY_MODULE": "before-call"}), \
         patch.object(module, "_array_module_lock", ObservedLock()), \
         patch.object(module, "_internal_single", internal), \
         ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(single_snp, data, data, xp=np, leave_out_one_chrom=False)
        try:
            assert first_entered.wait(10)
            second = executor.submit(single_snp, data, data, xp=np, leave_out_one_chrom=False)
            assert second_attempted.wait(10)
            assert not second_entered.is_set()
        finally:
            release_first.set()
        first.result(timeout=10)
        second.result(timeout=10)
        assert os.environ["ARRAY_MODULE"] == "before-call"


class _FailingBed(_TracingBed):
    def _read_with_standardizing(self, *args, **kwargs):
        _record(self.trace, "forced_failure", os.environ["ARRAY_MODULE"])
        raise RuntimeError("intentional worker read failure")


def test_environment_restored_after_worker_failure(tmp_path, monkeypatch):
    if not has_gpu_support():
        pytest.skip("requires bed-reader built with the cubecl-cuda Cargo feature")
    pytest.importorskip("cupy")
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "failure.bed"
    rng = np.random.default_rng(32)
    to_bed(path, rng.integers(0, 3, size=(10, 4), dtype=np.int8),
           properties={"chromosome": [1, 1, 2, 2]})
    trace = tmp_path / "failure"
    reader = _FailingBed(path, trace)
    pheno = SnpData(iid=reader.iid, sid=["y"], val=rng.normal(size=(10, 1)), xp=np)
    runner = LocalMultiProc(1, taskindex_to_environ=_worker_environment)
    with patch.dict("os.environ", {"ARRAY_MODULE": "before-call"}):
        with pytest.raises(Exception, match="non-zero return code"):
            single_snp(reader, pheno, xp=np, runner=runner, h2=0.3)
        assert os.environ["ARRAY_MODULE"] == "before-call"
    records = [json.loads(line) for file in tmp_path.glob("failure.*.jsonl")
               for line in file.read_text().splitlines()]
    assert any(record["pid"] != os.getpid() and record["kind"] == "forced_failure" for record in records)
