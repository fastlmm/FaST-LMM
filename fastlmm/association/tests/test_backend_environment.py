"""Backend selection at real process boundaries, with actual read/math observations."""
import importlib
import json
import os
from unittest.mock import patch

import numpy as np
import pytest
from bed_reader import to_bed
from pysnptools.util.gpu import has_gpu_support
from pysnptools.snpreader import Bed, SnpData
from pysnptools.util.mapreduce1.runner import LocalMultiProc

from fastlmm.association import single_snp
from fastlmm.association.tests._backend_tracing import _record, _TracingBed
from fastlmm.inference.lmm_cov import LMM


def _worker_environment(task_index):
    return {"ARRAY_MODULE": "cupy" if task_index % 2 == 0 else "numpy"}


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("phenotype_count", [1, 3])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mixed_workers_select_environment_and_restore(tmp_path, monkeypatch, outer, phenotype_count, dtype):
    if not has_gpu_support():
        pytest.skip("requires CuPy and a usable CUDA device")
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
    assert any(record["kind"] == "eigh" for record in records)
    columns = ["sid_index"] + (["Pheno"] if phenotype_count > 1 else [])
    actual = actual.sort_values(columns)
    expected = expected.sort_values(columns)
    # Max |Δlog10 p| measured across these cases: 8.97e-7 for float32
    # (including float32 kernel formation on both backends) and 5.55e-15 for float64.
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
        pytest.skip("requires CuPy and a usable CUDA device")
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


def _nested_lmm():
    from fastlmm.inference.lmm import LMM as InferenceLMM

    rng = np.random.default_rng(17)
    model = InferenceLMM()
    model.setG(rng.standard_normal((40, 10)))
    model.setX(np.ones((40, 1)))
    model.sety(rng.standard_normal(40))
    return model


def test_nested_cpu64_calls_reuse_owner_environment(monkeypatch):
    # Inner-loop methods called inside a workflow must not rewrite the
    # environment on every call (an epistasis slowdown), yet must still
    # return the same results as an unwrapped call.
    from fastlmm.inference.lmm import LMM as InferenceLMM
    from fastlmm.util import _backend

    model = _nested_lmm()
    expected = InferenceLMM.nLLeval.__wrapped__(model, delta=1.0, REML=False)
    with _backend.cpu64_scope("outer"):
        def forbidden(*args, **kwargs):
            raise AssertionError("nested call re-entered the CPU64 environment")

        monkeypatch.setattr(_backend, "_cpu64_environment", forbidden)
        actual = model.nLLeval(delta=1.0, REML=False)
    assert actual["nLL"] == expected["nLL"]
    np.testing.assert_array_equal(actual["beta"], expected["beta"])


def test_nested_cpu64_calls_still_check_model_and_inputs():
    from fastlmm.util import _backend

    model = _nested_lmm()
    with _backend.cpu64_scope("outer"):
        float32_y = np.ones(40, dtype=np.float32)
        with pytest.warns(UserWarning, match="copied to numpy/float64"):
            model.sety(float32_y)
        assert model.y.dtype == np.float64
        model.S = model.S.astype(np.float32)
        with pytest.raises(ValueError, match="cached model S is float32"):
            model.nLLeval(delta=1.0, REML=False)


def test_nested_cpu64_call_restores_changed_environment():
    # If code inside a workflow changes the environment, a nested call takes
    # the full path: it runs with NumPy/float64 and then restores the change.
    from fastlmm.util import _backend
    from fastlmm.util._backend import cpu64

    observed = []

    @cpu64(api="nested probe")
    def probe():
        observed.append(os.environ["ARRAY_DTYPE"])

    with patch.dict("os.environ", {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float64"}):
        with _backend.cpu64_scope("outer"):
            os.environ["ARRAY_DTYPE"] = "float32"
            with pytest.warns(UserWarning, match="qualified only for numpy/float64"):
                probe()
            assert os.environ["ARRAY_DTYPE"] == "float32"
            os.environ["ARRAY_DTYPE"] = "float64"
    assert observed == ["float64"]
