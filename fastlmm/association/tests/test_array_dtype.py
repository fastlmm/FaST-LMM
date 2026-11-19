"""Workflow-wide precision, raw-input validation, and environment restoration."""
import os
import warnings

import numpy as np
import pytest
from bed_reader import GpuFloat64Warning, has_gpu_support, to_bed
from pysnptools.snpreader import Bed, SnpData
from fastlmm.association import single_snp
from fastlmm.association.single_snp import _as_dtype, _workflow_array_module


@pytest.mark.parametrize("configured", ["float32", "float64"])
def test_workflow_precision_precedence_and_restoration(monkeypatch, configured):
    monkeypatch.setenv("ARRAY_DTYPE", configured)
    monkeypatch.setenv("ARRAY_MODULE", "numpy")
    for explicit in (None, "float32", "float64"):
        expected = configured if explicit is None else explicit
        with _workflow_array_module(np, False, dtype=explicit):
            assert os.environ["ARRAY_DTYPE"] == expected
        assert os.environ["ARRAY_DTYPE"] == configured
        with pytest.raises(RuntimeError, match="deliberate"):
            with _workflow_array_module(np, False, dtype=explicit):
                raise RuntimeError("deliberate")
        assert os.environ["ARRAY_DTYPE"] == configured


@pytest.mark.parametrize("unsafe", [1e-50, 1e40, -1e-50, -1e40])
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_range_validation_precedes_narrowing(unsafe, backend):
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    raw = np.array([[unsafe], [1.]], dtype=np.float64)
    with pytest.raises(ValueError, match="outside the range"):
        _as_dtype(xp, raw, np.float32, "pheno")
    assert raw[0, 0] == unsafe


@pytest.mark.parametrize("explicit", [None, "float32"])
@pytest.mark.parametrize("chromosome_covariates", [False, True])
def test_float32_workflow_avoids_float64_gpu_allocations(tmp_path, monkeypatch, explicit, chromosome_covariates):
    if not has_gpu_support():
        pytest.skip("requires the cubecl-cuda Cargo feature")
    cupy = pytest.importorskip("cupy")
    monkeypatch.setenv("ARRAY_DTYPE", "float32" if explicit is None else "float64")
    rng = np.random.default_rng(123)
    path = tmp_path / "precision.bed"
    to_bed(path, rng.integers(0, 3, size=(40, 24), dtype=np.int8),
           properties={"chromosome": np.repeat([1, 2], 12)})
    reader = Bed(path, count_A1=True)
    # Use a file to exercise Pheno parsing and its metadata-only initial access.
    pheno_path = tmp_path / "phenotype.txt"
    with pheno_path.open("w") as stream:
        stream.write("FID IID y\n")
        for iid, value in zip(reader.iid, rng.normal(size=40)):
            stream.write(f"{iid[0]} {iid[1]} {value}\n")
    covariate = SnpData(iid=reader.iid, sid=["covariate"], val=rng.normal(size=(40, 1)), xp=np)
    by_chromosome = {chromosome: covariate for chromosome in (1, 2)} if chromosome_covariates else None
    # A duplicate covariate would be collinear; use it only as a chromosome term.
    arguments = dict(pheno=pheno_path, covar_by_chrom=by_chromosome,
                     h2=0.3, force_full_rank=True, dtype=explicit)
    expected = single_snp(reader, xp=np, **arguments)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GpuFloat64Warning)
        actual = single_snp(reader, xp=cupy, **arguments)
    np.testing.assert_array_equal(actual.SNP, expected.SNP)
    np.testing.assert_allclose(np.log10(actual.PValue), np.log10(expected.PValue), rtol=0, atol=1.5e-4)


def test_concurrent_default_precision_resolves_after_restoration(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setenv("ARRAY_DTYPE", "float64")
    first_entered, second_attempted, release_first = Event(), Event(), Event()
    observed = []

    def first():
        with _workflow_array_module(np, False, dtype="float32"):
            observed.append(os.environ["ARRAY_DTYPE"])
            first_entered.set()
            assert release_first.wait(10)

    def second():
        second_attempted.set()
        with _workflow_array_module(np, False):
            observed.append(os.environ["ARRAY_DTYPE"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(first)
        try:
            assert first_entered.wait(10)
            second_result = executor.submit(second)
            assert second_attempted.wait(10)
        finally:
            release_first.set()
        first_result.result(timeout=10)
        second_result.result(timeout=10)
    assert observed == ["float32", "float64"]
    assert os.environ["ARRAY_DTYPE"] == "float64"


def test_gpu_user_values_are_validated_before_widening_or_narrowing(monkeypatch):
    from fastlmm.association.single_snp import _cpu_user_data, _read_user_values

    cupy = pytest.importorskip("cupy")
    monkeypatch.setenv("ARRAY_DTYPE", "float32")
    raw = SnpData(iid=[["f", "0"], ["f", "1"]], sid=["y"],
                  val=cupy.array([[1.], [2.]], dtype=cupy.float32), xp=cupy)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GpuFloat64Warning)
        cpu = _cpu_user_data(raw)
        result = _read_user_values(cpu, cupy, np.float32, "pheno")
    assert isinstance(cpu.val, np.ndarray) and cpu.val.dtype == np.float64
    assert isinstance(result, cupy.ndarray) and result.dtype == np.float32
    np.testing.assert_array_equal(cupy.asnumpy(result), [[1.], [2.]])
    unsafe = SnpData(iid=raw.iid, sid=raw.sid,
                     val=cupy.array([[1e-50], [1.]], dtype=cupy.float64), xp=cupy)
    with pytest.raises(ValueError, match="outside the range"):
        _read_user_values(_cpu_user_data(unsafe), cupy, np.float32, "pheno")


def test_conflicting_worker_precision_fails_clearly(monkeypatch):
    from fastlmm.association.single_snp import _check_worker_dtype

    monkeypatch.setenv("ARRAY_DTYPE", "float64")
    with pytest.raises(ValueError, match="disagrees with workflow dtype"):
        _check_worker_dtype(np.float32)
    _check_worker_dtype(np.float64)
    monkeypatch.delenv("ARRAY_DTYPE")
    _check_worker_dtype(np.float32)  # legacy dependencies do not establish the policy
