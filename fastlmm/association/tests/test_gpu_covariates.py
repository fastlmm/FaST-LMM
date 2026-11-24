"""Chromosome covariates follow the selected backend and align sample IDs."""

from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest
from pysnptools.snpreader import SnpData
from pysnptools.util import asnumpy
from pysnptools.util.gpu import has_gpu_support

from fastlmm.association.single_snp import _create_covar_chrom


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_chromosome_covariates_on_selected_backend(backend, dtype, monkeypatch):
    if backend == "cupy" and not has_gpu_support():
        pytest.skip("requires CuPy and a usable CUDA device")
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    iid = np.array([["family", str(sample)] for sample in range(5)])
    values = np.arange(5, dtype=dtype).reshape(-1, 1)
    covariates = SnpData(iid=iid, sid=["covariate"], val=values, xp=np)
    # A different sample order must still align with the shared covariates.
    rows = [3, 0, 4, 1, 2]
    chromosome = SnpData(iid=iid[rows], sid=["chromosome_covariate"],
                         val=(values + 10)[rows], xp=np)
    monkeypatch.setenv("ARRAY_MODULE", backend)
    monkeypatch.delenv("ARRAY_DTYPE", raising=False)
    result = _create_covar_chrom(covariates, {1: chromosome}, 1)
    assert isinstance(result.val, xp.ndarray)
    np.testing.assert_array_equal(result.iid, iid)
    np.testing.assert_array_equal(result.sid, ["covariate", "chromosome_covariate"])
    # The legacy read default is float64; this fix must not change that default.
    assert result.val.dtype == np.float64
    np.testing.assert_array_equal(asnumpy(result.val), np.column_stack((values, values + 10)))
    np.testing.assert_array_equal(covariates.val, values)
    np.testing.assert_array_equal(chromosome.val, (values + 10)[rows])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("in_memory", [False, True])
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_loco_model_inputs(tmp_path, dtype, in_memory, backend):
    from bed_reader import to_bed
    from fastlmm.association import single_snp
    from fastlmm.inference.lmm_cov import LMM
    from pysnptools.snpreader import Bed

    if backend == "cupy" and not has_gpu_support():
        pytest.skip("requires CuPy and a usable CUDA device")
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    rng = np.random.default_rng(42)
    path = tmp_path / "covariates.bed"
    to_bed(path, rng.integers(0, 3, size=(40, 24), dtype=np.int8),
           properties={"chromosome": np.repeat([1, 2], 12)})
    reader = Bed(path, count_A1=True)
    phenotype = SnpData(iid=reader.iid, sid=["phenotype"], val=rng.normal(size=(40, 1)), xp=np)
    covariates = SnpData(iid=reader.iid, sid=["covariate"], val=rng.normal(size=(40, 1)), xp=np)
    chromosome_values = {chromosome: rng.normal(size=(40, 1)) for chromosome in (1, 2)}
    by_chromosome = {
        chromosome: SnpData(iid=reader.iid[::-1], sid=["chromosome_covariate"],
                            val=values[::-1].copy(), xp=np)
        for chromosome, values in chromosome_values.items()
    }
    arguments = dict(pheno=phenotype, covar=covariates, covar_by_chrom=by_chromosome,
                     h2=0.3, force_full_rank=True, dtype=dtype)
    expected = single_snp(reader, xp=np, **arguments)
    source = reader.read(dtype=dtype, xp=xp) if in_memory else reader
    module = import_module("fastlmm.association.single_snp")
    original_snp_tester = module._snp_tester
    original_evaluate = LMM.nLLeval
    tested_chromosomes = set()
    tested_blocks = []

    def snp_tester(test_snps, interact, pheno, lmm, *args, **kwargs):
        chromosome, = np.unique(test_snps.pos[:, 0])
        rows = reader.iid_to_index(test_snps.iid)
        assert isinstance(lmm.X, xp.ndarray)
        assert lmm.X.dtype == np.dtype(dtype)
        expected_design = np.column_stack((
            covariates.val[rows], chromosome_values[chromosome][rows], np.ones(len(rows))
        )).astype(dtype)
        np.testing.assert_array_equal(asnumpy(lmm.X), expected_design)
        tested_chromosomes.add(chromosome)
        return original_snp_tester(test_snps, interact, pheno, lmm, *args, **kwargs)

    def evaluate(model, *args, **kwargs):
        if kwargs.get("snps") is not None:
            values = kwargs["snps"]
            assert isinstance(values, xp.ndarray)
            assert values.dtype == np.dtype(dtype)
            tested_blocks.append(values.shape)
        return original_evaluate(model, *args, **kwargs)

    with patch.object(module, "_snp_tester", snp_tester), patch.object(LMM, "nLLeval", evaluate):
        actual = single_snp(source, xp=xp, **arguments)
    assert tested_chromosomes == {1, 2}
    assert tested_blocks
    actual = actual.sort_values("SNP")
    expected = expected.sort_values("SNP")
    np.testing.assert_array_equal(actual.SNP, expected.SNP)
    # Allow the same CPU/GPU rounding margins as the other association tests.
    tolerance = 1.5e-4 if dtype == "float32" else 2e-8
    np.testing.assert_allclose(np.log10(actual.PValue), np.log10(expected.PValue), rtol=0, atol=tolerance)
