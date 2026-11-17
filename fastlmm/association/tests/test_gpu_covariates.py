"""Chromosome covariates follow the selected backend and align sample IDs."""

from unittest.mock import patch

import numpy as np
import pytest
from pysnptools.snpreader import SnpData
from pysnptools.util import asnumpy

from fastlmm.association.single_snp import _create_covar_chrom


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_chromosome_covariates_on_selected_backend(backend, dtype):
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    iid = np.array([["family", str(sample)] for sample in range(5)])
    values = np.arange(5, dtype=dtype).reshape(-1, 1)
    covariates = SnpData(iid=iid, sid=["covariate"], val=values, xp=np)
    # A different sample order must still align with the shared covariates.
    rows = [3, 0, 4, 1, 2]
    chromosome = SnpData(iid=iid[rows], sid=["chromosome_covariate"],
                         val=(values + 10)[rows], xp=np)
    with patch.dict("os.environ", {"ARRAY_MODULE": backend}):
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
def test_loco_covariates_and_native_test_snps(tmp_path, dtype, in_memory):
    from bed_reader import has_gpu_support, to_bed
    from fastlmm.association import single_snp
    from fastlmm.inference.lmm_cov import LMM
    from pysnptools.snpreader import Bed

    if not has_gpu_support():
        pytest.skip("requires bed-reader built with the cubecl-cuda Cargo feature")
    cupy = pytest.importorskip("cupy")
    from bed_reader import is_cubecl_backed
    rng = np.random.default_rng(42)
    path = tmp_path / "covariates.bed"
    to_bed(path, rng.integers(0, 3, size=(40, 24), dtype=np.int8),
           properties={"chromosome": np.repeat([1, 2], 12)})
    reader = Bed(path, count_A1=True)
    phenotype = SnpData(iid=reader.iid, sid=["phenotype"], val=rng.normal(size=(40, 1)), xp=np)
    covariates = SnpData(iid=reader.iid, sid=["covariate"], val=rng.normal(size=(40, 1)), xp=np)
    by_chromosome = {chromosome: SnpData(iid=reader.iid[::-1], sid=["chromosome_covariate"],
                                       val=rng.normal(size=(40, 1)), xp=np)
                     for chromosome in (1, 2)}
    arguments = dict(pheno=phenotype, covar=covariates, covar_by_chrom=by_chromosome,
                     h2=0.3, force_full_rank=True, dtype=dtype)
    expected = single_snp(reader, xp=np, **arguments)
    source = reader.read(dtype=dtype, xp=cupy) if in_memory else reader
    original_evaluate = LMM.nLLeval
    tested_blocks = []

    def evaluate(model, *args, **kwargs):
        if kwargs.get("snps") is not None:
            values = kwargs["snps"]
            assert isinstance(values, cupy.ndarray) and is_cubecl_backed(values)
            tested_blocks.append(values.shape)
        return original_evaluate(model, *args, **kwargs)

    with patch.object(LMM, "nLLeval", evaluate):
        actual = single_snp(source, xp=cupy, **arguments)
    assert tested_blocks
    actual = actual.sort_values("SNP")
    expected = expected.sort_values("SNP")
    np.testing.assert_array_equal(actual.SNP, expected.SNP)
    # Established backend bounds account for float32 kernel formation,
    # decomposition, and rotation; float64 retains the tighter bound.
    tolerance = 1.5e-4 if dtype == "float32" else 2e-8
    np.testing.assert_allclose(np.log10(actual.PValue), np.log10(expected.PValue), rtol=0, atol=tolerance)
