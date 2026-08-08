"""Smoke-test an installed FaST-LMM distribution outside its source tree."""

from importlib.resources import files
from pathlib import Path

import numpy as np
from pysnptools.snpreader import SnpData

import fastlmm
import fastlmm.association
import fastlmm.inference
from fastlmm.association import single_snp_linreg


def main():
    installed_package = Path(fastlmm.__file__).resolve()
    source_package = Path(__file__).resolve().parents[1] / "fastlmm"
    assert source_package.resolve() not in installed_package.parents

    package = files("fastlmm")
    required_data = [
        package.joinpath("util", "fastlmm.hashdown.json"),
        package.joinpath("association", "Fastlmm_autoselect", "fastlmmc"),
        package.joinpath(
            "feature_selection", "examples", "toydata.5chrom.bed"
        ),
    ]
    assert all(path.is_file() for path in required_data)

    iid = np.array([["f0", "i0"], ["f1", "i1"], ["f2", "i2"], ["f3", "i3"]])
    test_snps = SnpData(
        iid=iid,
        sid=np.array(["s0", "s1"]),
        val=np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [0.0, 2.0]]),
    )
    pheno = SnpData(
        iid=iid,
        sid=np.array(["p"]),
        val=np.array([[1.0], [2.0], [3.0], [4.0]]),
    )
    result = single_snp_linreg(
        test_snps=test_snps,
        pheno=pheno,
        count_A1=False,
    )
    assert set(result.SNP) == {"s0", "s1"}
    assert np.isfinite(result.PValue).all()

    print(f"Tested installed FaST-LMM from {fastlmm.__file__}")


if __name__ == "__main__":
    main()
