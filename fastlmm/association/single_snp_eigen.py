import logging
import os
from pathlib import Path
import numpy as np
import pysnptools.util as pstutil
from unittest.mock import patch
from fastlmm.inference.fastlmm_predictor import (
    _pheno_fixup,
    _snps_fixup,
    _kernel_fixup,
)
from pysnptools.standardizer import Unit
from fastlmm.inference import LMM


def eigen_from_kernel(K, count_A1=None):
    """!!!cmk documentation"""
    # !!!cmk could offer a low-memory path that uses memmapped files
    assert K is not None
    K = _kernel_fixup(K, iid_if_none=None, standardizer=Unit(), count_A1=count_A1)
    assert K.iid0 is K.iid1, "Expect K to be square"

    K = K._read_with_standardizing(to_kerneldata=True, return_trained=False)
    # !!! cmk ??? pass in a new argument, the kernel_standardizer(???)
    logging.debug("About to eigh")
    w, v = np.linalg.eigh(K.val)  # !!! cmk do SVD sometimes?
    logging.debug("Done with to eigh")
    if np.any(w < -0.1):
        logging.warning("kernel contains a negative Eigenvalue")
    return w, v


# !!!LATER add warning here (and elsewhere) K0 or K1.sid_count < test_snps.sid_count,
#  might be a covar mix up.(but only if a SnpKernel
def single_snp_eigen(
    test_snps,
    pheno,
    w,
    v,
    covar=None,  # !!!cmk covar_by_chrom=None, leave_out_one_chrom=True,
    output_file_name=None,
    log_delta=None,
    # !!!cmk cache_file=None, GB_goal=None, interact_with_snp=None,
    # !!!cmk runner=None, map_reduce_outer=True,
    # !!!cmk pvalue_threshold=None,
    # !!!cmk random_threshold=None,
    # !!!cmk random_seed = 0,
    min_log_delta=-5,  # !!!cmk make this a range???
    max_log_delta=10,
    # !!!cmk xp=None,
    count_A1=None,
):
    """cmk documentation"""
    # !!!LATER raise error if covar has NaN
    # cmk t0 = time.time()

    if output_file_name is not None:
        os.makedirs(Path(output_file_name).parent, exist_ok=True)

    xp = pstutil.array_module("numpy")
    with patch.dict("os.environ", {"ARRAY_MODULE": xp.__name__}) as _:

        assert test_snps is not None, "test_snps must be given as input"
        test_snps = _snps_fixup(test_snps, count_A1=count_A1)
        pheno = _pheno_fixup(pheno, count_A1=count_A1).read()
        good_values_per_iid = (pheno.val == pheno.val).sum(axis=1)
        assert not np.any(
            (good_values_per_iid > 0) * (good_values_per_iid < pheno.sid_count)
        ), "With multiple phenotypes, an individual's values must either be all missing or have no missing."
        # !!!cmk multipheno
        # drop individuals with no good pheno values.
        pheno = pheno[good_values_per_iid > 0, :]
        covar = _pheno_fixup(covar, iid_if_none=pheno.iid, count_A1=count_A1)

        # !!!cmk assert covar_by_chrom is None, "When 'leave_out_one_chrom' is False,
        #  'covar_by_chrom' must be None"
        # !!!cmk fix up w and v
        test_snps, pheno, covar = pstutil.intersect_apply(
            [test_snps, pheno, covar]
        )  # !!!cmk w and v
        logging.debug("# of iids now {0}".format(test_snps.iid_count))
        # !!!cmk K0, K1, block_size = _set_block_size(K0, K1, mixing, GB_goal,
        #  force_full_rank, force_low_rank)

        # !!! cmk
        # if h2 is not None and not isinstance(h2, np.ndarray):
        #     h2 = np.repeat(h2, pheno.shape[1])

        covar_val = xp.asarray(covar.read(view_ok=True, order="A").val)
        covar_val = xp.c_[
            covar_val, xp.ones((test_snps.iid_count, 1))
        ]  # view_ok because np.c_ will allocation new memory

        assert pheno.sid_count >= 1, "Expect at least one phenotype"

        # view_ok because this code already did a fresh read to look for any
        #  missing values
        multi_y = xp.asarray(pheno.read(view_ok=True, order="A").val)

        lmm = LMM()
        lmm.U = w
        lmm.S = v

        if log_delta is None:
            logging.info("searching for internal delta")
            # !!! cmk so covar and pheno don't matter if log delta is given, right????
            lmm.setX(
                covar
            )  # !!! cmk with multipheno is it going to be O(covar*covar*y)???
            lmm.sety(multi_y)  # !!! cmk need to check that just one pheno for now
            # log delta is used here. Might be better to use findH2, but if so will need to normalized G so that its K's diagonal would sum to iid_count

            # As per the paper, we optimized delta with REML=True, but
            # we will later optimize beta and find log likelihood with ML (REML=False)
            # !!! cmk so sid_count need not be given if doing full rank, right?
            result = lmm.find_log_delta(
                REML=True,
                sid_count=1,
                min_log_delta=min_log_delta,
                max_log_delta=max_log_delta,
            )
            # !!! cmkwhat about findA2H2? minH2=0.00001
            log_delta = result["log_delta"]

        # !!!cmk internal/external doesn't matter if full rank, right???
        delta = np.exp(log_delta)
        logging.info("delta={0}".format(delta))
        logging.info("log_delta={0}".format(log_delta))

        frame = _snp_tester(
            test_snps,
            interact,
            pheno,
            lmm,
            block_size,
            output_file_name,
            runner,
            h2,
            mixing,
            pvalue_threshold,
            random_threshold,
            random_seed,
        )

        return frame

    return frame
