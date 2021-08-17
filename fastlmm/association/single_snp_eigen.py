import logging
import pandas as pd
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import pysnptools.util as pstutil
from unittest.mock import patch
from pysnptools.standardizer import Unit
from pysnptools.eigenreader import EigenData
from fastlmm.inference.fastlmm_predictor import (
    _pheno_fixup,
    _snps_fixup,
    _kernel_fixup,
)
from fastlmm.inference import LMM


def eigen_from_kernel(K, kernel_standardizer, count_A1=None):
    """!!!cmk documentation"""
    # !!!cmk could offer a low-memory path that uses memmapped files
    assert K is not None
    K = _kernel_fixup(K, iid_if_none=None, standardizer=Unit(), count_A1=count_A1)
    assert K.iid0 is K.iid1, "Expect K to be square"

    #!!!cmk understand _read_kernel, _read_with_standardizing

    K = K._read_with_standardizing(kernel_standardizer=kernel_standardizer,to_kerneldata=True, return_trained=False)
    # !!! cmk ??? pass in a new argument, the kernel_standardizer(???)
    logging.debug("About to eigh")
    w, v = np.linalg.eigh(K.val)  # !!! cmk do SVD sometimes?
    logging.debug("Done with to eigh")
    if np.any(w < -0.1):
        logging.warning("kernel contains a negative Eigenvalue") #!!!cmk this shouldn't happen with a RRM, right?
    # !!!cmk remove very small eigenvalues
    # !!!cmk remove very small eigenvalues in a way that doesn't require a memcopy?
    eigen = EigenData(values=w, vectors=v, iid=K.iid)
    #eigen.vectors[:,eigen.values<.0001]=0.0
    #eigen.values[eigen.values<.0001]=0.0
    #eigen = eigen[:,eigen.values >= .0001] # !!!cmk const
    return eigen


# !!!LATER add warning here (and elsewhere) K0 or K1.sid_count < test_snps.sid_count,
#  might be a covar mix up.(but only if a SnpKernel
def single_snp_eigen(
    test_snps,
    pheno,
    eigenreader,
    covar=None,  # !!!cmk covar_by_chrom=None, leave_out_one_chrom=True,
    output_file_name=None,
    log_delta=None,
    # !!!cmk cache_file=None, GB_goal=None, interact_with_snp=None,
    # !!!cmk runner=None, map_reduce_outer=True,
    # !!!cmk pvalue_threshold=None,
    # !!!cmk random_threshold=None,
    # !!!cmk random_seed = 0,
    #min_log_delta=-5,  # !!!cmk make this a range???
    #max_log_delta=10,
    # !!!cmk xp=None,
    fit_log_delta_via_reml = True,
    test_via_reml = False,
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
        iid_count_before = eigenreader.iid_count
        test_snps, pheno, eigenreader, covar = pstutil.intersect_apply(
            [test_snps, pheno, eigenreader, covar]
        ) 
        logging.debug("# of iids now {0}".format(test_snps.iid_count))
        assert eigenreader.iid_count == iid_count_before, "Expect all of eigenreader's individuals to be in test_snps, pheno, and covar." #cmk ok to lose some?
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
        eigendata = eigenreader.read(view_ok=True, order="A")

        lmm = LMM(forcefullrank=True)
        lmm.X = covar_val
        # !!! cmk with multipheno is it going to be O(covar*covar*y)???
        lmm.y = multi_y[:,0]
        lmm.U = eigendata.vectors
        lmm.S = eigendata.values

        #============
        # iid_count x eid_count  *  iid_count x covar => eid_count * covar
        # O(iid_count x eid_count x covar)
        #=============
        lmm.UX = lmm.U.T.dot(lmm.X)

        #============
        # iid_count x eid_count  *  iid_count x pheno_count => eid_count * pheno_count
        # O(iid_count x eid_count x pheno_count)
        #=============
        lmm.Uy = lmm.U.T.dot(lmm.y)

        if log_delta is None:
            # !!!cmk log delta is used here. Might be better to use findH2, but if so will need to normalized G so that its K's diagonal would sum to iid_count

            logging.info("searching for delta/h2/logdelta")
            result = lmm.findH2(REML=fit_log_delta_via_reml, minH2=0.00001 )
            h2 = result["h2"]
            delta = 1.0/h2-1.0
            log_delta = np.log(delta)
            # cmk As per the paper, we optimized delta with REML=True, but
            # cmk we will later optimize beta and find log likelihood with ML (REML=False)
            # !!! cmk so sid_count need not be given if doing full rank, right?
        else:
            # !!!cmk internal/external doesn't matter if full rank, right???
            delta = np.exp(log_delta)


        logging.info("delta={0}".format(delta))
        logging.info("log_delta={0}".format(log_delta))

        # As per the paper, we previously optimized delta with REML=True, but
        # we optimize beta and find loglikelihood with ML (REML=False)
        # !!! cmk if test_via_reml == fit_log_delta_via_reml this could be skipped
        assert not test_via_reml # !!!cmk
        res_null = nLLevalx(lmm, delta=delta)
        ll_null = -res_null["nLL"]
        h2 = res_null["h2"]

        dataframe = _create_dataframe(test_snps.sid_count)

        # !!!cmk real in batches

        pvalue_list = []
        beta_list = []
        variance_beta_list = []
        for sid_index in range(test_snps.sid_count):
            snps_read = test_snps[:, sid_index].read().standardize()
            lmm.X = np.hstack((covar_val, snps_read.val))
            lmm.UX  = lmm.U.T.dot(lmm.X)


            res_alt = nLLevalx(lmm, delta)


            ll_alt = -res_alt["nLL"]
            beta = res_alt["beta"][-1]
            variance_beta = (
                res_alt["variance_beta"][-1] if "variance_beta" in res_alt else np.nan
            )
            test_statistic = ll_alt - ll_null
            pvalue = stats.chi2.sf(2.0 * test_statistic, df=1)

            pvalue_list.append(pvalue)
            beta_list.append(beta)
            variance_beta_list.append(variance_beta)

        dataframe["sid_index"] = range(test_snps.sid_count)
        dataframe['SNP'] = test_snps.sid
        dataframe['Chr'] = test_snps.pos[:,0]
        dataframe['GenDist'] = test_snps.pos[:,1]
        dataframe['ChrPos'] = test_snps.pos[:,2]
        dataframe["PValue"] = pvalue_list
        dataframe['SnpWeight'] = beta_list
        dataframe['SnpWeightSE'] = np.sqrt(np.array(variance_beta_list))
        # dataframe['SnpFractVarExpl'] = np.sqrt(fraction_variance_explained_beta[:,0])
        # dataframe['Mixing'] = np.zeros((len(sid))) + 0
        dataframe['Nullh2'] = np.zeros(test_snps.sid_count) + h2

    dataframe.sort_values(by="PValue", inplace=True)
    dataframe.index = np.arange(len(dataframe))


    if output_file_name is not None:
        dataframe.to_csv(output_file_name, sep="\t", index=False)


    return dataframe

def nLLevalx(self, delta):
        k = len(self.S)       # number of eigenvalues (and eigenvectors)
        N = self.y.shape[0]   # number of individuals
        D = self.UX.shape[1]  # number of covariates (usually includes a bias term)
        
        Sd = (self.S+delta)

        UXS = self.UX / Sd.reshape(-1,1)
        UyS = self.Uy / Sd

        XKX = UXS.T.dot(self.UX)
        XKy = UXS.T.dot(self.Uy)
        yKy = UyS.T.dot(self.Uy)

        logdetK = np.log(Sd).sum()
                
        [SxKx,UxKx]= np.linalg.eigh(XKX)
        #optionally regularize the beta weights by penalty
        i_pos = SxKx>1E-10
        beta = UxKx[:,i_pos].dot(UxKx[:,i_pos].T.dot(XKy)/SxKx[i_pos])

        r2 = yKy-XKy.dot(beta)

        sigma2 = r2 / (N)
        nLL =  0.5 * ( logdetK + N * ( np.log(2.0*np.pi*sigma2) + 1 ) )
        if delta is not None:
            h2 = 1.0/(delta+1)
        # This is a faster version of h2 * sigma2 * np.diag(LA.inv(XKX))
        # where h2*sigma2 is sigma2_g
        variance_beta = h2 * sigma2 * (UxKx[:,i_pos]/SxKx[i_pos] * UxKx[:,i_pos]).sum(-1)
        result = {
                'nLL':nLL,
                'sigma2':sigma2,
                'beta':beta,
                'variance_beta': variance_beta,
                'h2':h2,
                'REML':False,
                'a2':self.a2,
                'scale':1.0
                }

        assert np.all(np.isreal(nLL)), "nLL has an imaginary component, possibly due to constant covariates"
        if result['variance_beta'] is None:
            del result['variance_beta']
        logging.debug(result)
        return result


# !!!cmk similar to single_snp.py and single_snp_scale
def _create_dataframe(row_count):
    dataframe = pd.DataFrame(
        index=np.arange(row_count),
        columns=(
            "sid_index",
            "SNP",
            "Chr",
            "GenDist",
            "ChrPos",
            "PValue",
            "SnpWeight",
            "SnpWeightSE",
            "SnpFractVarExpl",
            "Mixing",
            "Nullh2",
        ),
    )
    #!!Is this the only way to set types in a dataframe?
    dataframe["sid_index"] = dataframe["sid_index"].astype(np.float)
    dataframe["Chr"] = dataframe["Chr"].astype(np.float)
    dataframe["GenDist"] = dataframe["GenDist"].astype(np.float)
    dataframe["ChrPos"] = dataframe["ChrPos"].astype(np.float)
    dataframe["PValue"] = dataframe["PValue"].astype(np.float)
    dataframe["SnpWeight"] = dataframe["SnpWeight"].astype(np.float)
    dataframe["SnpWeightSE"] = dataframe["SnpWeightSE"].astype(np.float)
    dataframe["SnpFractVarExpl"] = dataframe["SnpFractVarExpl"].astype(np.float)
    dataframe["Mixing"] = dataframe["Mixing"].astype(np.float)
    dataframe["Nullh2"] = dataframe["Nullh2"].astype(np.float)

    return dataframe
