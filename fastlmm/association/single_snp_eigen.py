import logging
import pandas as pd
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import pysnptools.util as pstutil
from unittest.mock import patch
from pysnptools.standardizer import Unit
from pysnptools.snpreader import SnpData
from pysnptools.eigenreader import EigenData
from fastlmm.inference.fastlmm_predictor import (
    _pheno_fixup,
    _snps_fixup,
    _kernel_fixup,
)
from fastlmm.inference import LMM


#!!!cmk move to pysnptools
def eigen_from_kernel(K, kernel_standardizer, count_A1=None):
    """!!!cmk documentation"""
    # !!!cmk could offer a low-memory path that uses memmapped files
    from pysnptools.kernelreader import SnpKernel
    from pysnptools.kernelstandardizer import Identity as KS_Identity

    assert K is not None
    K = _kernel_fixup(K, iid_if_none=None, standardizer=Unit(), count_A1=count_A1)
    assert K.iid0 is K.iid1, "Expect K to be square"

    if isinstance(K,SnpKernel): #!!!make eigen creation a method on all kernel readers
        assert isinstance(kernel_standardizer, KS_Identity), "cmk need code for other kernel standardizers"
        vectors,sqrt_values,_ = np.linalg.svd(K.snpreader.read().standardize(K.standardizer).val, full_matrices=False)
        if np.any(sqrt_values < -0.1):
            logging.warning("kernel contains a negative Eigenvalue")
        eigen = EigenData(values=sqrt_values*sqrt_values, vectors=vectors, iid=K.iid)
    else:
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

        # view_ok because this code already did a fresh read to look for any
        #  missing values
        eigendata = eigenreader.read(view_ok=True, order="A")

        covar_val0 = covar.read(view_ok=True, order="A").val
        covar_val1 = np.c_[covar_val0, np.ones((test_snps.iid_count, 1))]  # view_ok because np.c_ will allocation new memory
        #!!!cmk what is "bias' is already used as column name
        covar_and_bias = SnpData(iid=covar.iid, sid=list(covar.sid)+["bias"], val=covar_val1, name=f"{covar}&bias")
        #============
        # iid_count x eid_count  *  iid_count x covar => eid_count * covar
        # O(iid_count x eid_count x covar)
        #=============
        covar_rotated = eigendata.rotate(covar_and_bias)


        assert pheno.sid_count >= 1, "Expect at least one phenotype"
        assert pheno.sid_count == 1, "currently only have code for one pheno"
        # view_ok because this code already did a fresh read to look for any
        # missing values
        #============
        # iid_count x eid_count  *  iid_count x pheno_count => eid_count * pheno_count
        # O(iid_count x eid_count x pheno_count)
        #=============
        # !!! cmk with multipheno is it going to be O(covar*covar*y)???
        y_rotated = eigendata.rotate(pheno.read(view_ok=True, order="A"))

        if log_delta is None:
            # !!!cmk log delta is used here. Might be better to use findH2, but if so will need to normalized G so that its K's diagonal would sum to iid_count

            logging.info("searching for delta/h2/logdelta")
            result = _find_h2(eigendata, covar_rotated, y_rotated, REML=fit_log_delta_via_reml, minH2=0.00001)
            h2 = result["h2"]
            delta = 1.0/h2-1.0
            log_delta = np.log(delta)
            # cmk As per the paper, we optimized delta with REML=True, but
            # cmk we will later optimize beta and find log likelihood with ML (REML=False)
            # !!! cmk so sid_count need not be given if doing full rank, right?
        else:
            # !!!cmk internal/external doesn't matter if full rank, right???
            delta = np.exp(log_delta)
            h2 = 1.0/(delta+1)
        logging.info("delta={0}".format(delta))
        logging.info("log_delta={0}".format(log_delta))

        K = Kthing(eigendata, delta)

        yKy         = AKB(y_rotated,     K, y_rotated)
        covarKcovar = AKB(covar_rotated, K, covar_rotated)
        covarKy     = AKB(covar_rotated, K, y_rotated, a_by_Sd=covarKcovar.a_by_Sd)

        assert not test_via_reml # !!!cmk
        ll_null, beta, variance_beta = _loglikelihood_ml(eigenreader.iid_count, K.logdet, h2, yKy.aKb, covarKcovar.aKb, covarKy.aKb)


        # !!!cmk really do this in batches in different processes
        dataframe = _create_dataframe(test_snps.sid_count)
        pvalue_list = []
        beta_list = [] 
        variance_beta_list = []

        cc = covar_and_bias.sid_count # number of covariates including bias

        XKX = np.full(shape=(cc+1,cc+1),fill_value=np.NaN)
        XKX[:cc,:cc] = covarKcovar.aKb

        XKy = np.full(shape=(cc+1,pheno.sid_count),fill_value=np.NaN)
        XKy[:cc,:] = covarKy.aKb
         
        batch_size = 1000 #!!!cmk const
        for sid_start in range(0,test_snps.sid_count,batch_size):
            sid_end = np.min([sid_start+batch_size,test_snps.sid_count])

            snps_batch = test_snps[:, sid_start:sid_end].read().standardize()
            # !!!cmk should biobank precompute this?
            alt_batch_rotated = eigendata.rotate(snps_batch)

            covarKalt_batch = AKB(covar_rotated,     K, alt_batch_rotated, a_by_Sd=covarKcovar.a_by_Sd)
            alt_batchKy     = AKB(alt_batch_rotated, K, y_rotated)

            for i in range(sid_end-sid_start):

                alt_rotated = alt_batch_rotated[i]
                altKy = alt_batchKy[i]
                altKalt = AKB(alt_rotated, K, alt_rotated, a_by_Sd=altKy.a_by_Sd)

                XKX[:cc,cc:] = covarKalt_batch.aKb[:,i:i+1]
                XKX[cc:,:cc] = XKX[:cc,cc:].T
                XKX[cc:,cc:] = altKalt.aKb

                XKy[cc:,:]   = altKy.aKb

                # O(sid_count * (covar+1)^6)
                ll_alt, beta, variance_beta = _loglikelihood_ml(eigenreader.iid_count, K.logdet, h2, yKy.aKb, XKX, XKy)
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

class Kthing:
    def __init__(self, eigendata, delta):
        self.delta = delta
        self.Sd = (eigendata.values+delta).reshape(-1,1)
        self.logdet = np.log(self.Sd).sum()
        self.is_low_rank = eigendata.is_low_rank
        if eigendata.is_low_rank: # !!!cmk test this
            self.logdet += (eigendata.iid_count - eigendata.eid_count) * np.log(delta)


class AKB:
    def __init__(self, a_rotated, K, b_rotated, a_by_Sd=None):
        if a_by_Sd is None:
            # "reshape" lets it broadcast
            self.a_by_Sd = a_rotated.rotated.val / K.Sd
        else:
            self.a_by_Sd = a_by_Sd

        self.aKb = self.a_by_Sd.T.dot(b_rotated.rotated.val)
        if K.is_low_rank:
            self.aKb += a_rotated.double_rotated.val.T.dot(b_rotated.double_rotated.val)/K.delta

    def __getitem__(self, index):
        akbi = AKB.__new__(AKB)
        akbi.a_by_Sd = self.a_by_Sd[:,index:index+1]
        akbi.aKb = self.aKb[index:index+1,:]
        return akbi



def _find_h2(eigendata, X_rotated, y_rotated, REML, minH2=0.00001):
    #!!!cmk expect one pass per y column
    lmm = LMM()
    lmm.S = eigendata.values
    lmm.U = eigendata.vectors
    lmm.UX = X_rotated.rotated.val # !!!cmk This is precomputed because we'll be dividing it by (eigenvalues+delta) over and over again
    lmm.UUX = X_rotated.double_rotated.val if X_rotated.double_rotated is not None else None
    lmm.Uy = y_rotated.rotated.val[:,0]  # !!!cmk precomputed for the same reason
    lmm.UUy = y_rotated.double_rotated.val[:,0] if y_rotated.double_rotated is not None else None

    return lmm.findH2(REML=REML, minH2=0.00001)

#!!!cmk add __loglikelihood_ml
def _loglikelihood_ml(iid_count, logdetK, h2, yKy, XKX, XKy):
    yKy = float(yKy) # !!!cmk assuming one pheno
    XKy = XKy.reshape(-1) # cmk should be 2-D to support multiple phenos
    # Must do one test at a time
    SxKx,UxKx= np.linalg.eigh(XKX)
    # Remove tiny eigenvectors
    i_pos = SxKx>1E-10
    UxKx = UxKx[:,i_pos]
    SxKx = SxKx[i_pos]

    beta = UxKx.dot(UxKx.T.dot(XKy)/SxKx)
    r2 = yKy-XKy.dot(beta)
    sigma2 = r2 / iid_count
    nLL =  0.5 * ( logdetK + iid_count * ( np.log(2.0*np.pi*sigma2) + 1 ))
    assert np.all(np.isreal(nLL)), "nLL has an imaginary component, possibly due to constant covariates"
    variance_beta = h2 * sigma2 * (UxKx/SxKx * UxKx).sum(-1)
    return -nLL, beta, variance_beta




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
