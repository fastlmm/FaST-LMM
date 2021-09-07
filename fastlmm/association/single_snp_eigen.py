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
        y = xp.asarray(pheno.read(view_ok=True, order="A").val)
        eigendata = eigenreader.read(view_ok=True, order="A")

        #============
        # iid_count x eid_count  *  iid_count x covar => eid_count * covar
        # O(iid_count x eid_count x covar)
        #=============
        rotated_covar_pair = eigendata.rotate(covar_val)

        #============
        # iid_count x eid_count  *  iid_count x pheno_count => eid_count * pheno_count
        # O(iid_count x eid_count x pheno_count)
        #=============
        # !!! cmk with multipheno is it going to be O(covar*covar*y)???
        rotated_y_pair = eigendata.rotate(y[:,0])

        if log_delta is None:
            # !!!cmk log delta is used here. Might be better to use findH2, but if so will need to normalized G so that its K's diagonal would sum to iid_count

            logging.info("searching for delta/h2/logdelta")
            result = _find_h2(eigendata, rotated_covar_pair, rotated_y_pair, REML=fit_log_delta_via_reml, minH2=0.00001)
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

        # As per the paper, we previously optimized delta with REML=True, but
        # we optimize beta and find loglikelihood with ML (REML=False)
        # !!! cmk if test_via_reml == fit_log_delta_via_reml this could be skipped
        assert not test_via_reml # !!!cmk

        

        # pheno_count x eid_count * eid_count x pheno_count -> pheno_count x pheno_count, O(pheno^2*eid_count)
        # covar x eid_count * eid_count x covar  -> covar x covar, O(covar^2*eid_count)
        # covar x eid_count * eid_count x pheno_count -> covar x pheno_count, O(covar*pheno*eid_count)
        yKy, Sd, logdetK, UyS = _AKB(eigendata, rotated_y_pair, delta, rotated_y_pair, Sd=None, logdetK=None, a_by_Sd=None)
        covarKcovar, _, _, covarS = _AKB(eigendata, rotated_covar_pair, delta, rotated_covar_pair, Sd=Sd.reshape(-1,1), logdetK=logdetK, a_by_Sd=None) # cmk "reshape" lets it broadcast
        covarKy, _, _, _ = _AKB(eigendata, rotated_covar_pair, delta, rotated_y_pair, Sd=Sd,  logdetK=logdetK, a_by_Sd=covarS)

        covarKy = covarKy.reshape(-1,1) # cmk make 2-d now so eaiser to support multiphenotype later

        ll_null, beta, variance_beta = ll_eval(eigenreader.iid_count, logdetK, h2, yKy, covarKcovar, covarKy)


        dataframe = _create_dataframe(test_snps.sid_count)

        # !!!cmk really do this in batches in different processes

        pvalue_list = []
        beta_list = [] 
        variance_beta_list = []

        ncov = len(covarKcovar) # number of covariates (usually includes a bias term)
        npheno = covarKy.shape[1] # number of phenotypes
        XKX = np.full(shape=(ncov+1,ncov+1),fill_value=np.NaN)
        XKX[:ncov,:ncov] = covarKcovar
        XKy = np.full(shape=(ncov+1,npheno),fill_value=np.NaN)
        XKy[:ncov,:] = covarKy

        batch_size = 1000 #!!!cmk const
        for sid_start in range(0,test_snps.sid_count,batch_size):
            sid_end = np.min([sid_start+batch_size,test_snps.sid_count])
            snps_batch = test_snps[:, sid_start:sid_end].read().standardize()

            # eid_count x iid_count * iid_count x sid_count -> eid_count x sid_count, O(eid_count * iid_count * sid_count)
            # !!!cmk should biobank precompute this?
            alt_batch_pair = eigendata.rotate(snps_batch.val)

            # covar x eid_count * eid_count x sid_count -> covar * sid_count,  O(covar * eid_count * sid_count)
            covarSalt_batch, _, _, _ = _AKB(eigendata, rotated_covar_pair, delta, alt_batch_pair, Sd=Sd, logdetK=logdetK, a_by_Sd=covarS)
            #covarSalt_batch = covarS.T.dot(alt_batch_pair[0])


            for sid_index in range(sid_start,sid_end):
                XKX[:ncov,ncov:] = covarSalt_batch[:,sid_index-sid_start:sid_index-sid_start+1]
                XKX[ncov:,:ncov] = XKX[:ncov,ncov:].T

                alt_pair = (alt_batch_pair[0][:,sid_index-sid_start:sid_index-sid_start+1],
                            alt_batch_pair[1][:,sid_index-sid_start:sid_index-sid_start+1] if alt_batch_pair[1] is not None else None)


                # sid_count x eid_count * eid_count x pheno_count -> sid_count x pheno_count, O(sid_count * eid_count * pheno_count)
                altKalt,_,_, UaltS = _AKB(eigendata, alt_pair, delta, alt_pair, Sd=Sd.reshape(-1,1), logdetK=logdetK, a_by_Sd=None)
                XKX[ncov:,ncov:] = altKalt

                ## sid_count x eid_count * eid_count x pheno_count -> sid_count x pheno_count, O(sid_count * eid_count * pheno_count)
                altKy,_,_,_ = _AKB(eigendata, alt_pair, delta, rotated_y_pair, Sd=Sd.reshape(-1,1), logdetK=logdetK, a_by_Sd=UaltS)
                XKy[ncov:,:] = altKy

                # O(sid_count * (covar+1)^6)
                ll_alt, beta, variance_beta = ll_eval(eigenreader.iid_count, logdetK, h2, yKy, XKX, XKy)
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

# !!!cmk what is this mathematically? What's a better name
def _AKB(eigendata, rotated_pair_a, delta, rotated_pair_b, Sd=None, logdetK=None, a_by_Sd=None):
    if Sd is None:
        Sd = eigendata.values+delta

    if logdetK is None:
        logdetK = np.log(Sd).sum()
        if eigendata.is_low_rank: # !!!cmk test this
            logdetK += (eigendata.iid_count - eigendata.eid_count) * np.log(delta)

    if a_by_Sd is None:
        a_by_Sd = rotated_pair_a[0] / Sd

    aKb = a_by_Sd.T.dot(rotated_pair_b[0])
    if eigendata.is_low_rank:
        aKb += rotated_pair_a[1].T.dot(rotated_pair_b[1])/delta # !!!cmk test this

    return aKb, Sd, logdetK, a_by_Sd


def _find_h2(eigendata, rotated_X_pair, rotated_y_pair, REML, minH2=0.00001):
    #!!!cmk expect one pass per y column
    lmm = LMM()
    lmm.S = eigendata.values
    lmm.U = eigendata.vectors
    lmm.UX = rotated_X_pair[0] # !!!cmk This is precomputed because we'll be dividing it by (eigenvalues+delta) over and over again
    lmm.UUX = rotated_X_pair[1]
    lmm.Uy = rotated_y_pair[0]  # !!!cmk precomputed for the same reason
    lmm.UUy = rotated_y_pair[1]

    return lmm.findH2(REML=REML, minH2=0.00001)

#!!!cmk add _ll_eval
def ll_eval(iid_count, logdetK, h2, yKy, XKX, XKy):
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
