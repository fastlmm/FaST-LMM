import logging
import pandas as pd
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import pysnptools.util as pstutil
from pysnptools.standardizer import Unit
from pysnptools.snpreader import SnpData
from pysnptools.pstreader import PstData
from pysnptools.eigenreader import EigenData
from pysnptools.util.mapreduce1 import map_reduce
from fastlmm.inference.fastlmm_predictor import (
    _pheno_fixup,
    _snps_fixup,
    _kernel_fixup,
)
from fastlmm.util.mingrid import minimize1D


# !!!LATER add warning here (and elsewhere) K0 or K1.sid_count < test_snps.sid_count,
#  might be a covar mix up.(but only if a SnpKernel
def single_snp_eigen(
    test_snps,
    pheno,
    K0_eigen,
    covar=None,  # !!!cmk covar_by_chrom=None, leave_out_one_chrom=True,
    output_file_name=None,
    log_delta=None,
    # !!!cmk cache_file=None, GB_goal=None, interact_with_snp=None,
    # !!!cmk runner=None, map_reduce_outer=True,
    # !!!cmk pvalue_threshold=None,
    # !!!cmk random_threshold=None,
    # !!!cmk random_seed = 0,
    # min_log_delta=-5,  # !!!cmk make this a range???
    # max_log_delta=10,
    # !!!cmk xp=None,
    find_delta_via_reml=True,
    test_via_reml=False,
    count_A1=None,
    runner=None,
):
    """cmk documentation"""
    # !!!LATER raise error if covar has NaN
    if output_file_name is not None:
        os.makedirs(Path(output_file_name).parent, exist_ok=True)

    # =========================
    # Figure out the data format for every input
    # =========================
    test_snps = _snps_fixup(test_snps, count_A1=count_A1)
    pheno = _pheno_fixup_and_check_missing(pheno, count_A1)
    covar = _pheno_fixup(covar, iid_if_none=pheno.iid, count_A1=count_A1)

    # =========================
    # Intersect and order individuals.
    # Make sure every K0_eigen individual
    # has data in test_snps, pheno, and covar.
    # =========================
    iid_count_before = K0_eigen.row_count
    test_snps, pheno, K0_eigen, covar = pstutil.intersect_apply(
        [test_snps, pheno, K0_eigen, covar]
    )
    assert (
        K0_eigen.row_count == iid_count_before
    ), "Must have test_snps, pheno, and covar data for each K0_eigen individual"

    # !!!cmk assert covar_by_chrom is None, "When 'leave_out_one_chrom' is False,
    #  'covar_by_chrom' must be None"
    # !!!cmk K0, K1, block_size = _set_block_size(K0, K1, mixing, GB_goal,
    #  force_full_rank, force_low_rank)

    # !!! cmk
    # if h2 is not None and not isinstance(h2, np.ndarray):
    #     h2 = np.repeat(h2, pheno.shape[1])

    # =========================
    # Read K0_eigen, covar, pheno into memory.
    # Next rotate covar and pheno.
    #
    # An "EigenReader" object includes both the vectors and values.
    # A Rotation object always includes rotated=eigenvectors * a.
    # If low-rank EigenReader, also includes double=a-eigenvectors*rotated.
    # =========================
    K0_eigen = K0_eigen.read(view_ok=True, order="A")
    covar = _covar_read_with_bias(covar)
    pheno = pheno.read(view_ok=True, order="A")

    covar_r = K0_eigen.rotate(covar)
    pheno_r = K0_eigen.rotate(pheno)

    def mapper_search(pheno_index):
        pheno_r_i = pheno_r[pheno_index]

        # =========================
        # Find the K0+delta I with the best likelihood.
        # A KdI object includes
        #   * Sd = eigenvalues + delta
        #   * is_low_rank (True/False)
        #   * logdet (depends on is_low_rank)
        # =========================
        K0_kdi_i = _find_best_kdi_as_needed(
            K0_eigen,
            covar,
            covar_r,
            pheno_r_i,
            use_reml=find_delta_via_reml,
            log_delta=log_delta,  # optional
        )
        return K0_kdi_i

    K0_kdi_list = map_reduce(
        range(pheno_r.col_count), mapper=mapper_search, runner=runner
    )
    K0_kdi = KdI.from_list(K0_kdi_list)
    del K0_kdi_list

    search_result_list = []
    for pheno_index in range(pheno_r.col_count):
        K0_kdi_i = K0_kdi[pheno_index]
        pheno_r_i = pheno_r[pheno_index]

        # =========================
        # Find A^T * K^-1 * B for covar and pheno.
        # Then find null likelihood for testing.
        # "AKB.from_rotated" works for both full and low-rank.
        # A AKB object includes
        #   * The AKB value
        #   * The KdI objected use to create it.
        # =========================
        covarKcovar_i, covarK_i = AKB.from_rotated_cmka(covar_r, K0_kdi_i, covar_r)
        phenoKpheno_i, _ = AKB.from_rotated_cmka(pheno_r_i, K0_kdi_i, pheno_r_i)
        covarKpheno_i, _ = AKB.from_rotated_cmka(
            covar_r, K0_kdi_i, pheno_r_i, aK=covarK_i
        )

        ll_null_i, beta_i, variance_beta_i = _loglikelihood(
            covar, phenoKpheno_i, covarKcovar_i, covarKpheno_i, use_reml=test_via_reml
        )

        search_result_list.append(
            {
                "covarKcovar": covarKcovar_i,
                "covarK": covarK_i,
                "covarKpheno": covarKpheno_i,
                "phenoKpheno": phenoKpheno_i,
                "ll_null": ll_null_i,
            }
        )

    # search_result_list = map_reduce(
    #    range(pheno_r.col_count), mapper=mapper_search, runner=runner
    # )

    # ==================================
    # X is the covariates (with bias) and one test SNP.
    # Create an X, XKX, and XKpheno where
    # the last part can be swapped for each test SNP.
    # ==================================
    cc = covar.sid_count  # number of covariates including bias
    # !!!cmk what if alt is not unique?
    xkx_sid = np.append(covar.sid, "alt")
    if test_via_reml:
        # Only need "X" for REML
        X = SnpData(
            val=np.full((covar.iid_count, len(xkx_sid)), fill_value=np.nan),
            iid=covar.iid,
            sid=xkx_sid,
        )
        X.val[:, :cc] = covar.val  # left
    else:
        X = None

    XKX_list = []
    XKpheno_list = []
    for pheno_index in range(pheno_r.col_count):
        search_result = search_result_list[pheno_index]
        K0_kdi_i = K0_kdi[pheno_index]
        covarKcovar_i = search_result["covarKcovar"]
        covarKpheno_i = search_result["covarKpheno"]

        pheno_col = pheno_r.col[pheno_index : pheno_index + 1]

        XKX_i = AKB.empty(row=xkx_sid, col=xkx_sid, kdi=K0_kdi_i)
        XKX_i[:cc, :cc] = covarKcovar_i  # upper left
        XKpheno_i = AKB.empty(xkx_sid, pheno_col, kdi=K0_kdi_i)
        XKpheno_i[:cc, :] = covarKpheno_i  # upper

        XKX_list.append(XKX_i)
        XKpheno_list.append(XKpheno_i)

        del search_result
        del K0_kdi_i
        del covarKcovar_i
        del covarKpheno_i
        del XKX_i
        del XKpheno_i
        del pheno_col

    # ==================================
    # Test SNPs in batches
    # ==================================
    # !!!cmk really do this in batches in different processes
    batch_size = 100  # !!!cmk const

    def mapper(sid_start):
        # ==================================
        # Read and standardize a batch of test SNPs. Then rotate.
        # Find A^T * K^-1 * B for covar & pheno vs. the batch
        # ==================================
        alt_batch = (
            test_snps[:, sid_start : sid_start + batch_size].read().standardize()
        )
        alt_batch_r = K0_eigen.rotate(alt_batch)

        alt_batchK_list = []
        covarKalt_batch_list = []
        alt_batchKy_list = []
        for pheno_index in range(pheno_r.col_count):
            search_result = search_result_list[pheno_index]
            K0_kdi_i = K0_kdi[pheno_index]
            covarK_i = search_result["covarK"]
            pheno_r_i = pheno_r[pheno_index]

            covarKalt_batch_i, _ = AKB.from_rotated_cmka(
                covar_r, K0_kdi_i, alt_batch_r, aK=covarK_i
            )
            alt_batchKy_i, alt_batchK_i = AKB.from_rotated_cmka(
                alt_batch_r, K0_kdi_i, pheno_r_i
            )

            alt_batchK_list.append(alt_batchK_i)
            covarKalt_batch_list.append(covarKalt_batch_i)
            alt_batchKy_list.append(alt_batchKy_i)

            del search_result
            del K0_kdi_i
            del covarK_i
            del covarKalt_batch_i
            del alt_batchKy_i
            del alt_batchK_i
            del pheno_r_i

        # ==================================
        # For each test SNP in the batch
        # ==================================
        result_list = []
        for i in range(alt_batch.sid_count):
            alt_r = alt_batch_r[i]

            if test_via_reml:  # Only need "X" for REML
                X.val[:, cc:] = alt_batch.val[:, i : i + 1]  # right

            for pheno_index in range(pheno_r.col_count):
                search_result = search_result_list[pheno_index]
                K0_kdi_i = K0_kdi[pheno_index]
                alt_batchK_i = alt_batchK_list[pheno_index]
                XKX_i = XKX_list[pheno_index]
                covarKalt_batch_i = covarKalt_batch_list[pheno_index]
                alt_batchKy_i = alt_batchKy_list[pheno_index]
                XKpheno_i = XKpheno_list[pheno_index]
                phenoKpheno_i = search_result["phenoKpheno"]
                ll_null_i = search_result["ll_null"]

                # ==================================
                # Find alt^T * K^-1 * alt for the test SNP.
                # Fill in last value of X, XKX and XKpheno
                # with the alt value.
                # ==================================
                altKalt_i, _ = AKB.from_rotated_cmka(
                    alt_r,
                    K0_kdi_i,
                    alt_r,
                    aK=alt_batchK_i[:, i : i + 1].read(view_ok=True),
                )

                XKX_i[:cc, cc:] = covarKalt_batch_i[:, i : i + 1]  # upper right
                XKX_i[cc:, :cc] = XKX_i[:cc, cc:].T  # lower left
                XKX_i[cc:, cc:] = altKalt_i  # lower right

                # !!!cmk rename alt_batchKy so no "y"?
                XKpheno_i[cc:, :] = alt_batchKy_i[i : i + 1, :]  # lower

                # ==================================
                # Find likelihood with test SNP and score.
                # ==================================
                # O(sid_count * (covar+1)^6)
                ll_alt_i, beta_i, variance_beta_i = _loglikelihood(
                    X, phenoKpheno_i, XKX_i, XKpheno_i, use_reml=test_via_reml
                )
                test_statistic_i = float(ll_alt_i - ll_null_i)
                result_list.append(
                    {
                        "PValue": stats.chi2.sf(2.0 * test_statistic_i, df=1),
                        "SnpWeight": beta_i,  #!!!cmk .val.reshape(-1),
                        "SnpWeightSE": np.sqrt(variance_beta_i),
                        # !!!cmk right name and place?
                        "Pheno": pheno_r.col[pheno_index],
                    }
                )

                del search_result
                del K0_kdi_i
                del alt_batchK_i
                del XKX_i
                del XKpheno_i
                del test_statistic_i
                del ll_alt_i
                del beta_i
                del variance_beta_i
                del altKalt_i

        dataframe = _create_dataframe().append(result_list, ignore_index=True)
        dataframe["sid_index"] = np.repeat(
            np.arange(sid_start, sid_start + alt_batch.sid_count), pheno_r.col_count
        )
        dataframe["SNP"] = np.repeat(alt_batch.sid, pheno_r.col_count)
        dataframe["Chr"] = np.repeat(alt_batch.pos[:, 0], pheno_r.col_count)
        dataframe["GenDist"] = np.repeat(alt_batch.pos[:, 1], pheno_r.col_count)
        dataframe["ChrPos"] = np.repeat(alt_batch.pos[:, 2], pheno_r.col_count)
        dataframe["Nullh2"] = np.tile(
            K0_kdi.h2.reshape(-1),
            alt_batch.sid_count,
        )
        # !!!cmk in lmmcov, but not lmm
        # dataframe['SnpFractVarExpl'] = np.sqrt(fraction_variance_explained_beta[:,0])
        # !!!cmk Feature not supported. could add "0"
        # dataframe['Mixing'] = np.zeros((len(sid))) + 0

        return dataframe

    dataframe_list = map_reduce(
        list(range(0, test_snps.sid_count, batch_size)), mapper=mapper, runner=runner
    )
    dataframe = pd.concat(dataframe_list)

    dataframe.sort_values(by="PValue", inplace=True)
    dataframe.index = np.arange(len(dataframe))

    if output_file_name is not None:
        dataframe.to_csv(output_file_name, sep="\t", index=False)

    return dataframe


def _pheno_fixup_and_check_missing(pheno, count_A1):
    pheno = _pheno_fixup(pheno, count_A1=count_A1).read()
    good_values_per_iid = (pheno.val == pheno.val).sum(axis=1)
    assert not np.any(
        (good_values_per_iid > 0) * (good_values_per_iid < pheno.sid_count)
    ), "With multiple phenotypes, an individual's values must either be all missing or have no missing."
    # !!!cmk multipheno
    # drop individuals with no good pheno values.
    pheno = pheno[good_values_per_iid > 0, :]

    assert pheno.sid_count >= 1, "Expect at least one phenotype"
    # !!!cmk assert pheno.sid_count == 1, "currently only have code for one pheno"

    return pheno


def _covar_read_with_bias(covar):
    covar_val0 = covar.read(view_ok=True, order="A").val
    covar_val1 = np.c_[
        covar_val0, np.ones((covar.iid_count, 1))
    ]  # view_ok because np.c_ will allocation new memory
    # !!!cmk what is "bias' is already used as column name
    covar_and_bias = SnpData(
        iid=covar.iid,
        sid=list(covar.sid) + ["bias"],
        val=covar_val1,
        name=f"{covar}&bias",
    )
    return covar_and_bias


# !!!cmk needs better name
class KdI:
    def __init__(self, hld, row, pheno, is_low_rank, logdet, Sd):
        self.h2, self.log_delta, self.delta = hld
        self.row = row
        self.pheno = pheno
        self.is_low_rank = is_low_rank
        self.logdet = logdet
        self.Sd = Sd

    @staticmethod
    def from_eigendata(eigendata, pheno, h2=None, log_delta=None, delta=None):
        hld = KdI._hld(h2, log_delta, delta)
        _, _, delta = hld
        logdet, Sd = eigendata.logdet(float(delta))

        return KdI(
            hld,
            row=eigendata.row,
            pheno=pheno,
            is_low_rank=eigendata.is_low_rank,
            logdet=logdet.reshape(logdet.shape[0], logdet.shape[1], 1),
            Sd=Sd.reshape(Sd.shape[0], Sd.shape[1], 1),
        )

    @staticmethod
    def _hld(h2=None, log_delta=None, delta=None):
        assert (
            sum([h2 is not None, log_delta is not None, delta is not None]) == 1
        ), "Exactly one of h2, etc should have a value"
        if h2 is not None:
            delta = 1.0 / h2 - 1.0
            log_delta = np.log(delta)
        elif log_delta is not None:
            log_delta = log_delta
            delta = np.exp(log_delta)
            h2 = 1.0 / (delta + 1)
        elif delta is not None:
            delta = delta
            log_delta = np.log(delta) if delta != 0 else None
            h2 = 1.0 / (delta + 1)
        else:
            assert False, "real assert"
        return np.array([h2]), np.array([log_delta]), np.array([delta])

    @staticmethod
    def from_list(kdi_list):
        assert len(kdi_list) > 0, "list must contain at least one item"
        h2 = np.r_[[kdi.h2 for kdi in kdi_list]]
        log_delta = np.r_[[kdi.log_delta for kdi in kdi_list]]
        delta = np.r_[[kdi.delta for kdi in kdi_list]]
        logdet = _stack([kdi.logdet for kdi in kdi_list])
        Sd = _stack([kdi.Sd for kdi in kdi_list])
        pheno = _stack([kdi.pheno for kdi in kdi_list])
        return KdI(
            (h2, log_delta, delta),
            row=kdi_list[0].row,
            pheno=pheno,
            is_low_rank=kdi_list[0].is_low_rank,
            logdet=logdet,
            Sd=Sd,
        )

    def __getitem__(self, pheno_index):
        h2 = self.h2[pheno_index : pheno_index + 1]
        log_delta = self.log_delta[pheno_index : pheno_index + 1]
        delta = self.delta[pheno_index : pheno_index + 1]
        logdet = self.logdet[..., pheno_index : pheno_index + 1]
        Sd = self.Sd[..., pheno_index : pheno_index + 1]
        pheno = self.pheno[pheno_index : pheno_index + 1]
        return KdI(
            (h2, log_delta, delta),
            row=self.row,
            pheno=pheno,
            is_low_rank=self.is_low_rank,
            logdet=logdet,
            Sd=Sd,
        )

    @property
    def row_count(self):
        return len(self.row)

    @property
    def pheno_count(self):
        return len(self.pheno)


# better way to stack the last dimension?
def _stack(array_list):
    pheno_count = len(array_list)
    assert pheno_count > 0, "cmk"
    shape = list(array_list[0].shape)
    assert shape[-1] == 1, "cmk"
    shape[-1] = pheno_count
    result = np.empty(shape=shape, dtype=array_list[0].dtype)
    if len(shape) > 1:
        result[...] = np.nan
        for pheno_index in range(pheno_count):
            result[..., pheno_index] = array_list[pheno_index][..., 0]
    else:
        for pheno_index in range(pheno_count):
            result[pheno_index] = array_list[pheno_index][0]
    return result


# !!!cmk move to PySnpTools
def AK_cmka(a_r, kdi, aK=None):
    if aK is None:
        assert kdi.pheno_count == 1, "cmk"
        return PstData(val=a_r.val / kdi.Sd[:, :, 0], row=a_r.row, col=a_r.col)
    else:
        return aK


# !!!cmk move to PySnpTools
class AKB(PstData):
    def __init__(self, val, row, col, kdi):
        super().__init__(val=val, row=row, col=col)
        self.kdi = kdi

    @staticmethod
    def from_rotated_cmka(a_r, kdi, b_r, aK=None):
        aK = AK_cmka(a_r, kdi, aK)

        val = aK.val.T.dot(b_r.val)
        if kdi.is_low_rank:
            val += a_r.double.val.T.dot(b_r.double.val) / kdi.delta
        result = AKB(val=val, row=a_r.col, col=b_r.col, kdi=kdi)
        return result, aK

    @staticmethod
    def empty(row, col, kdi):
        return AKB(
            val=np.full(shape=(len(row), len(col)), fill_value=np.NaN),
            row=row,
            col=col,
            kdi=kdi,
        )

    def __setitem__(self, key, value):
        # !!!cmk may want to check that the kdi's are equal
        self.val[key] = value.val

    def __getitem__(self, index):
        # !!!cmk fast enough?
        result0 = super(AKB, self).__getitem__(index).read(view_ok=True)
        result = AKB(val=result0.val, row=result0.row, col=result0.col, kdi=self.kdi)
        return result  # !!! cmk right type?

    @property
    def T(self):
        return AKB(val=self.val.T, row=self.col, col=self.row, kdi=self.kdi)


# !!!cmk change use_reml etc to 'use_reml'
def _find_h2(
    eigendata, X, X_r, pheno_r, use_reml, nGridH2=10, minH2=0.0, maxH2=0.99999
):
    # !!!cmk log delta is used here. Might be better to use findH2, but if so will need to normalized G so that its kdi's diagonal would sum to iid_count
    logging.info("searching for delta/h2/logdelta")

    resmin = [None]

    def f(x, resmin=resmin, **kwargs):
        # This kdi is Kg+delta I
        kdi = KdI.from_eigendata(eigendata, pheno=pheno_r.col, h2=x)
        # aKb is  a.T * kdi^-1 * b
        phenoKpheno, _ = AKB.from_rotated_cmka(pheno_r, kdi, pheno_r)
        XKX, XK = AKB.from_rotated_cmka(X_r, kdi, X_r)
        XKpheno, _ = AKB.from_rotated_cmka(X_r, kdi, pheno_r, aK=XK)

        nLL, _, _ = _loglikelihood(X, phenoKpheno, XKX, XKpheno, use_reml=use_reml)
        nLL = -float(nLL)  # !!!cmk
        if (resmin[0] is None) or (nLL < resmin[0]["nLL"]):
            resmin[0] = {"nLL": nLL, "h2": x}
        logging.debug(f"search\t{x}\t{nLL}")
        return nLL

    _ = minimize1D(f=f, nGrid=nGridH2, minval=0.00001, maxval=maxH2)
    return resmin[0]


def _eigen_from_akb(akb, keep_above=np.NINF):
    # !!!cmk check that square aKa not just aKb???
    w, v = np.linalg.eigh(akb.val)  # !!! cmk do SVD sometimes?
    eigen = EigenData(values=w, vectors=v, row=akb.row)
    if keep_above > np.NINF:
        eigen = eigen[:, eigen.values > keep_above].read(view_ok=True)
    return eigen


def _eigen_from_xtx(xtx):
    # !!!cmk check that square aKa not just aKb???
    w, v = np.linalg.eigh(xtx.val)  # !!! cmk do SVD sometimes?
    eigen = EigenData(values=w, vectors=v, row=xtx.row)
    return eigen


def _common_code(phenoKpheno, XKX, XKpheno):  # !!! cmk rename
    # _cmk_common_code(phenoKpheno, XKX, XKpheno)

    # !!!cmk may want to check that all three kdi's are equal
    # !!!cmk may want to check that all three kdi's are equal

    eigen_xkx = _eigen_from_akb(XKX, keep_above=1e-10)

    kd0 = KdI.from_eigendata(eigen_xkx, pheno=XKpheno.col, delta=0)
    XKpheno_r = eigen_xkx.rotate(XKpheno)
    XKphenoK = AK_cmka(XKpheno_r, kd0)
    beta = eigen_xkx.t_rotate(XKphenoK)
    r2 = PstData(
        val=phenoKpheno.val - XKpheno.val.T.dot(beta.val),
        row=phenoKpheno.row,
        col=phenoKpheno.col,
    )

    return r2, beta, eigen_xkx
    ####!!!cmk
    # beta0 = eigen_xkx.vectors.dot(
    #        eigen_xkx.rotate(XKpheno).val.reshape(-1) / eigen_xkx.values
    #    )

    # r0 = float(phenoKpheno.val - XKpheno.val.reshape(-1).dot(beta0))
    # r2 = float(r2.val)
    # beta = beta.val.reshape(-1)
    # assert np.all(np.equal(beta,beta0))
    # assert r0==r2
    # return r0, beta0, eigen_xkx #!!!cmk float(r2.val), beta.val.reshape(-1), eigen_xkx


def _loglikelihood(X, phenoKpheno, XKX, XKpheno, use_reml):
    if use_reml:
        nLL, beta = _loglikelihood_reml(X, phenoKpheno, XKX, XKpheno)
        return nLL, beta, np.nan
    else:
        return _loglikelihood_ml(phenoKpheno, XKX, XKpheno)


def _loglikelihood_reml(X, phenoKpheno, XKX, XKpheno):
    kdi = phenoKpheno.kdi  # !!!cmk may want to check that all three kdi's are equal

    r2, beta, eigen_xkx = _common_code(phenoKpheno, XKX, XKpheno)

    # !!!cmk isn't this a kernel?
    XX = PstData(val=X.val.T.dot(X.val), row=X.sid, col=X.sid)
    eigen_xx = _eigen_from_xtx(XX)
    logdetXX, _ = eigen_xx.logdet()

    logdetXKX, _ = eigen_xkx.logdet()
    X_row_less_col = X.row_count - X.col_count
    sigma2 = float(r2.val) / X_row_less_col
    nLL = 0.5 * (
        kdi.logdet
        + logdetXKX
        - logdetXX
        + X_row_less_col * (np.log(2.0 * np.pi * sigma2) + 1)
    )

    assert np.all(
        np.isreal(nLL)
    ), "nLL has an imaginary component, possibly due to constant covariates"
    # !!!cmk which is negative loglikelihood and which is LL?
    return -nLL, beta


def _loglikelihood_ml(phenoKpheno, XKX, XKpheno):
    r2, beta, eigen_xkx = _common_code(phenoKpheno, XKX, XKpheno)
    kdi = phenoKpheno.kdi  # !!!cmk may want to check that all three kdi's are equal
    sigma2 = float(r2.val) / kdi.row_count
    nLL = 0.5 * (kdi.logdet + kdi.row_count * (np.log(2.0 * np.pi * sigma2) + 1))
    assert np.all(
        np.isreal(nLL)
    ), "nLL has an imaginary component, possibly due to constant covariates"
    variance_beta = (
        kdi.h2
        * sigma2
        * (eigen_xkx.vectors / eigen_xkx.values * eigen_xkx.vectors).sum(-1)
    )
    # !!!cmk which is negative loglikelihood and which is LL?
    return -nLL, beta, variance_beta


# Returns a kdi that is the original Kg + delta I
def _find_best_kdi_as_needed(
    eigendata, covar, covar_r, pheno_r, use_reml, log_delta=None
):
    if log_delta is None:
        # cmk As per the paper, we optimized delta with use_reml=True, but
        # cmk we will later optimize beta and find log likelihood with ML (use_reml=False)
        h2 = _find_h2(
            eigendata, covar, covar_r, pheno_r, use_reml=use_reml, minH2=0.00001
        )["h2"]
        return KdI.from_eigendata(eigendata, pheno=pheno_r.col, h2=h2)
    else:
        # !!!cmk internal/external doesn't matter if full rank, right???
        return KdI.from_eigendata(eigendata, pheno=pheno_r.col, log_delta=log_delta)


# !!!cmk similar to single_snp.py and single_snp_scale
def _create_dataframe():
    # https://stackoverflow.com/questions/21197774/assign-pandas-dataframe-column-dtypes
    dataframe = pd.DataFrame(
        np.empty(
            (0,),
            dtype=[
                ("sid_index", np.float),
                ("SNP", "S"),
                ("Chr", np.float),
                ("GenDist", np.float),
                ("ChrPos", np.float),
                ("PValue", np.float),
                ("SnpWeight", np.float),
                ("SnpWeightSE", np.float),
                ("SnpFractVarExpl", np.float),
                ("Mixing", np.float),
                ("Nullh2", np.float),
            ],
        )
    )
    return dataframe


# !!!cmk move to pysnptools
def eigen_from_kernel(K0, kernel_standardizer, count_A1=None):
    """!!!cmk documentation"""
    # !!!cmk could offer a low-memory path that uses memmapped files
    from pysnptools.kernelreader import SnpKernel
    from pysnptools.kernelstandardizer import Identity as KS_Identity

    assert K0 is not None
    K0 = _kernel_fixup(K0, iid_if_none=None, standardizer=Unit(), count_A1=count_A1)
    assert K0.iid0 is K0.iid1, "Expect K0 to be square"

    if isinstance(
        K0, SnpKernel
    ):  # !!!make eigen creation a method on all kernel readers
        assert isinstance(
            kernel_standardizer, KS_Identity
        ), "cmk need code for other kernel standardizers"
        vectors, sqrt_values, _ = np.linalg.svd(
            K0.snpreader.read().standardize(K0.standardizer).val, full_matrices=False
        )
        if np.any(sqrt_values < -0.1):
            logging.warning("kernel contains a negative Eigenvalue")
        eigen = EigenData(values=sqrt_values * sqrt_values, vectors=vectors, row=K0.iid)
    else:
        # !!!cmk understand _read_kernel, _read_with_standardizing

        K0 = K0._read_with_standardizing(
            kernel_standardizer=kernel_standardizer,
            to_kerneldata=True,
            return_trained=False,
        )
        # !!! cmk ??? pass in a new argument, the kernel_standardizer(???)
        logging.debug("About to eigh")
        w, v = np.linalg.eigh(K0.val)  # !!! cmk do SVD sometimes?
        logging.debug("Done with to eigh")
        if np.any(w < -0.1):
            logging.warning(
                "kernel contains a negative Eigenvalue"
            )  # !!!cmk this shouldn't happen with a RRM, right?
        # !!!cmk remove very small eigenvalues
        # !!!cmk remove very small eigenvalues in a way that doesn't require a memcopy?
        eigen = EigenData(values=w, vectors=v, iid=K0.iid)
        # eigen.vectors[:,eigen.values<.0001]=0.0
        # eigen.values[eigen.values<.0001]=0.0
        # eigen = eigen[:,eigen.values >= .0001] # !!!cmk const
    return eigen
