import logging
import pandas as pd
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
from einops import rearrange
import pysnptools.util as pstutil
from pysnptools.standardizer import Unit
from pysnptools.snpreader import SnpData
from pysnptools.pstreader import PstData
from pysnptools.eigenreader import EigenData
from pysnptools.eigenreader.eigendata import Rotation
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
    pheno_r = K0_eigen.rotate(pheno, is_diagonal=True)

    # =========================
    # For each phenotype, in parallel, ...
    # Find the K0+delta I with the best likelihood.
    # A KdI object includes
    #   * Sd = eigenvalues + delta
    #   * is_low_rank (True/False)
    #   * logdet
    # =========================
    def mapper_search(pheno_index):
        return _find_best_kdi_as_needed(
            K0_eigen,
            covar,
            covar_r,
            pheno_r[pheno_index],
            use_reml=find_delta_via_reml,
            log_delta=log_delta,  # optional
        )

    K0_kdi = map_reduce(
        range(pheno_r.col_count),
        mapper=mapper_search,
        reducer=KdI.from_sequence,
        runner=runner,
    )

    # =========================
    # For each pheno (as the last dimension of the matrix) ...
    # Find A^T * K^-1 * B for covar and pheno.
    # Then find null likelihood for testing.
    # "AKB.from_rotations" works for both full and low-rank.
    # A AKB object includes
    #   * The AKB value
    #   * The KdI objected use to create it.
    # =========================

    covarKcovar, covarK = AKB.from_rotations(covar_r, K0_kdi, covar_r)
    phenoKpheno, _ = AKB.from_rotations(pheno_r, K0_kdi, pheno_r)
    covarKpheno, _ = AKB.from_rotations(covar_r, K0_kdi, pheno_r, aK=covarK)
    ll_null, _beta, _variance_beta = _loglikelihood(
        covar, phenoKpheno, covarKcovar, covarKpheno, use_reml=test_via_reml
    )

    # ==================================
    # X is the covariates (with bias) and one test SNP.
    # For each pheno (as the last dimension of the matrix) ...
    # Create an XKX, and XKpheno where
    # the last part can be swapped for each test SNP.
    # ==================================
    cc = covar.sid_count  # number of covariates including bias
    # !!!cmk what if alt is not unique?
    xkx_sid = np.append(covar.sid, "alt")
    if test_via_reml:
        # Only need explicit "X" for REML
        X = SnpData(
            val=np.full((covar.iid_count, len(xkx_sid)), fill_value=np.nan),
            iid=covar.iid,
            sid=xkx_sid,
        )
        X.val[:, :cc] = covar.val  # left
    else:
        X = None

    XKX = AKB.empty(row=xkx_sid, col=xkx_sid, kdi=K0_kdi)
    XKX[:cc, :cc] = covarKcovar  # upper left
    XKpheno = AKB.empty(xkx_sid, Rotation.diagonal_name, kdi=K0_kdi)
    XKpheno[:cc, :] = covarKpheno  # upper

    # ==================================
    # Test SNPs in batches
    # ==================================
    batch_size = 100  # !!!cmk const

    def mapper(sid_start):
        # ==================================
        # Read and standardize a batch of test SNPs. Then rotate.
        # For each pheno (as the last dimension in the matrix) ...
        # Find A^T * K^-1 * B for covar & pheno vs. the batch
        # ==================================
        alt_batch = (
            test_snps[:, sid_start : sid_start + batch_size].read().standardize()
        )
        alt_batch_r = K0_eigen.rotate(alt_batch)
        covarKalt_batch, _ = AKB.from_rotations(covar_r, K0_kdi, alt_batch_r, aK=covarK)
        alt_batchKpheno, alt_batchK = AKB.from_rotations(alt_batch_r, K0_kdi, pheno_r)

        # ==================================
        # For each test SNP in the batch
        # ==================================
        result_list = []
        for i in range(alt_batch.sid_count):
            alt_r = alt_batch_r[i]

            # ==================================
            # For each pheno (as the last dimension in the matrix) ...
            # Find alt^T * K^-1 * alt for the test SNP.
            # Fill in last value of X, XKX and XKpheno
            # with the alt value.
            # ==================================
            altKalt, _ = AKB.from_rotations(
                alt_r, K0_kdi, alt_r, aK=alt_batchK[:, i : i + 1, :]
            )

            XKX[:cc, cc:] = covarKalt_batch[:, i : i + 1, :]  # upper right
            XKX[cc:, :cc] = XKX[:cc, cc:, :].T  # lower left
            XKX[cc:, cc:] = altKalt[:, :, :]  # lower right

            XKpheno[cc:, :] = alt_batchKpheno[i : i + 1, :, :]  # lower

            if test_via_reml:  # Only need "X" for REML
                X.val[:, cc:] = alt_batch.val[:, i : i + 1]  # right

            # ==================================
            # Find likelihood with test SNP and score.
            # ==================================
            # O(sid_count * (covar+1)^6)
            ll_alt, beta, variance_beta = _loglikelihood(
                X, phenoKpheno, XKX, XKpheno, use_reml=test_via_reml
            )

            test_statistic = ll_alt - ll_null

            for pheno_index in range(pheno_r.col_count):
                result_list.append(
                    {
                        "PValue": stats.chi2.sf(
                            2.0 * test_statistic[pheno_index], df=1
                        ),
                        "SnpWeight": beta.val[:, pheno_index],  #!!!cmk
                        "SnpWeightSE": np.sqrt(variance_beta[:, pheno_index])
                        if variance_beta is not None
                        else None,
                        # !!!cmk right name and place?
                        "Pheno": pheno_r.col[pheno_index],
                    }
                )

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

        assert len(self.h2.shape) == 1, "Expect h2, etc, to be a 1-D array"
        assert len(logdet.shape) == 1, "Expect logdet to be a 1-D array"
        assert len(Sd.shape) == 2, "Expect Sd to be a 2-D array"

        self.row = row
        self.pheno = pheno
        self.is_low_rank = is_low_rank
        self.logdet = logdet
        self.Sd = Sd

    @staticmethod
    def from_eigendata(eigendata, pheno, h2=None, log_delta=None, delta=None):
        hld = KdI._hld(h2, log_delta, delta)
        _, _, delta = hld
        assert len(delta) == 1, "Expect delta, etc, to be a 1-D array of one value"

        logdet, Sd = eigendata.logdet(float(delta))

        return KdI(
            hld,
            row=eigendata.row,
            pheno=pheno,
            is_low_rank=eigendata.is_low_rank,
            logdet=np.array([logdet]),
            Sd=rearrange(Sd, "i->i 1"),
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
    def from_sequence(kdi_sequence):
        kdi_list = list(kdi_sequence)
        assert len(kdi_list) > 0, "list must contain at least one item"

        h2 = rearrange([kdi.h2 for kdi in kdi_list], "pheno 1 -> pheno")
        log_delta = rearrange([kdi.log_delta for kdi in kdi_list], "pheno 1 -> pheno")
        delta = rearrange([kdi.delta for kdi in kdi_list], "pheno 1 -> pheno")
        logdet = rearrange([kdi.logdet for kdi in kdi_list], "pheno 1 -> pheno")
        Sd = rearrange([kdi.Sd for kdi in kdi_list], "pheno i 1 -> i pheno")
        pheno = rearrange([kdi.pheno for kdi in kdi_list], "pheno 1 -> pheno")

        return KdI(
            (h2, log_delta, delta),
            row=kdi_list[0].row,
            pheno=pheno,
            is_low_rank=kdi_list[0].is_low_rank,
            logdet=logdet,
            Sd=Sd,
        )

    def __getitem__(self, pheno_index):
        if pheno_index == slice(None, None, None):
            return self
        if isinstance(pheno_index, slice):
            assert pheno_index.step is None
            assert pheno_index.start + 1 == pheno_index.stop
            pheno_index = pheno_index.start
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


# !!!cmk move to PySnpTools
class AK(PstData):
    def __init__(self, val, row, col, pheno):
        super().__init__(val=val, row=row, col=col)
        self.pheno = pheno

    @staticmethod
    def from_rotation(a_r, kdi, aK=None):
        if a_r.is_diagonal:
            val = a_r.val[:, np.newaxis, :] / kdi.Sd[:, None, :]
        else:
            val = a_r.val[:, :, np.newaxis] / kdi.Sd[:, None, :]

        return AK(val=val, row=a_r.row, col=a_r.diagonal_or_col, pheno=kdi.pheno)

    def __getitem__(self, index):
        val = self.val[index]
        return AK(
            val=val,
            row=self.row[index[0]],
            col=self.col[index[1]],
            pheno=self.pheno[index[2]],
        )


# !!!cmk move to PySnpTools
class AKB(PstData):
    def __init__(self, val, row, col, kdi):
        super().__init__(val=val, row=row, col=col)
        self.pheno = kdi.pheno
        self.kdi = kdi

    @staticmethod
    def from_rotations(a_r, kdi, b_r, aK=None):
        aK = AK.from_rotation(a_r, kdi, aK)

        ein_a, axis_a = a_r.ein("a")
        ein_b, axis_b = b_r.ein("b")
        ein_abd = Rotation.ein_cat(ein_a, ein_b, "d")
        axis_abd = (axis_a, axis_b, np.s_[:])
        ein_str0 = f"iad,i{ein_b}->{ein_abd}"
        val = np.einsum(ein_str0, aK.val, b_r.val)[axis_abd]

        if kdi.is_low_rank:
            ein_ab = Rotation.ein_cat(ein_a, ein_b)
            last_axis = np.s_[:] if a_r.is_diagonal or b_r.is_diagonal else np.newaxis
            axis_ab = (axis_a, axis_b, last_axis)
            ein_str1 = f"i{ein_a},i{ein_b}->{ein_ab}"
            val += np.einsum(ein_str1, a_r.double.val, b_r.double.val)[
                axis_ab
            ] / kdi.delta.reshape(-1)

        result = AKB(val=val, row=a_r.diagonal_or_col, col=b_r.diagonal_or_col, kdi=kdi)
        return result, aK

    @staticmethod
    def empty(row, col, kdi):
        return AKB(
            val=np.full(shape=(len(row), len(col), len(kdi.pheno)), fill_value=np.NaN),
            row=row,
            col=col,
            kdi=kdi,
        )

    def __setitem__(self, key, value):
        # !!!cmk may want to check that the kdi's are equal
        self.val[key] = value.val

    def __getitem__(self, index):
        val = self.val[index]
        return AKB(
            val=val,
            row=self.row[index[0]],
            col=self.col[index[1]],
            kdi=self.kdi[index[2]],
        )

    @property
    def T(self):
        return AKB(
            val=np.moveaxis(self.val, 0, 1), row=self.col, col=self.row, kdi=self.kdi
        )


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
        phenoKpheno, _ = AKB.from_rotations(pheno_r, kdi, pheno_r)
        XKX, XK = AKB.from_rotations(X_r, kdi, X_r)
        XKpheno, _ = AKB.from_rotations(X_r, kdi, pheno_r, aK=XK)

        nLL, _, _ = _loglikelihood(X, phenoKpheno, XKX, XKpheno, use_reml=use_reml)
        nLL = -float(nLL)  # !!!cmk
        if (resmin[0] is None) or (nLL < resmin[0]["nLL"]):
            resmin[0] = {"nLL": nLL, "h2": x}
        logging.debug(f"search\t{x}\t{nLL}")
        return nLL

    _ = minimize1D(f=f, nGrid=nGridH2, minval=0.00001, maxval=maxH2)
    return resmin[0]


def _common_code(yKy, XKX, XKy):  # !!! cmk rename
    # !!!cmk may want to check that all three kdi's are equal
    rss_list = []
    beta_list = []
    eigen_xkx_list = []
    for y_index in range(len(yKy.pheno)):  #!!!cmk kludge
        yKy_i = yKy[:, :, y_index : y_index + 1]
        XKX_i = XKX[:, :, y_index : y_index + 1]
        XKy_i = XKy[:, :, y_index : y_index + 1]

        ###############################################################
        # BETA
        #
        # ref: https://math.unm.edu/~james/w15-STAT576b.pdf
        # You can minimize squared error in linear regression with a beta of
        # XTX = X.T.dot(X)
        # beta = np.linalg.inv(XTX).dot(X.T.dot(y))
        #
        # ref: https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Matrix_inverse_via_eigendecomposition
        # You can find an inverse of XTX using eigen
        # print(np.linalg.inv(XTX))
        # values,vectors = np.linalg.eigh(XTX)
        # print((vectors/values).dot(vectors.T))
        #
        # So, beta = (vectors/values).dot(vectors.T).dot(X.T.dot(y))
        # or  beta = vectors.dot(vectors.T.dot(X.T.dot(y))/values)

        eigen_xkx_i = EigenData.from_aka(XKX_i, keep_above=1e-10)
        XKy_r_s = eigen_xkx_i.rotate_and_scale(XKy_i, ignore_low_rank=True)
        beta_i = eigen_xkx_i.rotate_back(XKy_r_s)


        ##################################################################
        # RSS (aka SSR aka SSE)
        #
        # ref 1: https://en.wikipedia.org/wiki/Residual_sum_of_squares#Matrix_expression_for_the_OLS_residual_sum_of_squares
        # ref 2: http://www.web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf
        # RSS = ((y-y_predicted)**2).sum()
        # RSS = ((y-y_predicted).T.dot(y-y_predicted))
        # RSS = (y - X.dot(beta)).T.dot(y - X.dot(beta))
        # recall that (a-b).T.dot(a-b)=a.T.dot(a)-2*a.T.dot(b)+b.T.dot(b)
        # RSS = y.T.dot(y) - 2*y.T.dot(X.dot(beta)) + X.dot(beta).T.dot(X.dot(beta))
        # ref2: beta is choosen s.t. y.T.dot(X) = X.dot(beta).T.dot(X) aka X.T.dot(X).dot(beta)
        # RSS = y.T.dot(y) - 2*y.T.dot(X.dot(beta)) + y.T.dot(X).dot(beta))
        # RSS = y.T.dot(y) - y.T.dot(X.dot(beta)))

        rss_i = PstData(
            val=yKy_i.val - XKy_i.val.T.dot(beta_i.val),
            row=yKy_i.row,
            col=yKy_i.col,
        )
        rss_list.append(rss_i)
        beta_list.append(beta_i)
        eigen_xkx_list.append(eigen_xkx_i)

    rss = PstData(
        val=np.array([pstdata.val[0, 0] for pstdata in rss_list]).reshape(1, 1, -1),
        row=yKy.row,
        col=yKy.col,
    )
    val = rearrange([pstdata.val for pstdata in beta_list], "p c 1 -> c p")
    beta = PstData(val=val, row=XKy.row, col=yKy.pheno)  #!!!cmk kludge

    return rss, beta, eigen_xkx_list


def _loglikelihood(X, yKy, XKX, XKy, use_reml):
    if use_reml:
        nLL, beta = _loglikelihood_reml(X, yKy, XKX, XKy)
        return nLL, beta, None
    else:
        return _loglikelihood_ml(yKy, XKX, XKy)


def _loglikelihood_reml(X, yKy, XKX, XKy):
    kdi = yKy.kdi  # !!!cmk may want to check that all three kdi's are equal

    rss, beta, eigen_xkx_list = _common_code(yKy, XKX, XKy)

    nLL_list = []
    for y_index, _ in enumerate(yKy.pheno):
        kdi_i = kdi[y_index]
        # !!!cmk isn't this a kernel?
        #!!!cmk rename XX to xtx
        XX = PstData(val=X.val.T.dot(X.val), row=X.sid, col=X.sid)
        eigen_xx = EigenData.from_aka(XX)
        logdetXX, _ = eigen_xx.logdet()

        logdetXKX, _ = eigen_xkx_list[y_index].logdet()
        X_row_less_col = X.row_count - X.col_count
        sigma2 = float(rss.val[:, :, y_index]) / X_row_less_col
        nLL_i = 0.5 * (
            kdi_i.logdet
            + logdetXKX
            - logdetXX
            + X_row_less_col * (np.log(2.0 * np.pi * sigma2) + 1)
        )

        assert np.all(
            np.isreal(nLL_i)
        ), "nLL has an imaginary component, possibly due to constant covariates"
        # !!!cmk which is negative loglikelihood and which is LL?
        nLL_list.append(nLL_i)
    nnLL = np.array([-float(nLL) for nLL in nLL_list]).reshape(-1)
    return nnLL, beta


def _loglikelihood_ml(yKy, XKX, XKy):
    rss, beta, eigen_xkx_list = _common_code(yKy, XKX, XKy)
    kdi = yKy.kdi  # !!!cmk may want to check that all three kdi's are equal

    nLL_list = []
    variance_beta_list = []
    for y_index, _ in enumerate(yKy.pheno):  #!!!cmk kludge
        eigen_xkx_i = eigen_xkx_list[y_index]
        kdi_i = kdi[y_index]
        sigma2 = float(rss.val[:, :, y_index]) / kdi_i.row_count
        nLL_i = 0.5 * (
            kdi_i.logdet + kdi_i.row_count * (np.log(2.0 * np.pi * sigma2) + 1)
        )
        assert np.all(
            np.isreal(nLL_i)
        ), "nLL has an imaginary component, possibly due to constant covariates"
        nLL_list.append(nLL_i)
        variance_beta_i = (
            kdi_i.h2
            * sigma2
            * (eigen_xkx_i.vectors / eigen_xkx_i.values * eigen_xkx_i.vectors).sum(-1)
        )
        assert len(variance_beta_i.shape) == 1, "!!!cmk"
        variance_beta_list.append(variance_beta_i)
        # !!!cmk which is negative loglikelihood and which is LL?
    nnLL = np.array([-float(nLL) for nLL in nLL_list]).reshape(-1)

    variance_beta = np.c_[variance_beta_list]
    assert len(variance_beta.shape) == 2, "!!!cmk"
    variance_beta = variance_beta.T
    # !!!cmk variance_beta = np.squeeze(variance_beta,0).T
    assert variance_beta.shape == (
        XKX.row_count,
        len(yKy.pheno),
    ), "!!!cmk"  #!!!cmk kludge
    return nnLL, beta, variance_beta


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
