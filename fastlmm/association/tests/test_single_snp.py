# flake8: noqa: E501

import sys
import numpy as np
import logging
import unittest
import os.path
import doctest
import pandas as pd
from numpy.random import RandomState
from pathlib import Path

from fastlmm.association import single_snp
from fastlmm.association import single_snp_linreg
import pysnptools.util.pheno as pstpheno
from fastlmm.feature_selection.test import TestFeatureSelection
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc
from pysnptools.kernelreader import Identity as KernelIdentity
from pysnptools.standardizer import Unit
from pysnptools.snpreader import Bed, Pheno, SnpData
from pysnptools.kernelreader import SnpKernel
from unittest.mock import patch


class TestSingleSnp(unittest.TestCase):

    # !!! created a Expect Durbin, too

    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary

        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
        )
        self.bedbase = os.path.join(
            self.pythonpath, "tests/datasets/all_chr.maf0.001.N300"
        )
        self.phen_fn = os.path.join(
            self.pythonpath, "tests/datasets/phenSynthFrom22.23.N300.randcidorder.txt"
        )
        self.cov_fn = os.path.join(
            self.pythonpath, "tests/datasets/all_chr.maf0.001.covariates.N300.txt"
        )

    tempout_dir = "tempout/single_snp"

    def test_match_cpp(self):
        """
        match
            FaSTLMM.207\Data\DemoData>..\.cd.\bin\windows\cpp_mkl\fastlmmc -bfile snps -extract topsnps.txt -bfileSim snps -extractSim ASout.snps.txt -pheno pheno.txt -covar covariate.txt -out topsnps.singlesnp.txt -logDelta 0 -verbose 100

        """
        logging.info("TestSingleSnp test_match_cpp")
        snps = Bed(
            os.path.join(self.pythonpath, "tests/datasets/selecttest/snps.bed"),
            count_A1=False,
        )
        pheno = os.path.join(self.pythonpath, "tests/datasets/selecttest/pheno.txt")
        covar = os.path.join(self.pythonpath, "tests/datasets/selecttest/covariate.txt")
        sim_sid = [
            "snp26250_m0_.19m1_.19",
            "snp82500_m0_.28m1_.28",
            "snp63751_m0_.23m1_.23",
            "snp48753_m0_.4m1_.4",
            "snp45001_m0_.26m1_.26",
            "snp52500_m0_.05m1_.05",
            "snp75002_m0_.39m1_.39",
            "snp41253_m0_.07m1_.07",
            "snp11253_m0_.2m1_.2",
            "snp86250_m0_.33m1_.33",
            "snp3753_m0_.23m1_.23",
            "snp75003_m0_.32m1_.32",
            "snp30002_m0_.25m1_.25",
            "snp26252_m0_.19m1_.19",
            "snp67501_m0_.15m1_.15",
            "snp63750_m0_.28m1_.28",
            "snp30001_m0_.28m1_.28",
            "snp52502_m0_.35m1_.35",
            "snp33752_m0_.31m1_.31",
            "snp37503_m0_.37m1_.37",
            "snp15002_m0_.11m1_.11",
            "snp3751_m0_.34m1_.34",
            "snp7502_m0_.18m1_.18",
            "snp52503_m0_.3m1_.3",
            "snp30000_m0_.39m1_.39",
            "isnp4457_m0_.11m1_.11",
            "isnp23145_m0_.2m1_.2",
            "snp60001_m0_.39m1_.39",
            "snp33753_m0_.16m1_.16",
            "isnp60813_m0_.2m1_.2",
            "snp82502_m0_.34m1_.34",
            "snp11252_m0_.13m1_.13",
        ]
        sim_idx = snps.sid_to_index(sim_sid)
        test_sid = [
            "snp26250_m0_.19m1_.19",
            "snp63751_m0_.23m1_.23",
            "snp82500_m0_.28m1_.28",
            "snp48753_m0_.4m1_.4",
            "snp45001_m0_.26m1_.26",
            "snp52500_m0_.05m1_.05",
            "snp75002_m0_.39m1_.39",
            "snp41253_m0_.07m1_.07",
            "snp86250_m0_.33m1_.33",
            "snp15002_m0_.11m1_.11",
            "snp33752_m0_.31m1_.31",
            "snp26252_m0_.19m1_.19",
            "snp30001_m0_.28m1_.28",
            "snp11253_m0_.2m1_.2",
            "snp67501_m0_.15m1_.15",
            "snp3753_m0_.23m1_.23",
            "snp52502_m0_.35m1_.35",
            "snp30000_m0_.39m1_.39",
            "snp30002_m0_.25m1_.25",
        ]
        test_idx = snps.sid_to_index(test_sid)

        for G0, G1 in [
            (snps[:, sim_idx], KernelIdentity(snps.iid)),
            (KernelIdentity(snps.iid), snps[:, sim_idx]),
        ]:
            frame_h2 = single_snp(
                test_snps=snps[:, test_idx],
                pheno=pheno,
                G0=G0,
                G1=G1,
                covar=covar,
                h2=0.5,
                leave_out_one_chrom=False,
                count_A1=False,
            )
            frame_log_delta = single_snp(
                test_snps=snps[:, test_idx],
                pheno=pheno,
                G0=G0,
                G1=G1,
                covar=covar,
                log_delta=0,
                leave_out_one_chrom=False,
                count_A1=False,
            )
            for frame in [frame_h2, frame_log_delta]:
                referenceOutfile = TestFeatureSelection.reference_file(
                    "single_snp/topsnps.single.txt"
                )
                reference = pd.read_csv(
                    referenceOutfile, sep="\t"
                )  # We've manually remove all comments and blank lines from this file
                assert len(frame) == len(reference)
                for _, row in reference.iterrows():
                    sid = row.SNP
                    pvalue = frame[frame["SNP"] == sid].iloc[0].PValue
                    reldiff = abs(row.Pvalue - pvalue) / row.Pvalue
                    assert (
                        reldiff < 0.035
                    ), "'{0}' pvalue_list differ too much {4} -- {2} vs {3}".format(
                        sid, None, row.Pvalue, pvalue, reldiff
                    )

    def file_name(self, testcase_name):
        temp_fn = os.path.join(self.tempout_dir, testcase_name + ".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_mixing(self):
        logging.info("TestSingleSnp test_mixing")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("mixing")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps[:, 10:100],
            leave_out_one_chrom=False,
            covar=covar,
            G1=test_snps[:, 100:200],
            mixing=None,
            output_file_name=output_file_name,
            count_A1=False,
        )

        self.compare_files(frame, "mixing")

    def test_mixingKs(self):
        logging.info("TestSingleSnp test_mixingKs")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("mixingKs")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            K0=SnpKernel(test_snps[:, 10:100], Unit()),
            leave_out_one_chrom=False,
            covar=covar,
            K1=SnpKernel(test_snps[:, 100:200], Unit()),
            mixing=None,
            output_file_name=output_file_name,
            count_A1=False,
        )

        self.compare_files(frame, "mixing")

    def test_mixid(self):
        logging.info("TestSingleSnp test_mixid")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("mixid")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps[:, 10:100],
            leave_out_one_chrom=False,
            covar=covar,
            K1=KernelIdentity(test_snps.iid),
            mixing=0.25,
            output_file_name=output_file_name,
            count_A1=False,
        )

        self.compare_files(frame, "mixid")

    def test_one(self):
        logging.info("TestSingleSnp test_one")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("one")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            mixing=0,
            leave_out_one_chrom=False,
            G0=test_snps,
            covar=covar,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "one")

    def test_snp_fract_var_exp(self):
        from fastlmm.util import example_file  # Download and return local file name

        logging.info("TestSingleSnp test_snp_fract_var_exp")
        bed_fn = example_file("tests/datasets/synth/all.*", "*.bed")
        pheno_fn = example_file("tests/datasets/synth/pheno_10_causals.txt")
        cov_fn = example_file("tests/datasets/synth/cov.txt")

        output_file = self.file_name("snp_fract_var_exp")
        frame = single_snp(
            bed_fn,
            pheno_fn,
            covar=cov_fn,
            show_snp_fract_var_exp=True,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(
            frame, "snp_fract_var_exp", columns=["PValue", "EffectSize", "SnpFractVarExpl"]
        )

    def test_zero_pheno(self):
        logging.info("TestSingleSnp test_zero_pheno")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = Pheno(self.phen_fn)[:, 0:0]
        covar = self.cov_fn

        got_expected_fail = False
        try:
            frame = single_snp(
                test_snps=test_snps[:, :10],
                pheno=pheno,
                mixing=0,
                leave_out_one_chrom=False,
                G0=test_snps,
                covar=covar,
                count_A1=False,
            )
        except Exception as e:
            got_expected_fail = True
        assert got_expected_fail, "Did not get expected fail"

    def test_missing_covar(self):
        logging.info("TestSingleSnp test_missing_covar")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = Pheno(self.cov_fn).read()
        covar.val[0, 0] = np.nan

        got_expected_fail = False
        try:
            frame = single_snp(
                test_snps=test_snps[:, :10],
                pheno=pheno,
                mixing=0,
                leave_out_one_chrom=False,
                G0=test_snps,
                covar=covar,
                count_A1=False,
            )
        except Exception as e:
            got_expected_fail = True
        assert got_expected_fail, "Did not get expected fail"

        covar_by_chrom = {chrom: covar for chrom in set(test_snps.pos[:, 0])}
        got_expected_fail = False
        try:
            frame = single_snp(
                test_snps=test_snps[:, :10],
                pheno=pheno,
                mixing=0,
                leave_out_one_chrom=True,
                G0=test_snps,
                covar_by_chrom=covar_by_chrom,
                count_A1=False,
            )
        except Exception as e:
            got_expected_fail = True
        assert got_expected_fail, "Did not get expected fail"

    def test_thres(self):
        logging.info("TestSingleSnp test_thres")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        reffile = TestFeatureSelection.reference_file("single_snp/one.txt")
        reference = pd.read_csv(reffile, delimiter="\s", comment=None, engine="python")

        for random_seed in [0, 1]:
            for pvalue_threshold in [0.5, None, 1.0]:
                for random_threshold in [0.5, None, 1.0]:
                    frame = single_snp(
                        test_snps=test_snps[:, :10],
                        pheno=pheno,
                        mixing=0,
                        leave_out_one_chrom=False,
                        G0=test_snps,
                        covar=covar,
                        pvalue_threshold=pvalue_threshold,
                        random_threshold=random_threshold,
                        random_seed=random_seed,
                        count_A1=False,
                    )

                    assert len(frame) <= len(
                        reference
                    ), "# of pairs differs from file '{0}'".format(reffile)
                    if len(frame) < len(reference):
                        assert frame["PValueCount"].iloc[0] == len(
                            reference
                        ), "row_count doesn't match the reference # of rows"
                    if random_threshold is not None:
                        assert np.all(
                            frame["RandomThreshold"] == random_threshold
                        ), "Expect all rows to have 'RandomThreshold'"
                        assert np.all(
                            frame["RandomSeed"] == random_seed
                        ), "Expect all rows to have right 'RandomSeed'"
                        if pvalue_threshold is not None:
                            assert np.all(
                                (frame["RandomValue"] <= random_threshold)
                                + (frame["PValue"] <= pvalue_threshold)
                            ), "Expect all rows have random value or pvalue less than threshold"
                    if pvalue_threshold is not None:
                        assert np.all(
                            frame["PValueThreshold"] == pvalue_threshold
                        ), "Expect all rows to have 'PValueThreshold'"
                    for _, row in frame.iterrows():
                        sid = row.SNP
                        pvalue = reference[reference["SNP"] == sid].iloc[0].PValue
                        diff = abs(row.PValue - pvalue)
                        if diff > 1e-5 or np.isnan(diff):
                            raise Exception(
                                "pair {0} differs too much from file '{1}'".format(
                                    sid, reffile
                                )
                            )
                        assert abs(row.PValue - pvalue) < 1e-5, "wrong"

    def test_linreg(self):
        logging.info("TestSingleSnp test_linreg")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("linreg")

        frame1 = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            mixing=0,
            leave_out_one_chrom=False,
            G0=KernelIdentity(iid=test_snps.iid),
            covar=covar,
            output_file_name=output_file,
            count_A1=False,
        )

        frame1 = frame1[["sid_index", "SNP", "Chr", "GenDist", "ChrPos", "PValue"]]
        self.compare_files(frame1, "linreg")

        with patch.dict("os.environ", {"ARRAY_MODULE": "numpy"}) as _:
            frame2 = single_snp_linreg(
                test_snps=test_snps[:, :10],
                pheno=pheno,
                covar=covar,
                output_file_name=output_file,
            )
        self.compare_files(frame2, "linreg")

    def test_noK0(self):
        logging.info("TestSingleSnp test_noK0")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("noK0")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            mixing=1,
            leave_out_one_chrom=False,
            G1=test_snps,
            covar=covar,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "one")

    def test_gb_goal(self):
        logging.info("TestSingleSnp test_gb_goal")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("gb_goal")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            mixing=0,
            leave_out_one_chrom=False,
            G0=test_snps,
            covar=covar,
            GB_goal=0,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "one")

        output_file = self.file_name("gb_goal2")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            mixing=0,
            leave_out_one_chrom=False,
            G0=test_snps,
            covar=covar,
            GB_goal=0.12,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "one")

    def test_other(self):
        logging.info("TestSingleSnp test_other")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("other")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            leave_out_one_chrom=False,
            K1=test_snps,
            covar=covar,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "one")

    def test_none(self):
        logging.info("TestSingleSnp test_none")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("none")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            mixing=0,
            leave_out_one_chrom=False,
            K0=KernelIdentity(test_snps.iid),
            covar=covar,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "none")

    def test_interact(self):
        logging.info("TestSingleSnp test_interact")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("interact")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            mixing=0,
            leave_out_one_chrom=False,
            G0=test_snps,
            covar=covar,
            interact_with_snp=1,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "interact")

    def test_preload_files(self):
        logging.info("TestSingleSnp test_preload_files")
        test_snps = self.bedbase
        pheno = pstpheno.loadOnePhen(self.phen_fn, vectorize=True)
        covar = pstpheno.loadPhen(self.cov_fn)
        bed = Bed(test_snps, count_A1=False)

        output_file_name = self.file_name("preload_files")

        frame = single_snp(
            test_snps=bed[:, :10],
            pheno=pheno,
            G0=test_snps,
            mixing=0,
            leave_out_one_chrom=False,
            covar=covar,
            output_file_name=output_file_name,
            count_A1=False,
        )
        self.compare_files(frame, "one")

    def test_SNC(self):
        logging.info("TestSNC")
        test_snps = self.bedbase
        pheno = pstpheno.loadOnePhen(self.phen_fn, vectorize=True)
        covar = pstpheno.loadPhen(self.cov_fn)
        bed = Bed(test_snps, count_A1=False)
        snc = bed.read()
        snc.val[:, 2] = 0  # make SNP #2 have constant values (aka a SNC)

        output_file_name = self.file_name("snc")

        frame = single_snp(
            test_snps=snc[:, :10],
            pheno=pheno,
            G0=snc,
            mixing=0,
            leave_out_one_chrom=False,
            covar=covar,
            output_file_name=output_file_name,
            count_A1=False,
        )
        self.compare_files(frame, "snc")

    def test_G0_has_reader(self):
        logging.info("TestSingleSnp test_G0_has_reader")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G0_has_reader")

        frame0 = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps,
            leave_out_one_chrom=False,
            covar=covar,
            mixing=0,
            output_file_name=output_file_name,
            count_A1=False,
        )
        self.compare_files(frame0, "one")

        frame1 = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=KernelIdentity(test_snps.iid),
            G1=test_snps,
            leave_out_one_chrom=False,
            covar=covar,
            mixing=1,
            output_file_name=output_file_name,
            count_A1=False,
        )
        self.compare_files(frame1, "one")

    def test_no_cov(self):
        logging.info("TestSingleSnp test_no_cov")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn

        output_file_name = self.file_name("no_cov")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps,
            mixing=0,
            leave_out_one_chrom=False,
            output_file_name=output_file_name,
            count_A1=False,
        )

        self.compare_files(frame, "no_cov")

    def test_no_cov_b(self):
        logging.info("TestSingleSnp test_no_cov_b")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn

        output_file_name = self.file_name("no_cov_b")
        covar = pstpheno.loadPhen(self.cov_fn)
        covar["vals"] = np.delete(covar["vals"], np.s_[:], 1)  # Remove all the columns
        covar["header"] = []

        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps,
            leave_out_one_chrom=False,
            covar=covar,
            mixing=0,
            output_file_name=output_file_name,
            count_A1=False,
        )

        self.compare_files(frame, "no_cov")

    def test_G1(self):
        logging.info("TestSingleSnp test_G1")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1")
        for force_full_rank, force_low_rank in [
            (False, True),
            (False, False),
            (True, False),
        ]:
            logging.info("{0},{1}".format(force_full_rank, force_low_rank))
            frame = single_snp(
                test_snps=test_snps[:, :10],
                pheno=pheno,
                G0=test_snps[:, 10:100],
                leave_out_one_chrom=False,
                covar=covar,
                G1=test_snps[:, 100:200],
                mixing=0.5,
                force_full_rank=force_full_rank,
                force_low_rank=force_low_rank,
                output_file_name=output_file_name,
                count_A1=False,
            )
            self.compare_files(frame, "G1")

    def test_file_cache(self):
        logging.info("TestSingleSnp test_file_cache")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1")
        cache_file = self.file_name("cache_file") + ".npz"
        if os.path.exists(cache_file):
            os.remove(cache_file)
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps[:, 10:100],
            leave_out_one_chrom=False,
            covar=covar,
            G1=test_snps[:, 100:200],
            mixing=0.5,
            output_file_name=output_file_name,
            cache_file=cache_file,
            count_A1=False,
        )
        self.compare_files(frame, "G1")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps[:, 10:100],
            leave_out_one_chrom=False,
            covar=covar,
            G1=test_snps[:, 100:200],
            mixing=0.5,
            output_file_name=output_file_name,
            cache_file=cache_file,
            count_A1=False,
        )
        self.compare_files(frame, "G1")

    def test_G1_mixing(self):
        logging.info("TestSingleSnp test_G1_mixing")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file_name = self.file_name("G1_mixing")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps,
            leave_out_one_chrom=False,
            covar=covar,
            G1=test_snps[:, 100:200],
            mixing=0,
            output_file_name=output_file_name,
            count_A1=False,
        )

        self.compare_files(frame, "one")

    def test_unknown_sid(self):
        logging.info("TestSingleSnp test_unknown_sid")

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        try:
            frame = single_snp(
                test_snps=test_snps,
                G0=test_snps,
                pheno=pheno,
                leave_out_one_chrom=False,
                mixing=0,
                covar=covar,
                sid_list=["1_4", "bogus sid", "1_9"],
                count_A1=False,
            )
            failed = False
        except:
            failed = True

        assert failed

    def test_cid_intersect(self):
        logging.info("TestSingleSnp test_cid_intersect")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = pstpheno.loadOnePhen(self.phen_fn, vectorize=True)
        pheno["iid"] = np.vstack([pheno["iid"][::-1], [["Bogus", "Bogus"]]])
        pheno["vals"] = np.hstack([pheno["vals"][::-1], [-34343]])

        covar = self.cov_fn
        output_file_name = self.file_name("cid_intersect")
        frame = single_snp(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            G0=test_snps,
            leave_out_one_chrom=False,
            covar=covar,
            mixing=0,
            output_file_name=output_file_name,
            count_A1=False,
        )

        self.compare_files(frame, "one")

    def compare_files(self, frame, ref_base, columns=["PValue"]):
        reffile = TestFeatureSelection.reference_file("single_snp/" + ref_base + ".txt")

        # sid_list,pvalue_list = frame['SNP'].values,frame['Pvalue'].values

        # sid_to_pvalue = {}
        # for index, sid in enumerate(sid_list):
        #    sid_to_pvalue[sid] = pvalue_list[index]

        reference = pd.read_csv(reffile, delimiter="\s", comment=None, engine="python")
        assert len(frame) == len(
            reference
        ), "# of pairs differs from file '{0}'".format(reffile)
        for _, row in reference.iterrows():
            sid = row.SNP
            for column in columns:
                value = frame[frame["SNP"] == sid].iloc[0][column]
                diff = abs(row[column] - value)
                if diff > 1e-5 or np.isnan(diff):
                    raise Exception(
                        f"pair {sid} differs too much on {column} from file '{reffile}'"
                    )
                assert abs(row[column] - value) < 1e-5, "wrong"

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)) + "/..")
        result = doctest.testmod(sys.modules["fastlmm.association.single_snp"])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


class TestSingleSnpLeaveOutOneChrom(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from pysnptools.util import create_directory_if_necessary

        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")
        )
        self.bedbase = os.path.join(
            self.pythonpath, "fastlmm/feature_selection/examples/toydata.5chrom.bed"
        )
        self.phen_fn = os.path.join(
            self.pythonpath, "fastlmm/feature_selection/examples/toydata.phe"
        )
        self.cov_fn = os.path.join(
            self.pythonpath, "fastlmm/feature_selection/examples/toydata.cov"
        )

    tempout_dir = "tempout/single_snp"

    def file_name(self, testcase_name):
        temp_fn = os.path.join(self.tempout_dir, testcase_name + ".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_leave_one_out_with_prekernels(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_leave_one_out_with_prekernels")
        from pysnptools.kernelstandardizer import DiagKtoN

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        chrom_to_kernel = {}
        with patch.dict("os.environ", {"ARRAY_MODULE": "numpy"}) as _:
            for chrom in np.unique(test_snps.pos[:, 0]):
                other_snps = test_snps[:, test_snps.pos[:, 0] != chrom]
                kernel = other_snps.read_kernel(
                    standardizer=Unit(), block_size=500
                )  # Create a kernel from the SNPs not used in testing
                chrom_to_kernel[chrom] = kernel.standardize(
                    DiagKtoN()
                )  # improves the kernel numerically by making its diagonal sum to iid_count

        output_file = self.file_name("one_looc_prekernel")
        frame = single_snp(
            test_snps,
            pheno,
            covar=covar,
            K0=chrom_to_kernel,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "one_looc")

    def test_one_looc(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_one_looc")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("one_looc")
        frame = single_snp(
            test_snps,
            pheno,
            covar=covar,
            mixing=0,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "one_looc")

    def test_runner(self):
        logging.info("TestRunner")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        runner = LocalMultiProc(6, just_one_process=True)
        for map_reduce_outer in [True, False]:
            frame = single_snp(
                test_snps,
                pheno,
                covar=covar,
                mixing=0,
                count_A1=False,
                runner=runner,
                map_reduce_outer=map_reduce_outer,
            )

            self.compare_files(frame, "one_looc")

    def test_multipheno(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_multipheno")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        pheno2 = Pheno(pheno).read()
        pheno2.val[0, 0] = 100
        pheno2.val[1, 0] = -100

        cache_file = None

        if True:
            pheno12 = SnpData(
                iid=pheno2.iid,
                sid=["pheno1", "pheno2"],
                val=np.c_[Pheno(pheno).read().val, pheno2.val],
            )
            output_file = self.file_name("multipheno12")
            frame = single_snp(
                test_snps[:, ::10],
                pheno12,
                force_full_rank=True,
                covar=covar,
                cache_file=cache_file,
                output_file_name=output_file,
            )
            frame1 = frame[frame["Pheno"] == "pheno1"]
            del frame1["Pheno"]
            frame1_ef = frame1.copy()
            self.compare_files(frame1, "two_looc")
            self.compare_files_effect_size(frame1_ef, "two_looc")

            frame2 = frame[frame["Pheno"] == "pheno2"]
            del frame2["Pheno"]
            frame2_ef = frame2.copy()
            self.compare_files(frame2, "multipheno2")
            self.compare_files_effect_size(frame2_ef, "multipheno2")

        if True:
            pheno11 = SnpData(
                iid=pheno2.iid,
                sid=["pheno1a", "pheno1b"],
                val=np.c_[Pheno(pheno).read().val, Pheno(pheno).read().val],
            )
            output_file = self.file_name("multipheno11")
            frame = single_snp(
                test_snps[:, ::10],
                pheno11,
                force_full_rank=True,
                covar=covar,
                output_file_name=output_file,
                count_A1=False,
            )

            frame1 = frame[frame["Pheno"] == "pheno1a"]
            del frame1["Pheno"]
            self.compare_files(frame1, "two_looc")

            frame2 = frame[frame["Pheno"] == "pheno1b"]
            del frame2["Pheno"]
            self.compare_files(frame2, "two_looc")

        if True:
            output_file = self.file_name("multipheno1")
            frame = single_snp(
                test_snps[:, ::10],
                pheno,
                force_full_rank=True,
                covar=covar,
                output_file_name=output_file,
                count_A1=False,
            )

            self.compare_files(frame, "two_looc")

        if True:
            output_file = self.file_name("multipheno2")
            frame = single_snp(
                test_snps[:, ::10],
                pheno2,
                force_full_rank=True,
                covar=covar,
                output_file_name=output_file,
                count_A1=False,
            )

            self.compare_files(frame, "multipheno2")

        if True:
            pheno22 = SnpData(
                iid=pheno2.iid,
                sid=["pheno2a", "pheno2b"],
                val=np.c_[pheno2.val, pheno2.val],
            )
            output_file = self.file_name("multipheno22")
            frame = single_snp(
                test_snps[:, ::10],
                pheno22,
                force_full_rank=True,
                covar=covar,
                output_file_name=output_file,
                count_A1=False,
            )
            frame1 = frame[frame["Pheno"] == "pheno2a"]
            del frame1["Pheno"]
            self.compare_files(frame1, "multipheno2")

            frame2 = frame[frame["Pheno"] == "pheno2b"]
            del frame2["Pheno"]
            self.compare_files(frame2, "multipheno2")

    def test_multipheno2(self):
        logging.info("test_multipheno")
        from fastlmm.util import example_file  # Download and return local file name

        bed = Bed(example_file("tests/datasets/synth/all.*", "*.bed"), count_A1=True)[
            :, ::10
        ]
        phen_fn = example_file("tests/datasets/synth/pheno_10_causals.txt")
        cov_fn = example_file("tests/datasets/synth/cov.txt")

        random_state = RandomState(29921)
        pheno_reference = Pheno(phen_fn).read()
        for pheno_count in [2, 5, 1]:
            val = random_state.normal(
                loc=pheno_count,
                scale=pheno_count,
                size=(pheno_reference.iid_count, pheno_count),
            )
            pheno_col = ["pheno{0}".format(i) for i in range(pheno_count)]
            pheno_multi = SnpData(iid=pheno_reference.iid, sid=pheno_col, val=val)

            reference = pd.concat(
                [
                    single_snp(
                        test_snps=bed, pheno=pheno_multi[:, pheno_index], covar=cov_fn
                    )
                    for pheno_index in range(pheno_count)
                ]
            )

            for force_full_rank, force_low_rank in [
                (True, False),
                (False, True),
                (False, False),
            ]:

                frame = single_snp(
                    test_snps=bed,
                    pheno=pheno_multi,
                    covar=cov_fn,
                    force_full_rank=force_full_rank,
                    force_low_rank=force_low_rank,
                )

                assert len(frame) == len(reference), "# of pairs differs"
                for sid in sorted(
                    set(reference.SNP)
                ):  # This ignores which pheno produces which pvalue
                    pvalue_frame = np.array(sorted(frame[frame["SNP"] == sid].PValue))
                    pvalue_reference = np.array(
                        sorted(reference[reference["SNP"] == sid].PValue)
                    )
                    assert (
                        abs(pvalue_frame - pvalue_reference) < 1e-5
                    ).all, "pair {0} differs too much from reference".format(sid)

    def create_phen3(self, phen):
        from fastlmm.util import example_file  # Download and return local file name

        phen = phen.read()
        rng = np.random.RandomState(seed=0)
        val1 = phen.val.copy()
        rng.shuffle(val1)
        val2 = phen.val.copy()
        rng.shuffle(val2)
        phen3 = SnpData(
            iid=phen.iid,
            sid=["phen0", "phen1", "phen2"],
            val=np.c_[phen.val, val1, val2],
        )
        return phen3

    def test_multipheno3(self):
        from pysnptools.kernelreader import SnpKernel
        from fastlmm.util import example_file  # Download and return local file name
        from pysnptools.standardizer import Standardizer, Unit

        bed = Bed(example_file("tests/datasets/synth/all.*", "*.bed"), count_A1=True)[
            :, ::10
        ]
        phen3 = self.create_phen3(
            Pheno(example_file("tests/datasets/synth/pheno_10_causals.txt"))
        )

        combo_index = 0
        for covar, interact in [
            (None, None),
            (Pheno(example_file("tests/datasets/synth/cov.txt")), None),
            (Pheno(example_file("tests/datasets/synth/cov.txt")), 0),
        ]:
            for force_full_rank, force_low_rank in [
                (True, False),
                (False, True),
                (False, False),
            ]:
                for k0_as_snps in [True, False]:
                    logging.info(f"combo_index {combo_index}")
                    combo_index += 1
                    k0 = bed
                    if not k0_as_snps:
                        k0 = SnpKernel(k0, standardizer=Unit())

                    for leave_out_one_chrom in [False, True]:

                        logging.info(
                            [
                                covar,
                                interact,
                                force_full_rank,
                                force_low_rank,
                                k0_as_snps,
                                leave_out_one_chrom,
                            ]
                        )

                        result3 = single_snp(
                            test_snps=bed,
                            pheno=phen3,
                            covar=covar,
                            K0=k0,
                            interact_with_snp=interact,
                            force_full_rank=force_full_rank,
                            force_low_rank=force_low_rank,
                            leave_out_one_chrom=leave_out_one_chrom,
                        )

                        result_list = [
                            single_snp(
                                test_snps=bed,
                                pheno=phen3[:, i],
                                covar=covar,
                                K0=k0,
                                interact_with_snp=interact,
                                force_full_rank=force_full_rank,
                                force_low_rank=force_low_rank,
                                leave_out_one_chrom=leave_out_one_chrom,
                            )
                            for i in range(3)
                        ]

                        for i in range(3):
                            self.compare_df(
                                result3[result3["Pheno"] == phen3.sid[i]],
                                result_list[i],
                                "test_multipheno3",
                            )

    def test_multipheno_expected_exceptions(self):

        from pysnptools.kernelreader import SnpKernel
        from fastlmm.util import example_file  # Download and return local file name
        from pysnptools.standardizer import Standardizer, Unit

        bed = Bed(example_file("tests/datasets/synth/all.*", "*.bed"), count_A1=True)[
            :, ::10
        ]
        phen = Pheno(example_file("tests/datasets/synth/pheno_10_causals.txt"))
        phen3 = self.create_phen3(phen)

        got_expected_fail = False
        try:
            single_snp(
                test_snps=bed,
                pheno=phen3,
                covar=None,
                K0=bed,
                K1=bed,
                interact_with_snp=None,
                force_full_rank=False,
                force_low_rank=False,
            )
        except Exception as e:
            assert "2nd kernel" in str(e)
            got_expected_fail = True
        assert got_expected_fail, "Did not get expected fail"

        phen3.val[1, :] = np.nan  # Add a missing value to all phenos
        single_snp(
            test_snps=bed,
            pheno=phen3,
            covar=None,
            K0=bed,
            interact_with_snp=None,
            force_full_rank=False,
            force_low_rank=False,
        )
        phen3.val[0, 0] = np.nan  # Add a missing value to one pheno, but not the others
        got_expected_fail = False
        try:
            single_snp(
                test_snps=bed,
                pheno=phen3,
                covar=None,
                K0=bed,
                interact_with_snp=None,
                force_full_rank=False,
                force_low_rank=False,
            )
        except Exception as e:
            assert "multiple phenotypes" in str(e)
            got_expected_fail = True
        assert got_expected_fail, "Did not get expected fail"

    def test_cache(self):
        test_snpsx = Bed(self.bedbase, count_A1=False)
        phen1 = self.phen_fn
        phen3 = self.create_phen3(Pheno(self.phen_fn))
        covar = self.cov_fn

        for leave_out_one_chrom, ref_file1, ref_file3, test_snps in [
            (True, "one_looc", "one_looc3", test_snpsx),
            (False, "one", "one3", test_snpsx[:, :10]),
        ]:
            for force_full_rank, force_low_rank in [
                (False, True),
                (True, False),
                (False, False),
            ]:
                for pheno, ref_file in [(phen3, ref_file3), (phen1, ref_file1)]:
                    output_file = self.file_name(
                        f"cache{leave_out_one_chrom}{force_full_rank}{force_low_rank}"
                    )
                    cache_file = self.file_name(output_file + "cache")
                    for p in Path(cache_file).parent.glob(Path(cache_file).name + ".*"):
                        p.unlink()
                    frame = single_snp(
                        test_snps,
                        pheno,
                        cache_file=cache_file,
                        covar=covar,
                        mixing=0,
                        output_file_name=output_file,
                        count_A1=False,
                        leave_out_one_chrom=leave_out_one_chrom,
                        force_full_rank=force_full_rank,
                        force_low_rank=force_low_rank,
                    )

                    self.compare_files(frame, ref_file)
                    frame = single_snp(
                        test_snps,
                        pheno,
                        cache_file=cache_file,
                        covar=covar,
                        mixing=0,
                        output_file_name=output_file,
                        count_A1=False,
                        leave_out_one_chrom=leave_out_one_chrom,
                        force_full_rank=force_full_rank,
                        force_low_rank=force_low_rank,
                    )
                    self.compare_files(frame, ref_file)

    def test_two_looc(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_two_looc")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("two_looc")
        frame = single_snp(
            test_snps[:, ::10],
            pheno,
            covar=covar,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "two_looc")

    def test_interact_looc(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_interact_looc")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("interact_looc")
        frame = single_snp(
            test_snps,
            pheno,
            covar=covar,
            mixing=0,
            interact_with_snp=0,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "interact_looc")

    def test_covar_by_chrom(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_covar_by_chrom")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = Pheno(self.cov_fn).read()
        covar = SnpData(iid=covar.iid, sid=["pheno-1"], val=covar.val)
        covar_by_chrom = {chrom: self.cov_fn for chrom in range(1, 6)}
        output_file = self.file_name("covar_by_chrom")
        frame = single_snp(
            test_snps,
            pheno,
            covar=covar,
            mixing=0,
            covar_by_chrom=covar_by_chrom,
            output_file_name=output_file,
            count_A1=False,
        )

        self.compare_files(frame, "covar_by_chrom")

    def test_covar_by_chrom_mixing(self):
        logging.info("TestSingleSnpLeaveOutOneChrom test_covar_by_chrom_mixing")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn
        covar = Pheno(self.cov_fn).read()
        covar = SnpData(iid=covar.iid, sid=["pheno-1"], val=covar.val)
        covar_by_chrom = {chrom: self.cov_fn for chrom in range(1, 6)}
        output_file = self.file_name("covar_by_chrom_mixing")
        frame = single_snp(
            test_snps,
            pheno,
            covar=covar,
            covar_by_chrom=covar_by_chrom,
            output_file_name=output_file,
            count_A1=False,
        )
        self.compare_files(frame, "covar_by_chrom_mixing")

    def compare_files(self, frame, ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp/" + ref_base + ".txt")

        # sid_list,pvalue_list = frame['SNP'].values,frame['Pvalue'].values

        # sid_to_pvalue = {}
        # for index, sid in enumerate(sid_list):
        #    sid_to_pvalue[sid] = pvalue_list[index]

        reference = pd.read_csv(reffile, delimiter="\s", comment=None, engine="python")
        self.compare_df(frame, reference, reffile)

    def compare_df(self, frame, reference, name):
        assert len(frame) == len(
            reference
        ), "# of pairs differs from file '{0}'".format(name)
        if "Pheno" not in frame.columns or "Pheno" not in reference.columns:
            frame.set_index("SNP", inplace=True)
            reference.set_index("SNP", inplace=True)
        else:
            frame.set_index(["Pheno", "SNP"], inplace=True)
            reference.set_index(["Pheno", "SNP"], inplace=True)

        diff = frame.PValue - reference.PValue
        bad = diff[np.abs(diff) > 1e-5]
        if len(bad) > 0:
            raise Exception(
                "snps differ too much from file '{0}' at these snps {1}".format(
                    name, bad
                )
            )

    def compare_files_effect_size(self, frame, ref_base):
        reffile = TestFeatureSelection.reference_file("single_snp/" + ref_base + ".txt")
        reference = pd.read_csv(reffile, delimiter="\s", comment=None, engine="python")
        self.compare_df_effect_size(frame, reference, reffile)

    def compare_df_effect_size(self, frame, reference, name):
        assert len(frame) == len(
            reference
        ), "# of pairs differs from file '{0}'".format(name)
        if "Pheno" not in frame.columns or "Pheno" not in reference.columns:
            frame.set_index("SNP", inplace=True)
            reference.set_index("SNP", inplace=True)
        else:
            frame.set_index(["Pheno", "SNP"], inplace=True)
            reference.set_index(["Pheno", "SNP"], inplace=True)

        diff = frame.EffectSize - reference.EffectSize
        bad = diff[np.abs(diff) > 1e-5]
        if len(bad) > 0:
            raise Exception(
                "EffectSize of snps differ too much from file '{0}' at these snps {1}".format(
                    name, bad
                )
            )


def getTestSuite():
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnp)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpLeaveOutOneChrom)
    return unittest.TestSuite([suite1, suite2])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalInParts

    # this import is needed for the runner
    from fastlmm.association.tests.test_single_snp import (
        TestSingleSnp,
        TestSingleSnpLeaveOutOneChrom,
    )

    suites = unittest.TestSuite([getTestSuite()])

    if True:  # Standard test run
        r = unittest.TextTestRunner(failfast=False)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    else:  # Cluster test run
        logging.basicConfig(level=logging.INFO)

        from pysnptools.util.mapreduce1.distributabletest import DistributableTest

        # runner = Local()
        runner = LocalMultiProc(taskcount=2, mkl_num_threads=5, just_one_process=False)
        # runner = LocalInParts(0,2,mkl_num_threads=1) # For debugging the cluster runs
        distributable_test = DistributableTest(suites, "temp_test")
        print(runner.run(distributable_test))

    logging.info("done with testing")
