import numpy as np
import logging
import unittest
import os.path
import sys
import doctest

from fastlmm.association import epistasis
from fastlmm.association.epistasis import write
import pysnptools.util.pheno as pstpheno
from fastlmm.feature_selection.test import TestFeatureSelection
# from pysnptools.util.mapreduce1.runner import Local


class TestEpistasis(unittest.TestCase):

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

    tempout_dir = "tempout/epistasis"

    def test_redundant_float32_interaction(self):
        """An interaction in the additive column space has no likelihood gain."""
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from pysnptools.snpreader import Bed, SnpReader

        dataset = Path(self.pythonpath) / "tests/datasets/synth"
        reader = Bed(dataset / "all.bed", count_A1=True)
        pair = ["snp626_m0_.01m1_", "snp1626_m0_.52m1_.17"]
        # This pair has a redundant product even though both SNPs vary.
        values = reader[:, reader.sid_to_index(pair)].read(dtype="float64", xp=np).val
        additive = np.column_stack((np.ones(reader.iid_count), values))
        product = values[:, 0] * values[:, 1]
        coefficients = np.linalg.lstsq(additive, product, rcond=None)[0]
        np.testing.assert_allclose(additive @ coefficients, product, rtol=0, atol=1e-12)

        original_read = SnpReader.read
        with TemporaryDirectory() as directory:
            for dtype in ("float32", "float64"):
                observed = []

                def read(snp_reader, *args, **kwargs):
                    # Exercise the internal product correction independently of
                    # the public CPU/float64 fallback. Only this SNP pair is
                    # supplied to the internal design at the test precision.
                    if snp_reader.sid_count == 2 and set(snp_reader.sid) == set(pair):
                        kwargs.update(dtype=dtype, xp=np)
                        observed.append(dtype)
                    return original_read(snp_reader, *args, **kwargs)

                with self.subTest(dtype=dtype), patch.dict(
                    os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": dtype}
                ), patch.object(SnpReader, "read", read):
                    result = epistasis(
                        reader,
                        str(dataset / "pheno_10_causals.txt"),
                        G0=reader[:, reader.pos[:, 0] != 1],
                        covar=str(dataset / "cov.txt"),
                        sid_list_0=pair[:1],
                        sid_list_1=pair[1:],
                        log_delta=0,
                        cache_file=str(Path(directory) / (dtype + ".npz")),
                    )
                    self.assertEqual(observed, [dtype])
                    # Retain the legacy fixture's absolute P-value bound. Before
                    # promotion, float32 products give P=0.9999730973442345.
                    np.testing.assert_allclose(result.PValue, 1.0, rtol=0, atol=1e-5)

    def test_cpu_float64_precision_policy(self):
        """Caller defaults cannot select an unqualified epistasis computation."""
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import warnings
        import pandas as pd
        from fastlmm.inference import LMM
        from pysnptools.snpreader import Bed

        reader = Bed(self.bedbase, count_A1=False)
        original_setG = LMM.setG
        original_evaluate = LMM.nLLeval
        observed = []

        def setG(model, G0=None, G1=None, **kwargs):
            for values in (G0, G1):
                if values is not None:
                    self.assertEqual(values.dtype, np.dtype("float64"))
            return original_setG(model, G0=G0, G1=G1, **kwargs)

        def evaluate(model, *args, **kwargs):
            for values in (model.S, model.U, model.X, model.y):
                self.assertEqual(values.dtype, np.dtype("float64"))
            observed.append(model.S.dtype)
            return original_evaluate(model, *args, **kwargs)

        reference = None
        with TemporaryDirectory() as directory:
            for backend in ("numpy", "cupy"):
                for dtype in ("float64", "float32"):
                    for in_memory in (False, True):
                        with self.subTest(backend=backend, dtype=dtype, in_memory=in_memory):
                            source_reader = reader.read(dtype="float32", xp=np) if in_memory else reader
                            environment = {"ARRAY_MODULE": backend, "ARRAY_DTYPE": dtype}
                            with patch.dict(os.environ, environment), warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                with patch.object(LMM, "setG", setG), patch.object(LMM, "nLLeval", evaluate):
                                    result = epistasis(
                                        source_reader, self.phen_fn, G0=source_reader, covar=self.cov_fn,
                                        sid_list_0=reader.sid[:10], sid_list_1=reader.sid[5:15],
                                        cache_file=str(Path(directory) / f"{backend}-{dtype}-{in_memory}.npz"),
                                    )
                                self.assertEqual(os.environ["ARRAY_MODULE"], backend)
                                self.assertEqual(os.environ["ARRAY_DTYPE"], dtype)
                            notices = [item for item in caught if "epistasis is qualified only" in str(item.message)]
                            self.assertEqual(len(notices), int(backend != "numpy" or dtype != "float64"))
                            if reference is None:
                                reference = result
                            else:
                                pd.testing.assert_frame_equal(result, reference, check_exact=True)
        self.assertTrue(observed)

    def test_float64_policy_restores_settings_on_failure(self):
        from unittest.mock import patch
        import warnings
        from fastlmm.association.epistasis import _Epistasis
        from pysnptools.snpreader import Bed

        reader = Bed(self.bedbase, count_A1=False)
        environment = {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}
        failure = RuntimeError("epistasis computation failed")
        with patch.dict(os.environ, environment), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch.object(_Epistasis, "fill_in_cache_file", side_effect=failure):
                with self.assertRaises(RuntimeError) as raised:
                    epistasis(reader, self.phen_fn, G0=reader,
                              sid_list_0=reader.sid[:1], sid_list_1=reader.sid[1:2])
            self.assertIs(raised.exception, failure)
            self.assertEqual(os.environ["ARRAY_MODULE"], "cupy")
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float32")
        self.assertEqual(sum("epistasis is qualified only" in str(item.message) for item in caught), 1)

    def test_float32_eigen_cache_requires_recomputation(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from pysnptools.snpreader import Bed

        reader = Bed(self.bedbase, count_A1=False)
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float64"}
        ):
            for vectors_dtype, values_dtype in (("float32", "float64"), ("float64", "float32")):
                with self.subTest(vectors_dtype=vectors_dtype, values_dtype=values_dtype):
                    cache = Path(directory) / "old-cache.npz"
                    np.savez(cache, np.eye(reader.iid_count, dtype=vectors_dtype),
                             np.ones(reader.iid_count, dtype=values_dtype))
                    with self.assertRaisesRegex(ValueError, "requires a numpy/float64 cache"):
                        epistasis(reader, self.phen_fn, G0=reader,
                                  sid_list_0=reader.sid[:1], sid_list_1=reader.sid[1:2],
                                  cache_file=str(cache))

    def test_worker_defaults_cannot_narrow_epistasis(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import pandas as pd
        from pysnptools.snpreader import Bed
        from pysnptools.util.mapreduce1.runner import LocalMultiProc

        reader = Bed(self.bedbase, count_A1=False)
        arguments = dict(test_snps=reader, pheno=self.phen_fn, G0=reader, covar=self.cov_fn,
                         sid_list_0=reader.sid[:10], sid_list_1=reader.sid[5:15])
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float64"}
        ):
            # Run the reference through the same one-thread worker runner: BLAS
            # results depend on the thread count (16 parent threads changed PValue
            # by up to 1.8e-7), so only the task environment may differ.
            expected = epistasis(cache_file=str(Path(directory) / "reference.npz"),
                                 runner=LocalMultiProc(taskcount=1, num_threads=1), **arguments)
            # Explicit task environments override parent defaults after worker
            # imports. Reader arguments must still enforce the qualified dtype.
            runner = LocalMultiProc(
                taskcount=1, num_threads=1,
                taskindex_to_environ=lambda task_index: {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"},
            )
            actual = epistasis(cache_file=str(Path(directory) / "worker.npz"),
                               runner=runner, **arguments)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_local_multithread_matches_local(self):
        import subprocess
        from textwrap import dedent

        # Bound the test in a separate process because a deadlocked runner
        # cannot clean up its own threads.
        subprocess.run(
            [sys.executable, "-c", dedent("""
                from pathlib import Path
                from tempfile import TemporaryDirectory
                from unittest.mock import patch
                import os
                import numpy as np
                import pandas as pd
                from fastlmm.association import epistasis
                from pysnptools.snpreader import SnpData
                from pysnptools.util.mapreduce1.runner import Local, LocalMultiThread

                rng = np.random.default_rng(10)
                iid = [["family", str(index)] for index in range(24)]
                snps = SnpData(iid=iid, sid=["a", "b", "c"],
                               val=rng.integers(0, 3, size=(24, 3)).astype(float), xp=np)
                pheno = dict(iid=np.asarray(iid), vals=rng.normal(size=24))
                settings = {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float32"}
                with TemporaryDirectory() as directory, patch.dict(os.environ, settings):
                    options = dict(G0=snps, log_delta=0,
                                   cache_file=str(Path(directory) / "cache.npz"))
                    expected = epistasis(snps, pheno, runner=Local(), **options)
                    actual = epistasis(snps, pheno, runner=LocalMultiThread(2), **options)
                    assert len(actual) == 3
                    pd.testing.assert_frame_equal(actual, expected)
                    assert {key: os.environ[key] for key in settings} == settings
            """)],
            check=True, timeout=30,
        )

    def test_match_cpp(self):
        r"""
        match
            FaSTLMM.207\Data\DemoData>fastlmmc -snpPairs -bfile snps -extract topsnps.txt -bfileSim snps -extractSim ASout.snps.txt -pheno pheno.txt -covar covariate.txt -out topsnps.pairs.txt -logDelta 0 -verbose 100

        """
        logging.info("TestEpistasis test_match_cpp")
        from pysnptools.snpreader import Bed

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

        frame = epistasis(
            snps[:, test_idx],
            pheno,
            covar=covar,
            G0=snps[:, sim_idx],
            log_delta=0,
            count_A1=False,
        )
        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )

        referenceOutfile = TestFeatureSelection.reference_file(
            "epistasis/topsnps.pairs.txt"
        )

        import pandas as pd

        table = pd.read_csv(
            referenceOutfile, sep="\t"
        )  # We've manually remove all comments and blank lines from this file
        assert len(pvalue_list) == len(table)
        for row in table.iterrows():
            snp0cpp, snp1cpp, pvaluecpp, i1, i2 = row[1]
            for i in range(len(pvalue_list)):
                found = False
                pvaluepy = pvalue_list[i]
                snp0py = sid0[i]
                snp1py = sid1[i]
                if (snp0py == snp0cpp and snp1py == snp1cpp) or (
                    snp0py == snp1cpp and snp1py == snp0cpp
                ):
                    found = True
                    diff = abs(pvaluecpp - pvaluepy) / pvaluecpp
                    assert (
                        diff < 0.035
                    ), "'{0}' '{1}' pvalue_list differ too much {4} -- {2} vs {3}".format(
                        snp0cpp, snp1cpp, pvaluecpp, pvaluepy, diff
                    )
                    break
            assert found

        # self.sorted_pvalue_list = table["Pvalue"].tolist()
        # self.sorted_snps = table["SNP"].tolist()

        # self.pvalue_list = table.sort_values("Position")["Pvalue"].tolist()

        # print "done"
        # for i,pvalue_list in enumerate(pvalue_list):
        #    print "{0}\t{1}\t{2}".format(sid0[i],sid1[i],pvalue_list)
        # print "more done"

    def file_name(self, testcase_name):
        temp_fn = os.path.join(self.tempout_dir, testcase_name + ".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_one(self):
        logging.info("TestEpistasis test_one")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("one")
        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=test_snps.sid[:10],  # first 10 snps
            sid_list_1=test_snps.sid[5:15],  # Skip 5 snps, use next 10
            output_file_name=output_file,
            count_A1=False,
        )
        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )

        # Check the output file
        self.compare_files(sid0, sid1, pvalue_list, "one")

        # Check the values returned
        output_file2 = self.file_name("one_again")
        write(sid0, sid1, pvalue_list, output_file2)
        self.compare_files(sid0, sid1, pvalue_list, "one")

    def test_preload_files(self):
        logging.info("TestEpistasis test_preload_files")
        from pysnptools.snpreader import Bed

        test_snps = self.bedbase
        pheno = pstpheno.loadOnePhen(self.phen_fn, vectorize=True)
        covar = pstpheno.loadPhen(self.cov_fn)
        bed = Bed(test_snps, count_A1=False)

        output_file = self.file_name("preload_files")

        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=bed.sid[:10],  # first 10 snps
            sid_list_1=bed.sid[5:15],  # Skip 5 snps, use next 10
            output_file_name=output_file,
            count_A1=False,
        )
        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "one")

    def test_G0_has_reader(self):
        logging.info("TestEpistasis test_G0_has_reader")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("G0_has_reader")

        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=test_snps.sid[:10],  # first 10 snps
            sid_list_1=test_snps.sid[5:15],  # Skip 5 snps, use next 10
            output_file_name=output_file,
            count_A1=False,
        )
        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "one")

    def test_no_sid_list_0(self):
        logging.info("TestEpistasis test_no_sid_list_0")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("no_sid_list_0")
        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=["1_4"],
            output_file_name=output_file,
            count_A1=False,
        )
        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "no_sid_list_0")

    def test_no_sid_list_1(self):
        logging.info("TestEpistasis test_no_sid_list_1")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("no_sid_list_1")
        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_1=["1_4"],
            count_A1=False,
        )

        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        write(sid0, sid1, pvalue_list, output_file)
        self.compare_files(
            sid1, sid0, pvalue_list, "no_sid_list_0"
        )  # Swap order of sid0 and sid1

    def test_no_cov(self):
        logging.info("TestEpistasis test_no_cov")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn

        output_file = self.file_name("no_cov")
        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            sid_list_0=test_snps.sid[:10],  # first 10 snps
            sid_list_1=test_snps.sid[5:15],  # Skip 5 snps, use next 10
            output_file_name=output_file,
            count_A1=False,
        )

        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "no_cov")

    def test_no_cov_b(self):
        logging.info("TestEpistasis test_no_cov_b")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn

        output_file = self.file_name("no_cov_b")
        covar = pstpheno.loadPhen(self.cov_fn)
        covar["vals"] = np.delete(covar["vals"], np.s_[:], 1)  # Remove all the columns

        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=test_snps.sid[:10],  # first 10 snps
            sid_list_1=test_snps.sid[5:15],  # Skip 5 snps, use next 10
            output_file_name=output_file,
            count_A1=False,
        )

        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "no_cov")

    def test_G1(self):
        logging.info("TestEpistasis test_G1")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("G1")
        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=test_snps.sid[:10],  # first 10 snps
            sid_list_1=test_snps.sid[5:15],  # Skip 5 snps, use next 10
            G1=test_snps,
            mixing=0.5,
            output_file_name=output_file,
            count_A1=False,
        )

        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "G1")

    def test_G1b(self):
        logging.info("TestEpistasis test_G1b")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("G1b")
        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=test_snps.sid[:10],  # first 10 snps
            sid_list_1=test_snps.sid[5:15],  # Skip 5 snps, use next 10
            G1=test_snps,
            mixing=0.5,
            output_file_name=output_file,
            count_A1=False,
        )

        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "G1")

    def test_G1_mixing(self):
        logging.info("TestEpistasis test_G1_mixing")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("G1_mixing")
        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=test_snps.sid[:10],  # first 10 snps
            sid_list_1=test_snps.sid[5:15],  # Skip 5 snps, use next 10
            G1=test_snps,
            mixing=0,
            output_file_name=output_file,
            count_A1=False,
        )

        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "one")

    # def test_REML_delta(self):
    #    logging.info("TestEpistasis test_REML_delta")

    #    from pysnptools.snpreader import Bed
    #    snps = Bed(os.path.join(self.pythonpath, "tests/datasets/selecttest/snps.bed"),count_A1=False)
    #    pheno = os.path.join(self.pythonpath, "tests/datasets/selecttest/pheno.txt")
    #    covar = os.path.join(self.pythonpath, "tests/datasets/selecttest/covariate.txt")
    #    sim_sid = ["snp26250_m0_.19m1_.19","snp82500_m0_.28m1_.28","snp63751_m0_.23m1_.23","snp48753_m0_.4m1_.4",
    # "snp45001_m0_.26m1_.26","snp52500_m0_.05m1_.05","snp75002_m0_.39m1_.39","snp41253_m0_.07m1_.07","snp11253_m0_.2m1_.2",
    # "snp86250_m0_.33m1_.33","snp3753_m0_.23m1_.23","snp75003_m0_.32m1_.32","snp30002_m0_.25m1_.25","snp26252_m0_.19m1_.19",
    # "snp67501_m0_.15m1_.15","snp63750_m0_.28m1_.28","snp30001_m0_.28m1_.28","snp52502_m0_.35m1_.35","snp33752_m0_.31m1_.31",
    # "snp37503_m0_.37m1_.37","snp15002_m0_.11m1_.11","snp3751_m0_.34m1_.34","snp7502_m0_.18m1_.18","snp52503_m0_.3m1_.3",
    # "snp30000_m0_.39m1_.39","isnp4457_m0_.11m1_.11","isnp23145_m0_.2m1_.2","snp60001_m0_.39m1_.39","snp33753_m0_.16m1_.16",
    # "isnp60813_m0_.2m1_.2","snp82502_m0_.34m1_.34","snp11252_m0_.13m1_.13"]
    #    sim_idx = snps.sid_to_index(sim_sid)
    #    test_sid = ["snp26250_m0_.19m1_.19","snp63751_m0_.23m1_.23","snp82500_m0_.28m1_.28","snp48753_m0_.4m1_.4",
    # "snp45001_m0_.26m1_.26","snp52500_m0_.05m1_.05","snp75002_m0_.39m1_.39","snp41253_m0_.07m1_.07","snp86250_m0_.33m1_.33"
    # ,"snp15002_m0_.11m1_.11","snp33752_m0_.31m1_.31","snp26252_m0_.19m1_.19","snp30001_m0_.28m1_.28","snp11253_m0_.2m1_.2",
    # snp67501_m0_.15m1_.15","snp3753_m0_.23m1_.23","snp52502_m0_.35m1_.35","snp30000_m0_.39m1_.39","snp30002_m0_.25m1_.25"]
    #    test_idx = snps.sid_to_index(test_sid)

    #    output_file = self.file_name("REML_delta")

    #    sid0,sid1,pvalue_list = epistasis(snps[:,test_idx], pheno,covar=covar, G0 = snps[:,sim_idx],log_delta=np.log(1),REML=True, G1=covar, mixing=.5,output_file=output_file,count_A1=False)
    #    self.compare_files(sid0,sid1,pvalue_list,"REML_delta")

    def test_unknown_sid(self):
        logging.info("TestEpistasis test_unknown_sid")

        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        try:
            _frame = epistasis(
                test_snps,
                pheno,
                covar=covar,
                sid_list_0=["1_4", "bogus sid", "1_9"],
                sid_list_1=test_snps.sid[5:15],
                count_A1=False,
            )  # Skip 5 snps, use next 10
            failed = False
        except Exception:
            failed = True

        assert failed

    def test_cid_intersect(self):
        logging.info("TestEpistasis test_cid_intersect")
        from pysnptools.snpreader import Bed

        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = pstpheno.loadOnePhen(self.phen_fn, vectorize=True)
        pheno["iid"] = np.vstack([pheno["iid"][::-1], [["Bogus", "Bogus"]]])
        pheno["vals"] = np.hstack([pheno["vals"][::-1], [-34343]])

        covar = self.cov_fn
        output_file = self.file_name("cid_intersect")
        frame = epistasis(
            test_snps,
            pheno,
            G0=test_snps,
            covar=covar,
            sid_list_0=test_snps.sid[:10],  # first 10 snps
            sid_list_1=test_snps.sid[5:15],  # Skip 5 snps, use next 10
            output_file_name=output_file,
            count_A1=False,
        )

        sid0, sid1, pvalue_list = (
            np.array(frame["SNP0"]),
            np.array(frame["SNP1"]),
            np.array(frame["PValue"]),
        )
        self.compare_files(sid0, sid1, pvalue_list, "one")

    def compare_files(self, sid0_list, sid1_list, pvalue_list, ref_base):
        reffile = TestFeatureSelection.reference_file("epistasis/" + ref_base + ".txt")

        pair_to_pvalue = {}
        for index, sid0 in enumerate(sid0_list):
            sid1 = sid1_list[index]
            if sid0 < sid1:
                key = (sid0, sid1)
            else:
                key = (sid1, sid0)
            pair_to_pvalue[key] = pvalue_list[index]

        reference = np.loadtxt(
            reffile, dtype="str", delimiter="\t", comments=None, skiprows=1
        )
        assert len(pvalue_list) == len(
            reference
        ), "# of pairs differs from file '{0}'".format(reffile)
        for row in reference:
            sid0 = row[0]
            sid1 = row[4]
            if sid0 < sid1:
                key = (sid0, sid1)
            else:
                key = (sid1, sid0)

            assert (
                abs(float(row[8]) - pair_to_pvalue[key]) < 1e-5
            ), "pair {0} differs too much from file '{1}'".format(key, reffile)

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)) + "/..")
        result = doctest.testmod(sys.modules["fastlmm.association.epistasis"])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


def getTestSuite():
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestEpistasis)
    return unittest.TestSuite([suite1])


if __name__ == "__main__":

    from fastlmm.association.tests.testepistasis import TestEpistasis

    suites = unittest.TestSuite([getTestSuite()])

    if True:  # Standard test run
        r = unittest.TextTestRunner(failfast=False)
        ret = r.run(suites)
        assert ret.wasSuccessful()
    # else:  # Cluster test run
    #     from pysnptools.util.mapreduce1.distributabletest import DistributableTest

    #     runner = HPC(
    #         10,
    #         "RR1-N13-09-H44",
    #         r"\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\Redmond",
    #         remote_python_parent=r"\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\REDMOND\carlk\Source\carlk\july_7_14\tests\runs\2014-07-24_15_02_02_554725991686\pythonpath",
    #         update_remote_python_parent=True,
    #         priority="AboveNormal",
    #         mkl_num_threads=1,
    #     )
    #     runner = Local()
    #     # runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
    #     # runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
    #     # runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
    #     distributable_test = DistributableTest(suites, "temp_test")
    #     print(runner.run(distributable_test))

    logging.info("done with testing")
