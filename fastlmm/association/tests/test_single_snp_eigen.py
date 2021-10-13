import logging
import datetime
import shutil

#!!!cmk kludge replace dot with @
logging.basicConfig(level=logging.DEBUG)  # cmk
import unittest
from pathlib import Path
import os.path
import numpy as np

from pysnptools.snpreader import Bed, Pheno, SnpData
from pysnptools.kernelstandardizer import Identity as KernelIdentity
from pysnptools.util.mapreduce1.runner import LocalMultiProc, Local
from pysnptools.util.mapreduce1 import map_reduce
from pysnptools.eigenreader import EigenData, EigenMemMap

from fastlmm.util import example_file  # Download and return local file name
from fastlmm.association import single_snp_eigen


class TestSingleSnpEigen(unittest.TestCase):
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

    tempout_dir = "tempout/single_snp_eigen"

    def file_name(self, testcase_name):
        temp_fn = os.path.join(self.tempout_dir, testcase_name + ".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    #!!!cmk0 make faster if possible by swapping UX and X
    #!!!cmk0 understand use_reml vs not
    def cmk0test_big_file(self):
        iid_count = 100_000
        sid_count = 10_000
        rng = np.random.RandomState(3343)

        #!!!cmk can't really run this
        bed = Bed(r"M:\deldir\genbgen\2\merged_487400x220000.1.bed",count_A1=False)[:iid_count,:sid_count]
        #print(bed.shape)
        #print(set(bed.pos[:,0]))
        #print("!!!cmk")

        #!!!cmk0 add some chroms
        pheno5 = SnpData(val=rng.random((bed.iid_count,5)),iid=bed.iid,sid=[f"pheno{i}" for i in range(5)])
        covar3 = SnpData(val=rng.random((bed.iid_count,3)),iid=bed.iid,sid=[f"covar{i}" for i in range(3)])

        ####### Runner ###############
        runner = LocalMultiProc(6, just_one_process=False)

        eigen_path = Path(f"m:/deldir/eigentest/{iid_count}.eigen.memmap")
        eigen_path_temp = Path(f"{eigen_path}.temp")
        if not eigen_path.exists():
            logging.info(f"about to create empty '{eigen_path}'")
            eigenmemmap = EigenMemMap.empty(iid=bed.iid,values=rng.random((iid_count)),filename=str(eigen_path_temp))
            step = int(np.ceil(iid_count**.5))
            for start in range(0,iid_count,step):
                logging.info(f"filling in random values {start:,d}:{iid_count:,d},{step:,d}")
                end = min(iid_count, start+step)
                width = int(np.ceil((end-start)**.5))
                random_bit = rng.random((iid_count,width))
                eigenmemmap.val[:,start:start+width] = random_bit
                #eigenmemmap.val[start:start+width,:] = random_bit.T
            logging.info(f"flushing")
            eigenmemmap.flush()
            shutil.move(eigen_path_temp, eigen_path) #!!!cmk what if alreay there?

        eigen = EigenMemMap(eigen_path)


        K0_eigen_by_chrom = {1: eigen}
        batch_size = int(iid_count**.5)
        cache_folder = Path(r"M:\deldir\eigentest\bigtest" + f"/{iid_count}")
        cache_folder.parent.mkdir(parents=True,exist_ok=True)


        frame = single_snp_eigen(
            test_snps=bed,
            pheno=pheno5,
            K0_eigen_by_chrom=K0_eigen_by_chrom,
            covar=covar3,
            output_file_name=None,
            log_delta=None,
            find_delta_via_reml=True,
            test_via_reml=False,
            batch_size=batch_size,
            cache_folder=cache_folder,
            stop_early=None, #stop_early,
            runner=runner,
        )

        print(frame)


    def test_same_as_old_code(self):  #!!!cmk too slow???
        test_count = 750

        bed_fn = example_file(
            "fastlmm/feature_selection/examples/toydata.5chrom.*", "*.bed"
        )
        snps_reader1 = Bed(bed_fn, count_A1=False)
        snps_reader5 = snps_reader1.read()
        snps_reader5.pos[:, 0] = [i % 5 + 1 for i in range(snps_reader5.sid_count)]

        pheno_fn = example_file("fastlmm/feature_selection/examples/toydata.phe")
        pheno0 = Pheno(pheno_fn).read()
        pheno000 = SnpData(
            val=np.repeat(pheno0.val, 3, axis=1),
            iid=pheno0.row,
            sid=["pheno0a", "pheno0b", "pheno0c"],
            name="pheno000",
        )
        val01 = np.repeat(pheno0.val, 2, axis=1)
        val01[::2, 1] *= 10
        pheno01 = SnpData(
            val=val01, iid=pheno0.row, sid=["pheno0", "pheno1"], name="pheno01"
        )

        cov_reader = Pheno(
            example_file("fastlmm/feature_selection/examples/toydata.cov")
        )

        date_str = str(datetime.datetime.now())[0:16]
        i = 0
        while True:
            cache_file0 = Path(r"m:/deldir/eigentest") / f"{date_str}.{i}".replace(
                ":", "-"
            )
            if not cache_file0.exists():
                break
            i += 1
        cache_file0.mkdir(parents=True)

        value_dict = {
            "False": False,
            "True": True,
            "50": 50,
            "750": 750,
            "None": None,
            "cov_reader": cov_reader,
            "1.0": 1.0,
            "0.20000600000000002": 0.20000600000000002,
            "pheno_fn": pheno_fn,
            "pheno01": pheno01,
            "pheno000": pheno000,
            "snps_reader1": snps_reader1,
            "snps_reader5": snps_reader5,
            "7": 7,
            "100_000": 100_000,
            "cache_file0": cache_file0,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
        }

        add_for_coverage = [
            {
                "use_reml": "True",
                "cache_file": "cache_file0",
                "stop_early": "3",
            }
        ]

        matrix = {
            "use_reml": ["False", "True"],
            "train_count": ["50", "750"],
            "cov": ["None", "cov_reader"],
            "delta": ["None", "1.0", "0.20000600000000002"],
            "pheno": ["pheno_fn", "pheno01", "pheno000"],
            "snps_reader": ["snps_reader1", "snps_reader5"],
            "batch_size": ["None", "7", "100_000"],
            "cache_file": ["None", "cache_file0"],
            "stop_early": ["None", "0", "1", "2", "3"],
        }

        if True:
            test_runner = None  # LocalMultiProc(6, just_one_process=True)
            runner = Local()  # LocalMultiProc(6, just_one_process=True)
            exception_to_catch = TimeoutError  # Exception #
            extra_lambda = lambda case_number: 0
        else:
            test_runner = LocalMultiProc(6, just_one_process=False)
            runner = None
            exception_to_catch = Exception
            extra_lambda = lambda case_number: case_number ** 0.5
        first_list = [
            {
                "use_reml": "True",
                "train_count": "750",
                "cov": "None",
                "delta": "None",
                "pheno": "pheno01",
                "snps_reader": "snps_reader1",
                "batch_size": "100_000",
                "cache_file": "cache_file0",
                "stop_early": "3",
            }
        ] + add_for_coverage

        def mapper2(index_total_option):
            index, total, option = index_total_option
            print(f"============{index} of {total}==================")
            print(f"==={option}")

            use_reml = value_dict[option["use_reml"]]
            cov = value_dict[option["cov"]]
            train_count = value_dict[option["train_count"]]
            delta = value_dict[option["delta"]]
            pheno = value_dict[option["pheno"]]
            snps_reader = value_dict[option["snps_reader"]]
            batch_size = value_dict[option["batch_size"]]
            cache_file = value_dict[option["cache_file"]]
            stop_early = value_dict[option["stop_early"]]

            try:
                import numpy as np
                from pysnptools.snpreader import Bed, Pheno, SnpData
                from pysnptools.kernelstandardizer import Identity as KernelIdentity
                from pysnptools.util.mapreduce1 import map_reduce
                from fastlmm.association import single_snp_eigen
                from fastlmm.association.single_snp_eigen import eigen_from_kernel

                if cache_file is not None:
                    cache_file2 = cache_file / f"test{index}"
                else:
                    cache_file2 = None

                # !!!cmk why not diag standardize?
                # The [:,:] stops it from being an in-memory EigenData
                K0_eigen = eigen_from_kernel(
                    snps_reader[:, :train_count],
                    kernel_standardizer=KernelIdentity(),
                )[:, :]
                test_snps = snps_reader[:, train_count : train_count + test_count]

                #!!!cmk kludge need to test with different eigens
                K0_eigen_by_chrom = {
                    chrom: K0_eigen for chrom in set(test_snps.pos[:, 0])
                }

                while True:
                    frame = single_snp_eigen(
                        test_snps=test_snps,
                        pheno=pheno,
                        K0_eigen_by_chrom=K0_eigen_by_chrom,
                        covar=cov,
                        output_file_name=None,
                        log_delta=np.log(delta) if delta is not None else None,
                        find_delta_via_reml=use_reml,
                        test_via_reml=use_reml,
                        count_A1=False,
                        batch_size=batch_size,
                        cache_folder=cache_file2,
                        stop_early=stop_early,
                        runner=runner,
                    )

                    if stop_early is None:
                        break
                    stop_early = None

                G = snps_reader.read().standardize().val
                if cov is not None:
                    cov_val = np.c_[cov.read().val, np.ones((cov.iid_count, 1))]
                else:
                    cov_val = None
                G_chr1, G_chr2 = (
                    G[:, :train_count],
                    G[:, train_count : train_count + test_count],
                )

                phenox = pheno if pheno is not pheno_fn else Pheno(pheno_fn)

                def mapper(pheno_index):
                    from fastlmm.association.tests.test_gwas import (
                        GwasPrototype,
                    )

                    y = phenox.read().val[:, pheno_index]
                    gwas = GwasPrototype(
                        G_chr1,
                        G_chr2,
                        y,
                        internal_delta=delta,
                        cov=cov_val,
                        REML=use_reml,
                    )
                    gwas.run_gwas()
                    return sorted(gwas.p_values)

                gwas_pvalues_list = map_reduce(
                    range(phenox.sid_count), mapper=mapper, runner=runner
                )
                # cmk kludge rename phenox
                for pheno_index in range(phenox.sid_count):
                    frame_i = frame[frame["Pheno"] == phenox.sid[pheno_index]]
                    # check p-values in log-space!
                    np.testing.assert_array_almost_equal(
                        np.log(gwas_pvalues_list[pheno_index]),
                        np.log(frame_i.PValue),  #!!!cmk
                        decimal=7,
                    )
            except exception_to_catch as e:
                print(str(e))
                return option
            return None

        def reducer2(bad_option_list):
            result = True
            for bad_option in bad_option_list:
                if bad_option is not None:
                    print(bad_option)
                    result = False
            return result

        is_ok = map_reduce(
            list(
                matrix_combo(
                    matrix,
                    seed=10234,
                    extra_lambda=extra_lambda,
                    first_list=first_list,
                )
            ),
            mapper=mapper2,
            reducer=reducer2,
            runner=test_runner,
        )
        assert is_ok

    def cmktest_one(self):
        logging.info("TestSingleSnpEigen test_one")
        test_snps = Bed(self.bedbase, count_A1=False)
        pheno = self.phen_fn
        covar = self.cov_fn

        output_file = self.file_name("one")
        eigenvalues, eigenvectors = EigenData.from_kernel(
            test_snps, kernel_standardizer=KernelIdentity()
        )  # !!!cmk why not diag standardize?
        frame = single_snp_eigen(
            test_snps=test_snps[:, :10],
            pheno=pheno,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            covar=covar,
            output_file_name=output_file,
            find_delta_via_reml=False,
            test_via_reml=False,
            count_A1=False,
        )

        self.compare_files(frame, "one")


#    def test_zero_pheno(self):
#        logging.info("TestSingleSnp test_zero_pheno")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = Pheno(self.phen_fn)[:,0:0]
#        covar = self.cov_fn

#        got_expected_fail = False
#        try:
#            frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
#                                      G0=test_snps, covar=covar,
#                                      count_A1=False
#                                      )
#        except Exception as e:
#            got_expected_fail = True
#        assert got_expected_fail, "Did not get expected fail"


#    def test_missing_covar(self):
#        logging.info("TestSingleSnp test_missing_covar")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = Pheno(self.cov_fn).read()
#        covar.val[0,0] = np.nan

#        got_expected_fail = False
#        try:
#            frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
#                                      G0=test_snps, covar=covar,
#                                      count_A1=False
#                                      )
#        except Exception as e:
#            got_expected_fail = True
#        assert got_expected_fail, "Did not get expected fail"

#        covar_by_chrom = {chrom:covar for chrom in set(test_snps.pos[:,0])}
#        got_expected_fail = False
#        try:
#            frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=True,
#                                      G0=test_snps, covar_by_chrom=covar_by_chrom,
#                                      count_A1=False
#                                      )
#        except Exception as e:
#            got_expected_fail = True
#        assert got_expected_fail, "Did not get expected fail"


#    def test_thres(self):
#        logging.info("TestSingleSnp test_thres")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        reffile = TestFeatureSelection.reference_file("single_snp/one.txt")
#        reference=pd.read_csv(reffile,delimiter='\s',comment=None,engine='python')

#        for random_seed in [0,1]:
#            for pvalue_threshold in [.5, None, 1.0]:
#                for random_threshold in [.5, None, 1.0]:
#                    frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
#                                              G0=test_snps, covar=covar, pvalue_threshold=pvalue_threshold, random_threshold=random_threshold,
#                                              random_seed=random_seed,
#                                              count_A1=False
#                                              )

#                    assert len(frame) <= len(reference), "# of pairs differs from file '{0}'".format(reffile)
#                    if len(frame) < len(reference):
#                        assert frame['PValueCount'].iloc[0] == len(reference), "row_count doesn't match the reference # of rows"
#                    if random_threshold is not None:
#                        assert np.all(frame['RandomThreshold']==random_threshold), "Expect all rows to have 'RandomThreshold'"
#                        assert np.all(frame['RandomSeed']==random_seed), "Expect all rows to have right 'RandomSeed'"
#                        if pvalue_threshold is not None:
#                            assert np.all((frame['RandomValue']<=random_threshold) + (frame['PValue']<=pvalue_threshold)), "Expect all rows have random value or pvalue less than threshold"
#                    if pvalue_threshold is not None:
#                        assert np.all(frame['PValueThreshold']==pvalue_threshold), "Expect all rows to have 'PValueThreshold'"
#                    for _, row in frame.iterrows():
#                        sid = row.SNP
#                        pvalue = reference[reference['SNP'] == sid].iloc[0].PValue
#                        diff = abs(row.PValue - pvalue)
#                        if diff > 1e-5 or np.isnan(diff):
#                            raise Exception("pair {0} differs too much from file '{1}'".format(sid,reffile))
#                        assert abs(row.PValue - pvalue) < 1e-5, "wrong"


#    def test_linreg(self):
#        logging.info("TestSingleSnp test_linreg")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("linreg")

#        frame1 = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
#                                    G0=KernelIdentity(iid=test_snps.iid), covar=covar,
#                                    output_file_name=output_file,count_A1=False
#                                    )

#        frame1 = frame1[['sid_index', 'SNP', 'Chr', 'GenDist', 'ChrPos', 'PValue']]
#        self.compare_files(frame1,"linreg")

#        with patch.dict('os.environ', {'ARRAY_MODULE': 'numpy'}) as _:
#            frame2 = single_snp_linreg(test_snps=test_snps[:,:10], pheno=pheno,
#                                        covar=covar,
#                                        output_file_name=output_file
#                                        )
#        self.compare_files(frame2,"linreg")

#    def test_noK0(self):
#        logging.info("TestSingleSnp test_noK0")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("noK0")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=1,leave_out_one_chrom=False,
#                                  G1=test_snps, covar=covar,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"one")


#    def test_gb_goal(self):
#        logging.info("TestSingleSnp test_gb_goal")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("gb_goal")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
#                                  G0=test_snps, covar=covar, GB_goal=0,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"one")

#        output_file = self.file_name("gb_goal2")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
#                                  G0=test_snps, covar=covar, GB_goal=.12,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"one")

#    def test_other(self):
#        logging.info("TestSingleSnp test_other")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("other")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,leave_out_one_chrom=False,
#                                  K1=test_snps, covar=covar,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"one")

#    def test_none(self):
#        logging.info("TestSingleSnp test_none")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("none")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
#                                  K0=KernelIdentity(test_snps.iid), covar=covar,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"none")

#    def test_interact(self):
#        logging.info("TestSingleSnp test_interact")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("interact")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, mixing=0,leave_out_one_chrom=False,
#                                  G0=test_snps, covar=covar, interact_with_snp=1,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"interact")

#    def test_preload_files(self):
#        logging.info("TestSingleSnp test_preload_files")
#        test_snps = self.bedbase
#        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
#        covar = pstpheno.loadPhen(self.cov_fn)
#        bed = Bed(test_snps, count_A1=False)

#        output_file_name = self.file_name("preload_files")

#        frame = single_snp(test_snps=bed[:,:10], pheno=pheno, G0=test_snps, mixing=0,leave_out_one_chrom=False,
#                                  covar=covar, output_file_name=output_file_name,count_A1=False
#                                  )
#        self.compare_files(frame,"one")

#    def test_SNC(self):
#        logging.info("TestSNC")
#        test_snps = self.bedbase
#        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
#        covar = pstpheno.loadPhen(self.cov_fn)
#        bed = Bed(test_snps, count_A1=False)
#        snc = bed.read()
#        snc.val[:,2] = 0 # make SNP #2 have constant values (aka a SNC)

#        output_file_name = self.file_name("snc")

#        frame = single_snp(test_snps=snc[:,:10], pheno=pheno, G0=snc, mixing=0,leave_out_one_chrom=False,
#                                  covar=covar, output_file_name=output_file_name,count_A1=False
#                                  )
#        self.compare_files(frame,"snc")

#    def test_G0_has_reader(self):
#        logging.info("TestSingleSnp test_G0_has_reader")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file_name = self.file_name("G0_has_reader")

#        frame0 = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, leave_out_one_chrom=False,
#                                  covar=covar, mixing=0,
#                                  output_file_name=output_file_name,count_A1=False
#                                  )
#        self.compare_files(frame0,"one")

#        frame1 = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=KernelIdentity(test_snps.iid), G1=test_snps, leave_out_one_chrom=False,
#                                  covar=covar, mixing=1,
#                                  output_file_name=output_file_name,count_A1=False
#                                  )
#        self.compare_files(frame1,"one")

#    def test_no_cov(self):
#        logging.info("TestSingleSnp test_no_cov")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn

#        output_file_name = self.file_name("no_cov")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, mixing=0,leave_out_one_chrom=False,
#                                          output_file_name=output_file_name,count_A1=False
#                                          )

#        self.compare_files(frame,"no_cov")

#    def test_no_cov_b(self):
#        logging.info("TestSingleSnp test_no_cov_b")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn

#        output_file_name = self.file_name("no_cov_b")
#        covar = pstpheno.loadPhen(self.cov_fn)
#        covar['vals'] = np.delete(covar['vals'], np.s_[:],1) #Remove all the columns
#        covar['header'] = []

#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, leave_out_one_chrom=False,
#                                  covar=covar, mixing=0,
#                                  output_file_name=output_file_name,count_A1=False
#                                  )

#        self.compare_files(frame,"no_cov")

#    def test_G1(self):
#        logging.info("TestSingleSnp test_G1")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file_name = self.file_name("G1")
#        for force_full_rank,force_low_rank in [(False,True),(False,False),(True,False)]:
#            logging.info("{0},{1}".format(force_full_rank,force_low_rank))
#            frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], leave_out_one_chrom=False,
#                                          covar=covar, G1=test_snps[:,100:200],
#                                          mixing=.5,force_full_rank=force_full_rank,force_low_rank=force_low_rank,
#                                          output_file_name=output_file_name,count_A1=False
#                                          )
#            self.compare_files(frame,"G1")


#    def test_file_cache(self):
#        logging.info("TestSingleSnp test_file_cache")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file_name = self.file_name("G1")
#        cache_file = self.file_name("cache_file")+".npz"
#        if os.path.exists(cache_file):
#            os.remove(cache_file)
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], leave_out_one_chrom=False,
#                                      covar=covar, G1=test_snps[:,100:200],
#                                      mixing=.5,
#                                      output_file_name=output_file_name,
#                                      cache_file = cache_file,count_A1=False
#                                      )
#        self.compare_files(frame,"G1")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno,G0=test_snps[:,10:100], leave_out_one_chrom=False,
#                                      covar=covar, G1=test_snps[:,100:200],
#                                      mixing=.5,
#                                      output_file_name=output_file_name,
#                                      cache_file = cache_file,count_A1=False
#                                      )
#        self.compare_files(frame,"G1")


#    def test_G1_mixing(self):
#        logging.info("TestSingleSnp test_G1_mixing")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file_name = self.file_name("G1_mixing")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, leave_out_one_chrom=False,
#                                      covar=covar,
#                                      G1=test_snps[:,100:200],
#                                      mixing=0,
#                                      output_file_name=output_file_name,count_A1=False
#                                      )

#        self.compare_files(frame,"one")

#    def test_unknown_sid(self):
#        logging.info("TestSingleSnp test_unknown_sid")

#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        try:
#            frame = single_snp(test_snps=test_snps,G0=test_snps,pheno=pheno,leave_out_one_chrom=False,mixing=0,covar=covar,sid_list=['1_4','bogus sid','1_9'],count_A1=False)
#            failed = False
#        except:
#            failed = True

#        assert(failed)

#    def test_cid_intersect(self):
#        logging.info("TestSingleSnp test_cid_intersect")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = pstpheno.loadOnePhen(self.phen_fn,vectorize=True)
#        pheno['iid'] = np.vstack([pheno['iid'][::-1],[['Bogus','Bogus']]])
#        pheno['vals'] = np.hstack([pheno['vals'][::-1],[-34343]])


#        covar = self.cov_fn
#        output_file_name = self.file_name("cid_intersect")
#        frame = single_snp(test_snps=test_snps[:,:10], pheno=pheno, G0=test_snps, leave_out_one_chrom=False,
#                                  covar=covar, mixing=0,
#                                  output_file_name=output_file_name,count_A1=False
#                                  )

#        self.compare_files(frame,"one")

#    def compare_files(self,frame,ref_base):
#        reffile = TestFeatureSelection.reference_file("single_snp/"+ref_base+".txt")

#        #sid_list,pvalue_list = frame['SNP'].values,frame['Pvalue'].values

#        #sid_to_pvalue = {}
#        #for index, sid in enumerate(sid_list):
#        #    sid_to_pvalue[sid] = pvalue_list[index]

#        reference=pd.read_csv(reffile,delimiter='\s',comment=None,engine='python')
#        assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(reffile)
#        for _, row in reference.iterrows():
#            sid = row.SNP
#            pvalue = frame[frame['SNP'] == sid].iloc[0].PValue
#            diff = abs(row.PValue - pvalue)
#            if diff > 1e-5 or np.isnan(diff):
#                raise Exception("pair {0} differs too much from file '{1}'".format(sid,reffile))
#            assert abs(row.PValue - pvalue) < 1e-5, "wrong"

#    def test_doctest(self):
#        old_dir = os.getcwd()
#        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
#        result = doctest.testmod(sys.modules['fastlmm.association.single_snp'])
#        os.chdir(old_dir)
#        assert result.failed == 0, "failed doc test: " + __file__

# class TestSingleSnpLeaveOutOneChrom(unittest.TestCase):

#    @classmethod
#    def setUpClass(self):
#        from pysnptools.util import create_directory_if_necessary
#        create_directory_if_necessary(self.tempout_dir, isfile=False)
#        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
#        self.bedbase = os.path.join(self.pythonpath, 'fastlmm/feature_selection/examples/toydata.5chrom.bed')
#        self.phen_fn = os.path.join(self.pythonpath, 'fastlmm/feature_selection/examples/toydata.phe')
#        self.cov_fn = os.path.join(self.pythonpath,  'fastlmm/feature_selection/examples/toydata.cov')

#    tempout_dir = "tempout/single_snp"

#    def file_name(self,testcase_name):
#        temp_fn = os.path.join(self.tempout_dir,testcase_name+".txt")
#        if os.path.exists(temp_fn):
#            os.remove(temp_fn)
#        return temp_fn

#    def test_leave_one_out_with_prekernels(self):
#        logging.info("TestSingleSnpLeaveOutOneChrom test_leave_one_out_with_prekernels")
#        from pysnptools.kernelstandardizer import DiagKtoN
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        chrom_to_kernel = {}
#        with patch.dict('os.environ', {'ARRAY_MODULE': 'numpy'}) as _:
#            for chrom in np.unique(test_snps.pos[:,0]):
#                other_snps = test_snps[:,test_snps.pos[:,0]!=chrom]
#                kernel = other_snps.read_kernel(standardizer=Unit(),block_size=500) #Create a kernel from the SNPs not used in testing
#                chrom_to_kernel[chrom] = kernel.standardize(DiagKtoN()) #improves the kernel numerically by making its diagonal sum to iid_count

#        output_file = self.file_name("one_looc_prekernel")
#        frame = single_snp(test_snps, pheno,
#                                  covar=covar,
#                                  K0 = chrom_to_kernel,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"one_looc")

#    def test_one_looc(self):
#        logging.info("TestSingleSnpLeaveOutOneChrom test_one_looc")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("one_looc")
#        frame = single_snp(test_snps, pheno,
#                                  covar=covar, mixing=0,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"one_looc")

#    def test_runner(self):
#        logging.info("TestRunner")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        runner = LocalMultiProc(6,just_one_process=True)
#        for map_reduce_outer  in [True,False]:
#            frame = single_snp(test_snps, pheno,
#                                      covar=covar, mixing=0,
#                                      count_A1=False,
#                                      runner=runner, map_reduce_outer=map_reduce_outer
#                                      )

#            self.compare_files(frame,"one_looc")


#    def test_multipheno(self):
#        logging.info("TestSingleSnpLeaveOutOneChrom test_multipheno")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        pheno2 = Pheno(pheno).read()
#        pheno2.val[0,0] = 100
#        pheno2.val[1,0] = -100

#        cache_file = None

#        if True:
#            pheno12 = SnpData(iid=pheno2.iid,sid=["pheno1","pheno2"],val = np.c_[Pheno(pheno).read().val,pheno2.val])
#            output_file = self.file_name("multipheno12")
#            frame = single_snp(test_snps[:,::10], pheno12,
#                               force_full_rank=True,
#                                      covar=covar,
#                                      cache_file=cache_file,
#                                      output_file_name=output_file
#                                      )
#            frame1 = frame[frame['Pheno']=='pheno1']
#            del frame1['Pheno']
#            self.compare_files(frame1,"two_looc")

#            frame2 = frame[frame['Pheno']=='pheno2']
#            del frame2['Pheno']
#            self.compare_files(frame2,"multipheno2")


#        if True:
#            pheno11 = SnpData(iid=pheno2.iid,sid=["pheno1a","pheno1b"],val = np.c_[Pheno(pheno).read().val,Pheno(pheno).read().val])
#            output_file = self.file_name("multipheno11")
#            frame = single_snp(test_snps[:,::10], pheno11,
#                               force_full_rank=True,
#                                      covar=covar,
#                                      output_file_name=output_file,count_A1=False
#                                      )

#            frame1 = frame[frame['Pheno']=='pheno1a']
#            del frame1['Pheno']
#            self.compare_files(frame1,"two_looc")

#            frame2 = frame[frame['Pheno']=='pheno1b']
#            del frame2['Pheno']
#            self.compare_files(frame2,"two_looc")


#        if True:
#            output_file = self.file_name("multipheno1")
#            frame = single_snp(test_snps[:,::10], pheno,
#                               force_full_rank=True,
#                                      covar=covar,
#                                      output_file_name=output_file,count_A1=False
#                                      )

#            self.compare_files(frame,"two_looc")


#        if True:
#            output_file = self.file_name("multipheno2")
#            frame = single_snp(test_snps[:,::10], pheno2,
#                               force_full_rank=True,
#                                      covar=covar,
#                                      output_file_name=output_file,count_A1=False
#                                      )

#            self.compare_files(frame,"multipheno2")


#        if True:
#            pheno22 = SnpData(iid=pheno2.iid,sid=["pheno2a","pheno2b"],val = np.c_[pheno2.val, pheno2.val])
#            output_file = self.file_name("multipheno22")
#            frame = single_snp(test_snps[:,::10], pheno22,
#                               force_full_rank=True,
#                                      covar=covar,
#                                      output_file_name=output_file,count_A1=False
#                                      )
#            frame1 = frame[frame['Pheno']=='pheno2a']
#            del frame1['Pheno']
#            self.compare_files(frame1,"multipheno2")

#            frame2 = frame[frame['Pheno']=='pheno2b']
#            del frame2['Pheno']
#            self.compare_files(frame2,"multipheno2")

#    def test_multipheno2(self):
#        logging.info("test_multipheno")
#        from fastlmm.util import example_file # Download and return local file name

#        bed = Bed(example_file('tests/datasets/synth/all.*','*.bed'),count_A1=True)[:,::10]
#        phen_fn = example_file("tests/datasets/synth/pheno_10_causals.txt")
#        cov_fn = example_file("tests/datasets/synth/cov.txt")


#        random_state =  RandomState(29921)
#        pheno_reference = Pheno(phen_fn).read()
#        for pheno_count in [2,5,1]:
#            val = random_state.normal(loc=pheno_count,scale=pheno_count,size=(pheno_reference.iid_count,pheno_count))
#            pheno_col = ['pheno{0}'.format(i) for i in range(pheno_count)]
#            pheno_multi = SnpData(iid=pheno_reference.iid,sid=pheno_col,val=val)

#            reference = pd.concat([single_snp(test_snps=bed, pheno=pheno_multi[:,pheno_index], covar=cov_fn) for pheno_index in range(pheno_count)])

#            for force_full_rank, force_low_rank in [(True,False),(False,True),(False,False)]:

#                frame = single_snp(test_snps=bed, pheno=pheno_multi, covar=cov_fn,
#                                  force_full_rank=force_full_rank, force_low_rank=force_low_rank)

#                assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(reffile)
#                for sid in sorted(set(reference.SNP)): #This ignores which pheno produces which pvalue
#                    pvalue_frame = np.array(sorted(frame[frame['SNP'] == sid].PValue))
#                    pvalue_reference = np.array(sorted(reference[reference['SNP'] == sid].PValue))
#                    assert (abs(pvalue_frame - pvalue_reference) < 1e-5).all, "pair {0} differs too much from reference".format(sid)

#    def create_phen3(self,phen):
#        from fastlmm.util import example_file # Download and return local file name
#        phen = phen.read()
#        rng = np.random.RandomState(seed=0)
#        val1 = phen.val.copy()
#        rng.shuffle(val1)
#        val2 = phen.val.copy()
#        rng.shuffle(val2)
#        phen3 = SnpData(iid=phen.iid,sid=["phen0","phen1","phen2"],
#                        val = np.c_[phen.val, val1, val2]
#                        )
#        return phen3

#    def test_multipheno3(self):
#        from pysnptools.kernelreader import SnpKernel
#        from fastlmm.util import example_file # Download and return local file name
#        from pysnptools.standardizer import Standardizer, Unit

#        bed = Bed(example_file('tests/datasets/synth/all.*','*.bed'),count_A1=True)[:,::10]
#        phen3 = self.create_phen3(Pheno(example_file("tests/datasets/synth/pheno_10_causals.txt")))

#        combo_index = 0
#        for covar,interact in [(None,None),
#                               (Pheno(example_file("tests/datasets/synth/cov.txt")),None),
#                               (Pheno(example_file("tests/datasets/synth/cov.txt")),0)]:
#            for force_full_rank, force_low_rank in [(True,False),(False,True),(False,False)]:
#                for k0_as_snps in [True,False]:
#                    logging.info(f"combo_index {combo_index}")
#                    combo_index += 1
#                    k0 = bed
#                    if not k0_as_snps:
#                        k0 = SnpKernel(k0,standardizer=Unit())

#                    for leave_out_one_chrom in [False,True]:

#                        logging.info([covar,interact,force_full_rank,force_low_rank,k0_as_snps,leave_out_one_chrom])

#                        result3 = single_snp(test_snps=bed,pheno=phen3,covar=covar,
#                                                        K0=k0, interact_with_snp=interact,
#                                                        force_full_rank=force_full_rank, force_low_rank=force_low_rank,
#                                                        leave_out_one_chrom=leave_out_one_chrom
#                                                        )

#                        result_list = [single_snp(test_snps=bed,pheno=phen3[:,i],covar=covar,
#                                                        K0=k0, interact_with_snp=interact,
#                                                        force_full_rank=force_full_rank, force_low_rank=force_low_rank,
#                                                        leave_out_one_chrom=leave_out_one_chrom)
#                                        for i in range(3)]

#                        for i in range(3):
#                            self.compare_df(result3[result3["Pheno"]==phen3.sid[i]], result_list[i], "test_multipheno3")

#    def test_multipheno_expected_exceptions(self):

#        from pysnptools.kernelreader import SnpKernel
#        from fastlmm.util import example_file # Download and return local file name
#        from pysnptools.standardizer import Standardizer, Unit

#        bed = Bed(example_file('tests/datasets/synth/all.*','*.bed'),count_A1=True)[:,::10]
#        phen = Pheno(example_file("tests/datasets/synth/pheno_10_causals.txt"))
#        phen3 = self.create_phen3(phen)

#        got_expected_fail = False
#        try:
#            single_snp(test_snps=bed,pheno=phen3,covar=None,
#                                            K0=bed, K1=bed, interact_with_snp=None,
#                                            force_full_rank=False, force_low_rank=False)
#        except Exception as e:
#            assert "2nd kernel" in str(e)
#            got_expected_fail = True
#        assert got_expected_fail, "Did not get expected fail"

#        phen3.val[1,:] = np.nan # Add a missing value to all phenos
#        single_snp(test_snps=bed,pheno=phen3,covar=None,
#                                        K0=bed, interact_with_snp=None,
#                                        force_full_rank=False, force_low_rank=False)
#        phen3.val[0,0] = np.nan # Add a missing value to one pheno, but not the others
#        got_expected_fail = False
#        try:
#            single_snp(test_snps=bed,pheno=phen3,covar=None,
#                                            K0=bed, interact_with_snp=None,
#                                            force_full_rank=False, force_low_rank=False)
#        except Exception as e:
#            assert "multiple phenotypes" in str(e)
#            got_expected_fail = True
#        assert got_expected_fail, "Did not get expected fail"

#    def test_cache(self):
#        test_snpsx = Bed(self.bedbase, count_A1=False)
#        phen1 = self.phen_fn
#        phen3 = self.create_phen3(Pheno(self.phen_fn))
#        covar = self.cov_fn

#        for leave_out_one_chrom, ref_file1, ref_file3, test_snps in [(True,"one_looc","one_looc3", test_snpsx),(False,"one","one3",test_snpsx[:,:10])]:
#            for force_full_rank, force_low_rank in [(False,True),(True,False),(False,False)]:
#                for pheno, ref_file in [(phen3,ref_file3),(phen1,ref_file1)]:
#                    output_file = self.file_name(f"cache{leave_out_one_chrom}{force_full_rank}{force_low_rank}")
#                    cache_file = self.file_name(output_file+"cache")
#                    for p in Path(cache_file).parent.glob(Path(cache_file).name+".*"):
#                        p.unlink()
#                    frame = single_snp(test_snps, pheno,
#                                                cache_file=cache_file,
#                                                covar=covar, mixing=0,
#                                                output_file_name=output_file,count_A1=False,
#                                                leave_out_one_chrom=leave_out_one_chrom,force_full_rank=force_full_rank,force_low_rank=force_low_rank
#                                                )

#                    self.compare_files(frame, ref_file)
#                    frame = single_snp(test_snps, pheno,
#                                                cache_file=cache_file,
#                                                covar=covar, mixing=0,
#                                                output_file_name=output_file,count_A1=False,
#                                                leave_out_one_chrom=leave_out_one_chrom,force_full_rank=force_full_rank,force_low_rank=force_low_rank
#                                                )
#                    self.compare_files(frame, ref_file)


#    def test_two_looc(self):
#        logging.info("TestSingleSnpLeaveOutOneChrom test_two_looc")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("two_looc")
#        frame = single_snp(test_snps[:,::10], pheno,
#                                  covar=covar,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"two_looc")


#    def test_interact_looc(self):
#        logging.info("TestSingleSnpLeaveOutOneChrom test_interact_looc")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn

#        output_file = self.file_name("interact_looc")
#        frame = single_snp(test_snps, pheno,
#                                  covar=covar, mixing=0, interact_with_snp=0,
#                                  output_file_name=output_file,count_A1=False
#                                  )

#        self.compare_files(frame,"interact_looc")

#    def test_covar_by_chrom(self):
#        logging.info("TestSingleSnpLeaveOutOneChrom test_covar_by_chrom")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = Pheno(self.cov_fn).read()
#        covar = SnpData(iid=covar.iid,sid=["pheno-1"],val=covar.val)
#        covar_by_chrom = {chrom:self.cov_fn for chrom in range(1,6)}
#        output_file = self.file_name("covar_by_chrom")
#        frame = single_snp(test_snps, pheno,
#                                    covar=covar, mixing=0,
#                                    covar_by_chrom=covar_by_chrom,
#                                    output_file_name=output_file,count_A1=False
#                                    )

#        self.compare_files(frame,"covar_by_chrom")

#    def test_covar_by_chrom_mixing(self):
#        logging.info("TestSingleSnpLeaveOutOneChrom test_covar_by_chrom_mixing")
#        test_snps = Bed(self.bedbase, count_A1=False)
#        pheno = self.phen_fn
#        covar = self.cov_fn
#        covar = Pheno(self.cov_fn).read()
#        covar = SnpData(iid=covar.iid,sid=["pheno-1"],val=covar.val)
#        covar_by_chrom = {chrom:self.cov_fn for chrom in range(1,6)}
#        output_file = self.file_name("covar_by_chrom_mixing")
#        frame = single_snp(test_snps, pheno,
#                                    covar=covar,
#                                    covar_by_chrom=covar_by_chrom,
#                                    output_file_name=output_file,count_A1=False
#                                    )
#        self.compare_files(frame,"covar_by_chrom_mixing")


#    def compare_files(self,frame,ref_base):
#        reffile = TestFeatureSelection.reference_file("single_snp/"+ref_base+".txt")

#        #sid_list,pvalue_list = frame['SNP'].values,frame['Pvalue'].values

#        #sid_to_pvalue = {}
#        #for index, sid in enumerate(sid_list):
#        #    sid_to_pvalue[sid] = pvalue_list[index]

#        reference=pd.read_csv(reffile,delimiter='\s',comment=None,engine='python')
#        self.compare_df(frame, reference, reffile)

#    def compare_df(self,frame,reference,name):
#        assert len(frame) == len(reference), "# of pairs differs from file '{0}'".format(name)
#        if 'Pheno' not in frame.columns or 'Pheno' not in reference.columns:
#            frame.set_index('SNP',inplace=True)
#            reference.set_index('SNP',inplace=True)
#        else:
#            frame.set_index(['Pheno','SNP'],inplace=True)
#            reference.set_index(['Pheno','SNP'],inplace=True)

#        diff = (frame.PValue-reference.PValue)
#        bad = diff[np.abs(diff)>1e-5]
#        if len(bad) > 0:
#            raise Exception("snps differ too much from file '{0}' at these snps {1}".format(name,bad))

# !!!cmk find similar code. Move to utils, perhaps pysnptools
def matrix_combo(option_matrix, seed, extra_lambda, first_list=[]):
    # https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    import itertools

    rng = np.random.RandomState(seed=seed)
    keys, old_values = zip(*option_matrix.items())
    values = [rng.permutation(value) for value in old_values]
    max_values = max([len(value) for value in values])

    # !!!cmk don't yield same ones again (but does it really matter?)
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    rng.shuffle(permutations_dicts)
    count = int(np.ceil(extra_lambda(len(permutations_dicts))))

    total = len(first_list) + max_values + count
    index = -1
    for dict_of_interest in first_list:
        index += 1
        output = {}
        for key_index, key in enumerate(keys):
            value = values[key_index]
            if key in dict_of_interest:
                value_index_of_interest = dict_of_interest[key]
                output[key] = value_index_of_interest
            else:
                output[key] = value[0]

        yield index, total, output

    for i in range(max_values):
        index += 1
        output = {}
        for key_index, key in enumerate(keys):
            value = values[key_index]
            output[key] = value[i % len(value)]
        yield index, total, output

    for i in range(count):
        index += 1
        yield index, total, permutations_dicts[i]


def getTestSuite():
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpEigen)
    # suite2 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpEigenLeaveOutOneChrom)
    return unittest.TestSuite([suite1])  # cmk,suite2])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalInParts

    suites = unittest.TestSuite([getTestSuite()])

    r = unittest.TextTestRunner(failfast=False)
    ret = r.run(suites)
    assert ret.wasSuccessful()

    logging.info("done with testing")
