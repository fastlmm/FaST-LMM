import logging
import unittest
import os.path
import numpy as np
import datetime
from pathlib import Path

from pysnptools.snpreader import Bed, Pheno, SnpData
from pysnptools.util.mapreduce1.runner import LocalMultiProc, Local
from pysnptools.util.mapreduce1 import map_reduce
from pysnptools.standardizer import Unit

from fastlmm.inference.fastlmm_predictor import _kernel_fixup
from fastlmm.util import example_file  # Download and return local file name
from fastlmm.association.single_snp_eigen import _append_bias


class TestSingleSnpSimple(unittest.TestCase):
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

    tempout_dir = "tempout/single_snp_simple"

    def file_name(self, testcase_name):
        temp_fn = os.path.join(self.tempout_dir, testcase_name + ".txt")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_same_as_old_code(self):  # !!!cmk too slow???
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
            cache_file0 = Path(
                r"m:/deldir/eigentest/same_as_old"
            ) / f"{date_str}.{i}".replace(":", "-")
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
            ".0001": 0.0001,
            "100_000": 100_000,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
        }

        matrix = {
            "use_reml": ["False", "True"],
            "train_count": ["50", "750"],
            "cov": ["None", "cov_reader"],
            "delta": ["None", "1.0", "0.20000600000000002"],
            "pheno": ["pheno_fn"],
            "snps_reader": ["snps_reader1", "snps_reader5"],
        }

        if True:
            test_runner = None  # LocalMultiProc(6, just_one_process=True)
            runner = None  # LocalMultiProc(6, just_one_process=True)
            exception_to_catch = TimeoutError  # Exception #
            extra_lambda = lambda case_number: case_number ** 0.5
        else:
            test_runner = LocalMultiProc(6, just_one_process=False)
            runner = Local()
            exception_to_catch = Exception
            extra_lambda = lambda case_number: case_number ** 0.5
        first_list = [
            {
                "use_reml": "True",
                "train_count": "750",
                "cov": "None",
                "delta": "None",
                "pheno": "pheno_fn",
                "snps_reader": "snps_reader5",
            }
        ]

        def mapper2(index_total_option):
            import numpy as np
            from pysnptools.snpreader import Pheno, SnpData

            index, total, option = index_total_option
            print(f"============{index} of {total}==================")
            print(f"==={option}")

            use_reml = value_dict[option["use_reml"]]
            cov = value_dict[option["cov"]]
            train_count = value_dict[option["train_count"]]
            delta = value_dict[option["delta"]]
            pheno = value_dict[option["pheno"]]
            snps_reader = value_dict[option["snps_reader"]]

            pheno = Pheno(pheno)

            if cov is None:
                cov1 = SnpData(
                    iid=pheno.iid, sid=["bias"], val=np.full((pheno.iid_count, 1), 1.0)
                )
            else:
                cov1 = _append_bias(cov)

            try:
                from pysnptools.kernelstandardizer import Identity as KernelIdentity
                from pysnptools.util.mapreduce1 import map_reduce
                from fastlmm.association._single_snp_simple import (
                    single_snp_simple,
                    Eigen,
                )
                from fastlmm.association.single_snp_eigen import eigen_from_kernel

                # !!!cmk why not diag standardize?
                K_eigen = eigen_from_kernel(
                    snps_reader[:, :train_count], kernel_standardizer=KernelIdentity(),
                )
                test_snps = snps_reader[:, train_count : train_count + test_count]

                frame = single_snp_simple(
                    test_snps=test_snps.read().standardize().val,
                    test_snps_ids=test_snps.sid,
                    pheno=pheno.read().val,
                    K_eigen=Eigen(K_eigen.values, K_eigen.vectors),
                    covar=cov1.read().val,
                    log_delta=np.log(delta) if delta is not None else None,
                    _find_delta_via_reml=use_reml,
                    _test_via_reml=use_reml,
                )

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
                    from fastlmm.association.tests.test_gwas import GwasPrototype

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
                    range(phenox.sid_count),
                    mapper=mapper,
                    runner=Local() if runner is None else runner,
                )
                np.testing.assert_array_almost_equal(
                    np.log(gwas_pvalues_list[0]),
                    np.log(sorted(frame.PValue.values)),
                    decimal=6,
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


# !!!cmk where should this live?
def to_kernel(K0, kernel_standardizer, count_A1=None):
    """!!!cmk documentation"""
    # !!!cmk could offer a low-memory path that uses memmapped files

    assert K0 is not None
    K0 = _kernel_fixup(K0, iid_if_none=None, standardizer=Unit(), count_A1=count_A1)
    assert K0.iid0 is K0.iid1, "Expect K0 to be square"

    # !!!cmk understand _read_kernel, _read_with_standardizing
    # !!!cmk test
    K0 = K0._read_with_standardizing(
        kernel_standardizer=kernel_standardizer,
        to_kerneldata=True,
        return_trained=False,
    )
    return K0


def getTestSuite():
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestSingleSnpSimple)
    return unittest.TestSuite([suite1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc, LocalInParts

    suites = unittest.TestSuite([getTestSuite()])

    r = unittest.TextTestRunner(failfast=False)
    ret = r.run(suites)
    assert ret.wasSuccessful()

    logging.info("done with testing")
