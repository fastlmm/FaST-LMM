import doctest
import sys

import unittest
import os.path
from unittest.mock import patch
import builtins


class TestOptionalHdf5(unittest.TestCase):
    def test_missing_backend(self):
        from fastlmm.pyplink.snpreader import Hdf5

        reader = Hdf5("unused.hdf5")
        with patch.dict(sys.modules, {"h5py": None}):
            for _ in range(2):
                with self.assertRaisesRegex(ModuleNotFoundError, r"fastlmm\[hdf5\]") as caught:
                    reader.run_once()
                self.assertEqual(caught.exception.name, "h5py")
                self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)

    def test_broken_backend(self):
        from fastlmm.pyplink.snpreader import Hdf5

        original_import = builtins.__import__
        error = ImportError("incompatible h5py binary")

        def broken_import(name, *args, **kwargs):
            if name == "h5py":
                raise error
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", broken_import):
            with self.assertRaises(ImportError) as caught:
                Hdf5("unused.hdf5").run_once()
            self.assertIs(caught.exception, error)


# We do it this way instead of using doctest.DocTestSuite because doctest.DocTestSuite requires modules to be pickled, which python doesn't allow.
# We need tests to be pickleable so that they can be run on a cluster.
class TestDocStrings(unittest.TestCase):

    def test_bed(self):

        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        result = doctest.testmod(sys.modules["fastlmm.pyplink.snpreader.Bed"])
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


def getTestSuite():
    """
    set up composite test suite
    """

    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDocStrings))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOptionalHdf5))

    return test_suite


if __name__ == "__main__":

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=False)
    ret = r.run(suites)
    assert ret.wasSuccessful()
    print("done")
