import os
import logging
import unittest
from pysnptools.util.filecache import Hashdown
import doctest


fastlmm_hashdown = Hashdown.load_hashdown(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "fastlmm.hashdown.json"
    ),
    directory=os.environ.get("FASTKNN_CACHE_HOME", None),
)

def example_file(pattern, endswith=None):
    """
    Returns the local location of a FaST-LMM example file, downloading it
    if needed.

    :param pattern: The name of the example file of interest. A 
        `file name pattern <https://docs.python.org/3.7/library/fnmatch.html>`__
        may be given. All matching files will be downloaded (if needed) and
        the name of one will be returned.
    :type file_name: string

    :param endswith: The pattern of the file name to return. By default, when
        no `endswith` is given, the name of the first matching file will be
        returned.
    :type file_name: string

    :rtype: string

    By default, the local location will be under the system temp directory
    (typically controlled with the TEMP environment variable).
    Alternatively, the directory can be set with the FASTLMM_CACHE_HOME
    environment variable.

    This function knows the MD5 hash of all FaST-LMM example files and uses
    that content-based hash to decide if a file needs to be downloaded.

    >>> from fastlmm.util import example_file # Download and return local file name # Download and return local file name
    >>> # Download the phenotype file if necessary. Return its local location.
    >>> pheno_fn = example_file("fastlmm/feature_selection/examples/toydata.phe")
    >>> print('The local file name is ', pheno_fn)
    The local file name is ...fastlmm/feature_selection/examples/toydata.phe
    >>>
    >>> # Download the bed,bim,&fam files if necessary. Return the location of bed file.
    >>> bedfile = example_file("fastlmm/feature_selection/examples/toydata.5chrom.*","*.bed")
    >>> print('The local file name is ', bedfile)
    The local file name is ...fastlmm/feature_selection/examples/toydata.5chrom.bed

    """
    return fastlmm_hashdown._example_file(pattern, endswith=endswith)


class TestExampleFile(unittest.TestCase):
    def test_doc_test(self):
        import fastlmm.util._example_file

        result = doctest.testmod(
            fastlmm.util._example_file, optionflags=doctest.ELLIPSIS
        )
        assert result.failed == 0, "failed doc test: " + __file__


def getTestSuite():
    """
    set up test suite
    """

    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExampleFile))

    return test_suite


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=False)
    ret = r.run(suites)
    assert ret.wasSuccessful()

    doctest.testmod(optionflags=doctest.ELLIPSIS)
