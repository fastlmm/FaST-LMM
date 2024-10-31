

class Beta(object):  # IStandardizer
    """The specification for beta standardization"""

    def __init__(self, a=1, b=25):

        #!!warnings.warn("This Beta is deprecated. Pysnptools includes newer versions of Beta", DeprecationWarning)
        self.a = a
        self.b = b

    def standardize(
        self, snps, blocksize=None, force_python_only=False, num_threads=None
    ):
        thing = self.lambdaFactory(
            snps,
            blocksize=blocksize,
            force_python_only=force_python_only,
            num_threads=num_threads,
        )
        import fastlmm.util.standardizer as stdizer

        return stdizer.standardize_with_lambda(snps, thing, blocksize)

    @staticmethod
    def _standardizer(snps, a, b, force_python_only, num_threads):
        from pysnptools.standardizer import Standardizer

        Standardizer._standardize_unit_and_beta(
            snps,
            is_beta=True,
            a=a,
            b=b,
            apply_in_place=True,
            use_stats=False,
            stats=None,
            force_python_only=force_python_only,
            num_threads=num_threads,
        )
        return snps

    def lambdaFactory(
        self, snps, blocksize=None, force_python_only=False, num_threads=None
    ):

        return lambda s, a=self.a, b=self.b, force_python_only=force_python_only: self._standardizer(
            snps, a, b, force_python_only, num_threads
        )
