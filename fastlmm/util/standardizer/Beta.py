from __future__ import absolute_import
import numpy as np
import scipy as sp
import logging

class Beta(object): #IStandardizer
    """The specificiation for beta standardization"""
    def __init__(self,a=1,b=25):
        import warnings
        #!!warnings.warn("This Beta is deprecated. Pysnptools includes newer versions of Beta", DeprecationWarning)
        self.a = a
        self.b = b

    def standardize(self, snps, blocksize=None, force_python_only=False):
        l = self.lambdaFactory(snps, blocksize=blocksize, force_python_only=force_python_only)
        import fastlmm.util.standardizer as stdizer
        return stdizer.standardize_with_lambda(snps, l, blocksize)

    @staticmethod
    def _standardizer(snps,a,b,force_python_only):
        from pysnptools.standardizer import Standardizer
        Standardizer._standardize_unit_and_beta(snps, is_beta=True, a=a, b=b, apply_in_place=True, use_stats=False,stats=None,force_python_only=force_python_only)
        return snps


    def lambdaFactory(self, snps, blocksize=None, force_python_only=False):
        from pysnptools.standardizer import Standardizer
        return lambda s,a=self.a,b=self.b,force_python_only=force_python_only:self._standardizer(snps,a,b,force_python_only)


