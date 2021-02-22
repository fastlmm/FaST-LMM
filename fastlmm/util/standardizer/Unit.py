from __future__ import absolute_import
import numpy as np
import scipy as sp
import logging

class Unit(object):  #IStandardizer
    """The specification for unit standardization"""
    def __init__(self):
        import warnings
        #!!warnings.warn("This Unit is deprecated. Pysnptools includes newer versions of Unit", DeprecationWarning)
        pass

    def standardize(self, snps, blocksize=None, force_python_only=False,num_threads=None):
        l = self.lambdaFactory(snps, blocksize=blocksize, force_python_only=force_python_only,num_threads=num_threads)
        import fastlmm.util.standardizer as stdizer
        return stdizer.standardize_with_lambda(snps, l, blocksize)

    @staticmethod
    def _standardizer(snps,force_python_only,num_threads):
        from pysnptools.standardizer import Standardizer
        Standardizer._standardize_unit_and_beta(snps, is_beta=False, a=float("NaN"), b=float("NaN"), apply_in_place=True, use_stats=False,stats=None,force_python_only=force_python_only,num_threads=num_threads)
        return snps

    def lambdaFactory(self, snps, blocksize=None, force_python_only=False,num_threads=None):
        return lambda s,force_python_only=force_python_only:self._standardizer(snps,force_python_only,num_threads)

    def __str__(self):
        return "{0}()".format(self.__class__.__name__)
