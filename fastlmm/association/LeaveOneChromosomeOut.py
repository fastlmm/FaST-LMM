#!/usr/bin/env python2.7
#
# Written (W) 2014 Christian Widmer
# Copyright (C) 2014 Microsoft Research

"""
Created on 2014-03-11
@author: Christian Widmer
@summary: Module for performing GWAS
"""

from __future__ import absolute_import
import logging

import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
import pylab

import time

import fastlmm.inference as fastlmm

import fastlmm.util.util as util 
from fastlmm.util.pickle_io import load, save
from fastlmm.util.util import argintersect_left


class LeaveOneChromosomeOut(object):
    """LeaveOneChromosomeOut cross validation iterator (based on sklearn).

    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds according to which chromosome they belong to.

    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.

    Parameters
    ----------
    chr : list
        List of chromosome identifiers

    indices : boolean, optional (default True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    random_state : int or RandomState
            Pseudo number generator state used for random sampling.

    """

    def __init__(self, chr_names, indices=True, random_state=None):

        #random_state = check_random_state(random_state)

        self.chr_names = np.array(chr_names)
        self.unique_chr_names = list(set(chr_names))
        self.unique_chr_names.sort()

        assert len(self.unique_chr_names) > 1
        self.n = len(self.chr_names)
        self.n_folds = len(self.unique_chr_names)
        self.indices = indices
        self.idxs = np.arange(self.n)


    def __iter__(self):
        if self.indices:
            ind = np.arange(self.n)
        
        for chr_name in self.unique_chr_names:
            
            test_index = self.chr_names == chr_name
            train_index = np.logical_not(test_index)
            
            if self.indices:
                train_index = ind[train_index]
                test_index = ind[test_index]
            
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i, n_folds=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_folds,
        )

    def __len__(self):
        return self.n_folds
