#!/usr/bin/env python2.7
#
# Copyright (C) 2014 Microsoft Research

"""
Created on 2014-04-02
@summary: Helper Module for precomputing principal components for Leave one Chromosme out GWAS
"""

from __future__ import absolute_import
from __future__ import print_function
import logging
import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
import pylab
import fastlmm.pyplink.plink as plink
import pysnptools.util.pheno as pstpheno
import pysnptools.util as pstutil
import fastlmm.util.util as util 
import fastlmm.util.standardizer as stdizer
from fastlmm.util.pickle_io import load, save
import os.path
from sklearn.decomposition import PCA

import fastlmm.association.LeaveOneChromosomeOut as LeaveOneChromosomeOut
import fastlmm.pyplink.snpset.AllSnps as AllSnps
from six.moves import range

def load_intersect(snp_reader, pheno_fn_or_none,snp_set=AllSnps()):
    """
    load SNPs and phenotype, intersect ids
    ----------------------------------------------------------------------
    Input:
    bed_reader : SnpReader object (e.g. BedReader)
    pheno_fn   : str, file name of phenotype file, defa
    ----------------------------------------------------------------------
    Output:
    G : numpy array containing SNP data
    y : numpy (1d) containing phenotype
    ----------------------------------------------------------------------
    """

    standardizer = stdizer.Unit()

    geno = snp_reader.read(order='C',snp_set=snp_set)
    G = geno['snps']
    G = standardizer.standardize(G)

    snp_names = geno['rs']
    chr_ids = geno['pos'][:,0]

    if not pheno_fn_or_none is None:

        # load phenotype
        pheno = pstpheno.loadOnePhen(pheno_fn_or_none, 0)
        y = pheno['vals'][:,0]

        # load covariates and intersect ids
        import warnings
        warnings.warn("This intersect_ids is deprecated. Pysnptools includes newer versions of intersect_ids", DeprecationWarning)
        indarr = util.intersect_ids([pheno['iid'], snp_reader.original_iids])
    
        #print "warning: random phen"
        #y = np.random.random_sample(len(y)) 


        if not (indarr[:,0] == indarr[:,1]).all():
            assert False, "ERROR: this code assumes the same order for snp and phen file"

            print("reindexing")
            y = y[indarr[:,0]]
            G = G[indarr[:,1]]
    else:
        y = None


    return G, y, snp_names, chr_ids



class PrecomputeLocoPcs(object) : #implements IDistributable
    '''
    Find the PCs of a snp data set and phenotype
    '''
    def __init__(self, chrom_count, snp_reader, pheno_fn, cache_prefix):
        self.snp_reader = snp_reader
        self.pheno_fn = pheno_fn
        self.cache_prefix = cache_prefix
        self.chrom_count = chrom_count

 #start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        return self.chrom_count

    def work_sequence(self):

        G, y, snp_name,chr_ids = load_intersect(self.snp_reader, self.pheno_fn)
        loco = LeaveOneChromosomeOut.LeaveOneChromosomeOut(chr_ids, indices=True)
        if len(loco) is not self.chrom_count :  raise Exception("The snp reader has {0} chromosome, not {1} as specified".format(len(loco),self.chrom_count))

    
        for i, (train_snp_idx, _) in enumerate(loco):
            yield lambda i=i, train_snp_idx=train_snp_idx,G=G: self.dowork(i,train_snp_idx,G)  # the 'i=i',etc is need to get around a strangeness in Python

    def reduce(self, result_sequence):
        '''
        '''
        for i, pcs in result_sequence:
            out_fn = self.create_out_fn(self.cache_prefix, i)
            pstutil.create_directory_if_necessary(out_fn)
            save(out_fn, pcs)
        return None

    #optional override -- the str name of the instance is used by the cluster as the job name
    def __str__(self):
        return "{0}({1},'{2}','{3}')".format(self.__class__.__name__, self.snp_reader, self.pheno_fn, self.cache_prefix)
 #end of IDistributable interface---------------------------------------

    def is_run_needed(self):
        # don't recompute if all files exist
        for i in range(self.chrom_count):
            pc_fn = self.create_out_fn(self.cache_prefix, i)
            if not os.path.isfile(pc_fn):
                return True
        return False

    @staticmethod
    def create_out_fn(cache_prefix, i):
        #TODO: throw exception if it's top level
        out_fn = "%s_%04d.pickle.bzip" % (cache_prefix, i)
        return out_fn

    def dowork(self, i, train_snp_idx, G):
        '''
        This can return anything, but note that it will be binary serialized (pickleable), and you don't want to have more than is required there for reduce
        '''
        # fast indexing (needs to be C-order)
        assert np.isfortran(G) == False
        G_train = G.take(train_snp_idx, axis=1)
        
        pca = PCA()
        pcs = pca.fit_transform(G_train)
        # n_ind, n_pcs

        return i, pcs


    #!! would be nice of this was optional and if not given the OS was asked
    # required by IDistributable
    @property
    def tempdirectory(self):
        return ".work_directory." + self.cache_prefix


    def copyinputs(self, copier):
        copier.input(self.pheno_fn)
        copier.input(self.snp_reader)

    #Note that the files created are not automatically copied. Instead,
    # whenever we want another file to be created, a second change must be made here so that it will be copied.
    def copyoutputs(self,copier):
        for i in range(self.chrom_count):
            out_fn = self.create_out_fn(self.cache_prefix, i)
            copier.output(out_fn)
