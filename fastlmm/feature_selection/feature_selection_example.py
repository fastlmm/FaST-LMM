"""
example of how to use feature selection from python (see also command line interface)
"""

from __future__ import absolute_import
import numpy as np
from fastlmm.feature_selection import FeatureSelectionStrategy
from pysnptools.snpreader import Bed

import logging

def runselect(bed_fn=None, pheno_fn=None, strategy=None, select_by_ll=True, output_prefix=None,num_pcs=0, random_state=3, num_snps_in_memory=1000, cov_fn=None, k_values=None, delta_values=None,num_folds=10,penalty=0.0):
    logging.basicConfig(level=logging.INFO)

    # set up data
    ##############################
    if bed_fn is None:
        bed_fn = Bed("examples/toydata")
        pheno_fn = "examples/toydata.phe"
    

    # set up grid
    ##############################
    num_steps_delta = 10
    num_steps_k = 5

    # log_2 space and all SNPs
    #k_values = np.logspace(0, 9, base=2, num=num_steps_k, endpoint=True).tolist() + [10000]
    if k_values is None:
        k_values = [0, 1, 5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 456345643256] #100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 400, 500, 1000]
    if delta_values is None:
        delta_values = np.logspace(-10, 10, endpoint=True, num=num_steps_delta, base=np.exp(1))
    #delta_values = [np.exp(1), np.exp(2), np.exp(3), np.exp(4), np.exp(5), np.exp(6)]

    if strategy is None:
        strategy = 'lmm_full_cv'
        select_by_ll = True
    if 0:
        strategy = 'insample_cv'
        select_by_ll = True
    # where to save output
    ##############################
    if output_prefix is None:
        output_prefix = "example_pc%i" % (num_pcs)
    
    # go!
    fss = FeatureSelectionStrategy(bed_fn, pheno_fn, num_folds, random_state=random_state, num_pcs=num_pcs, num_snps_in_memory=num_snps_in_memory, interpolate_delta=False, cov_fn=cov_fn)

    best_k, best_delta, best_obj, best_snps = fss.perform_selection(k_values, delta_values, output_prefix=output_prefix, select_by_ll=select_by_ll, strategy = strategy, penalty=penalty)
    res = {
           'best_k':best_k,
           'best_delta':best_delta,
           'best_obj':best_obj, 
           'best_snps':best_snps
           }
    return res
    

if __name__ == "__main__":
    result = runselect()

