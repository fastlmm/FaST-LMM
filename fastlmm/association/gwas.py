# Copyright 2014 Microsoft Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# original file at https://github.com/fastlmm/FaST-LMM

# modified 2015 by Christoph Lippert

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import scipy.stats as st
import scipy.special as ss
import pandas as pd
from fastlmm.util.mingrid import *
from fastlmm.util import *
import time, os
#import ipdb
from pysnptools.snpreader import Bed

import logging
import pysnptools.standardizer
import pysnptools
import pysnptools.util
import pysnptools.util.pheno
import time
import fastlmm.inference.lmm_cov as lmm_cov
import glob
from six.moves import range

def create_dir(filename, is_dir=True):
    if is_dir:
        dirname = filename
    else:
        dirname = os.path.dirname(filename)
    if (not os.path.exists(dirname)) or (not os.path.isdir(dirname)):
        os.makedirs(dirname)

class GWAS(object):
    def __init__(self, snps_test, phenotype, K=None, snps_K=None, covariates=None, 
                 h2=None, interact_with_snp=None, nGridH2=10, standardizer=None, add_bias=True, normalize_K=True, blocksize=10000):
        if standardizer is None:
            standardizer = pysnptools.standardizer.Unit()
        self.standardizer = standardizer
        self.phenotype = GWAS.check_pheno_format(phenotype=phenotype)
        self.covariates = GWAS.check_pheno_format(phenotype=covariates)
        self.snps_test = snps_test
        self.snps_K = snps_K
        self.K = K
        self.linreg = False
        if (self.K is None) and (self.snps_K is None):
            G = np.zeros((self.sample_count,0))
            self.linreg = True
        elif self.K is None:
            G = None
            self.K = snps_K.kernel(standardizer=standardizer, blocksize=blocksize)
            if normalize_K:
                self.K /= self.K.diagonal().mean()
        elif snps_K is None:
            G = None
            if normalize_K:
                self.K = self.K / self.K.diagonal().mean()
        else:
            raise NotImplementedError("either K or snps_K has to be None")
        if add_bias and (self.covariates is None):
            pass # this case is treated in LMM()
        elif add_bias:
            bias = pd.Series(np.ones((self.covariates.shape[0])), index=self.covariates.index)
            self.covariates['bias'] = bias
        elif (add_bias==False) and self.covariates is None:
            raise NotImplementedError("currently a model with neither bias term, nor covariates is not supported.")
        self.interact_with_snp = interact_with_snp
        self.nGridH2 = nGridH2
        if self.covariates is not None:
            self.lmm = lmm_cov.LMM(X=self.covariates.values, Y=None, G=G, K=self.K, inplace=True)
        else:
            self.lmm = lmm_cov.LMM(X=None, Y=None, G=G, K=self.K, inplace=True)
        self._find_h2()
    

    def compute_association(self, blocksize=10000, temp_dir=None):
        result = {}

        if temp_dir is not None:
            if (not os.path.exists(temp_dir)) or (not os.path.isdir(temp_dir)):
                os.makedirs(temp_dir)
        self._generate_intervals(blocksize=blocksize, total_snps=self.snps_test.sid.shape[0])
        for stop in range(1,len(self.intervals)):
            res = self.snps_test_block(block_start=self.intervals[stop-1],block_end=self.intervals[stop])
            if temp_dir is not None:
                self.save_results_block(dir_name=temp_dir, res=res, intervals=self.intervals, idx_interval=stop)
            else:
                for p in res.keys():
                    if p in result:
                        result[p] = pd.concat([result[p],res[p]])
                    else:
                        result[p] = res[p]
        return result

    def save_results_block(self, dir_name, res, intervals=None, idx_interval=None):
        for p in res.keys():
            mydir = dir_name + "/" + str(p) +"/"
            create_dir(mydir)
            if intervals is not None:
                fill = int(np.floor(np.log10(len(intervals))))+1
            else:
                fill = 0
            if idx_interval is None:
                stop = 1
            else:
                stop = idx_interval
            #block_format = '%0*d' % (fill, stop)
            myfile = "%s%s_block_%0*d.csv" % (mydir, p, fill, stop)
            res[p].to_csv(myfile)

    def snps_test_block(self, block_start, block_end):
        snps = self.snps_test[:,block_start:block_end].read().standardize(self.standardizer)
        result = {}
        for i, p in enumerate(self.phenotype.columns):
            h2 = self.h2[p]
            self.lmm.setY(self.phenotype[p][:,np.newaxis])
            res = self.lmm.nLLeval(h2=h2, dof=None, scale=1.0, penalty=0.0, snps=snps.val)
            result[p] = self._format_gwas_results(res=res, snps=snps, h2=h2)
        return result

    @property
    def sample_count(self):
        return self.phenotype.shape[0]
    
    def _find_h2(self):
        self.h2 = {}
        if self.linreg:
            for i, p in enumerate(self.phenotype.columns):
                self.h2[p] = 0.0
        else:
            for i, p in enumerate(self.phenotype.columns):
                self.lmm.setY(self.phenotype[p][:,np.newaxis])
                self.h2[p] = self.lmm.findH2(nGridH2=self.nGridH2)['h2']

    def _generate_intervals(self, blocksize, total_snps):
        if blocksize is None:
            blocksize = total_snps
        intervals = list(range(0,total_snps,blocksize))
        
        if intervals[-1]!= total_snps:
            intervals.append(total_snps)
        self.intervals = intervals

    def _format_gwas_results(self, res, snps, h2):
        beta = res['beta']
            
        chi2stats = beta*beta/res['variance_beta']

        #p_values = st.chi2.sf(chi2stats,1)[:,0]
        p_values = st.f.sf(chi2stats,1,self.lmm.U.shape[0]-(self.covariates.shape[0]+1))[:,0]#note that G.shape is the number of individuals

        items = [
            ('SNP', snps.sid),
            ('Chr', snps.pos[:,0]), 
            ('GenDist', snps.pos[:,1]),
            ('ChrPos', snps.pos[:,2]), 
            ('PValue', p_values),
            ('SnpWeight', beta[:,0]),
            ('SnpWeightSE', np.sqrt(res['variance_beta'][:,0])),
            ('SnpFractVarExpl', np.sqrt(res['fraction_variance_explained_beta'][:,0])),
            ('Nullh2', np.zeros((snps.sid_count)) + h2)
        ]
        return pd.DataFrame.from_items(items)

    @staticmethod
    def check_pheno_format(phenotype):
        if type(phenotype) == np.ndarray:
            if len(phenotype.shape)==1:
                phenotype = phenotype[:,np.newaxis]

            return pd.DataFrame(data=phenotype)
        elif type(phenotype) == pd.DataFrame:
            return phenotype
        elif type(phenotype) == dict:
            assert "iid" in phenotype, "assuming format to be the fastlmm format."
            assert "header" in phenotype, "assuming format to be the fastlmm format."
            assert "vals" in phenotype, "assuming format to be the fastlmm format."
            pheno_new = pd.DataFrame(data=phenotype["vals"], columns=phenotype["header"], index=phenotype["iid"])
            return pheno_new
        elif phenotype is None:
            return None
        else:
            raise NotImplementedError("phenotype is assumed to be a numpy.ndarray or pandas.DataFrame, found %s instead" % (str(type(phenotype))))

    @staticmethod
    def _snp_fixup(snp_input, iid_source_if_none=None,count_A1=None):
        if isinstance(snp_input, str):
            return Bed(snp_input,count_A1=count_A1)
        elif snp_input is None:
            return iid_source_if_none[:,0:0] #return snpreader with no snps
        else:
            return snp_input

    @staticmethod
    def _pheno_fixup(pheno_input, iid_source_if_none=None):
        if isinstance(pheno_input, str):
            return pysnptools.util.pheno.loadPhen(pheno_input) #!!what about missing=-9?

        if pheno_input is None:
            ret = {
            'header':[],
            'vals': np.empty((iid_source_if_none['vals'].shape[0], 0)),
            'iid':iid_source_if_none['iid']
            }
            return ret

        if len(pheno_input['vals'].shape) == 1:
            ret = {
            'header' : pheno_input['header'],
            'vals' : np.reshape(pheno_input['vals'],(-1,1)),
            'iid' : pheno_input['iid']
            }
            return ret

        return pheno_input

    @staticmethod
    def load_results(dir_name,pheno_name):
        result = None
        mydir = dir_name + "/" + pheno_name + "/"
        myfile_pattern = "%s%s_block_*.csv" % (mydir, pheno_name)
        files = glob.glob(myfile_pattern)
        for filename in files:
            res = pd.read_csv(filename)
            if result is None:
                result = res
            else:
                result = pd.concat((result,res),0)
        return result

if __name__ == "__main__":

    bed_fn = "../../test/data/plinkdata/toydata"
    pheno_fn = bed_fn + ".phe6"#"../../test/data/plinkdata/toydata.phe"
    covariate_fn = bed_fn + ".phe"

    
    blocksize = 20000
    snp_reader = Bed(bed_fn)#[:,0:50000]
    #snp_reader.run_once()
    if 1:
        standardizer = pysnptools.standardizer.Unit()
    else:
        standardizer = pysnptools.standardizer.Beta(1,2)
    pheno = pysnptools.util.pheno.loadPhen(filename=pheno_fn,   missing ='-9')
    pheno = GWAS._pheno_fixup(pheno)
    covariates = GWAS._pheno_fixup(covariate_fn, iid_source_if_none=pheno)
    print("intersecting data")
    t00 = time.time()
    snp_intersect, pheno_intersect = pysnptools.util.intersect_apply([snp_reader, pheno], sort_by_dataset=True)
    t1 = time.time()
    print("done intersecting after %.4fs" % (t1-t00))

    print("building kernel")
    t0 = time.time()
    if 0: 
        #snp_data = snp_intersect.read().standardize()
        snp_data = snp_intersect.read().standardize(standardizer)

        G = snp_data.val

        K = G.dot(G.T)
        K/=K.diagonal().mean()
    else:
        K = snp_intersect.kernel(standardizer=standardizer,blocksize=blocksize)
        K /= K.diagonal().mean()
    t1 = time.time()
    print("done building kernel after %.4fs" % (t1-t0))

    if 0:
        print("computing Eigenvalue decomposition of K")
        t0 = time.time()
        S,U = la.eigh(K)
        t1 = time.time()
        print("done computing eigenvalue decomposition of kernel after %.4fs" % (t1-t0))
        
    if 1:
        print("running GWAS")
        t0 = time.time()
        if 1:#LMM with pre-built kernel K
            mygwas = GWAS(K=K, snps_K=None, snps_test=snp_intersect, phenotype=pheno, covariates=covariates, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer)
        elif 0:#LMM building kernel inside from snps_K
            mygwas = GWAS(K=None, snps_K=snp_intersect, snps_test=snp_intersect, phenotype=pheno, covariates=covariates, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer)

        else:#linear regression
            mygwas = GWAS(K=None, snps_K=None, snps_test=snp_intersect, phenotype=pheno, covariates=covariates, h2=None, interact_with_snp=None, nGridH2=10, standardizer=standardizer)
        if 1:
            result = mygwas.compute_association(blocksize=blocksize, temp_dir=None)#'./temp_dir_testdata/')
        else:
            result_block = mygwas.snps_test_block(block_start=0, block_end=blocksize)
            mygwas.save_results_block("./res_check/", result_block)
        t1 = time.time()
        print("done running GWAS after %.4fs" % (t1-t0))
        print("total: %.4fs" % (t1-t00))

