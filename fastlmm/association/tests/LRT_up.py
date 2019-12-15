from __future__ import absolute_import
from __future__ import print_function
import fastlmm.inference.lmm_cov as lmm
import numpy as np
import fastlmm.util.stats.chi2mixture as c2
import fastlmm.association as association
import scipy.stats as st
from . import tests_util as tu
from six.moves import range

class lrt(association.varcomp_test):
    __slots__ = ["lmm","lrt","forcefullrank","nullModel","altModel","G0","K0","__testGcalled","model0","model1"]

    def __init__(self, Y, X=None, appendbias=False, forcefullrank=False, G0=None, K0=None, nullModel=None,altModel=None):
        association.varcomp_test.__init__(self,Y=Y,X=X,appendbias=appendbias)
        N = self.Y.shape[0]
        self.forcefullrank=forcefullrank
        self.nullModel = nullModel
        self.altModel = altModel
        self.G0=G0
        self.K0=K0
        self.__testGcalled=False
        self.lmm = lmm.LMM(forcefullrank=self.forcefullrank, X=self.X, linreg=None, Y=self.Y[:,np.newaxis], G=self.G0, K=self.K0, regressX=True)
        self.model0 = self.lmm.findH2()# The null model only has a single kernel and only needs to find h2
        self.model1=None

    @property
    def _testGcalled(self):
        return self.__testGcalled

    def testG(self, G1, type=None,i_exclude=None,G_exclude=None):    
        self.__testGcalled=True
        #compute the alternative likelihood
        (lik1,stat,alteqnull) = self._altModelMixedEffectLinear(G1,i_exclude=i_exclude,G_exclude=G_exclude)
                   
        #due to optimization the alternative log-likelihood might be a about 1E-6 worse than the null log-likelihood 
        pvreg = (st.chi2.sf(stat,1.0)) #standard way to compute p-value when no boundary conditions
        if np.isnan(pvreg) or pvreg>1.0:
            pvreg=1.0                
        pv = 0.5*pvreg                  #conservative 50/50 estimate
        if alteqnull: pv=1.0            #chi_0 component
                    
        test={
              'pv':pv,
              'stat':stat,
              'lik1':lik1,
              'lik0':self.model0,
              'alteqnull':alteqnull
              }
        return test

    def _altModelMixedEffectLinear(self, G1,tol=0.0,i_exclude=None,G_exclude=None):
        lik0=self.model0

        G, i_G1, n_exclude = tu.set_Gexclude(G_exclude, G1, i_exclude)
        UGup,UUGup = self.lmm.rotate(G)
        i_up=~i_G1
        #update null model if SNPs are excluded:
        if n_exclude:
            if UUGup is not None:
                UUGup_=UUGup[:,0:n_exclude]
            else:
                UUGup_=None
            lik0 = self.lmm.findH2_2K(nGridH2=100, minH2 = 0.0, maxH2 = 0.99999, i_up=i_up[0:n_exclude], i_G1=i_G1[0:n_exclude], UW=UGup[:,0:n_exclude], UUW=UUGup_)#The alternative model has two kernels and needs to find both a2 and h2

        #build indicator for test SNPs (i_G1) and excluded SNPs (i_up)
        #we currently don't account for exclusion of snps in G1 (low rank update could be even more low rank)
        
        #alternative model likelihood:
        lik1 = self.lmm.findH2_2K(nGridH2=100, minH2 = 0.0, maxH2 = 0.99999, i_up=i_up, i_G1=i_G1, UW=UGup, UUW=UUGup)#The alternative model has two kernels and needs to find both a2 and h2
        try:
            alteqnull=lik1['h2_1'][0]<=(0.0+tol)
        except:
            alteqnull=lik1['h2_1']<=(0.0+tol)

        stat = 2.0*(lik0['nLL'][0] - lik1['nLL'][0])
    
        self.model1=lik1

        return (lik1,stat,alteqnull)



class LRT_up(object):
    __slots__ = ["model0","model1","lrt","forcefullrank","nullModel","altModel","G0","__testGcalled"]
    """description of class"""
    def check_nperm(self,nperm):
        return nperm #permutations are fine, so just return

    def __str__(self):
        return "lrt_up"


    def construct(self, Y, X=None, forcefullrank = False, SNPs0 = None, i_exclude=None, nullModel = None, altModel = None,
                  scoring = None, greater_is_better = None):
        G0,K0=tu.set_snps0(SNPs0=SNPs0,sample_size=Y.shape[0],i_exclude=i_exclude)
        print("constructing LMM - this should only happen once.")
        return lrt(Y, X=X, forcefullrank=forcefullrank, G0=G0, K0=K0, nullModel=nullModel,altModel=altModel)


    def pv(squaredform,expectationsqform,varsqform,GPG):
        raise Exception("'pv' doesn't apply to lrt only to davies")

    @property
    def npvals(self):
        return 1 # return only 1 type of p-value

    def w2(self, G0, result):
        if G0 is not None:
            return result.h2_1
        else:
            raise NotImplementedError("only with backgr. K")

    def lrt_method(self, result):
        return result.stat

    def pv_adj_from_result(self, result):
        '''
        If local aUD exists, take that, if not, take the raw local.
        '''        
        if "pv-local-aUD" in result.test and not np.isnan(result.test["pv-local-aUD"]):
            return result.test["pv-local-aUD"]
        elif "pv-local" in result.test:
            return result.test["pv-local"]
        else:
            return np.nan

    def pv_adj_and_ind(self, nperm, pv_adj, nullfit, lrt, lrtperm,
                       alteqnull, alteqnullperm, qmax, nullfitfile, nlocalperm):        
        if nlocalperm and nlocalperm>0: #don't do the fitting
            ind = pv_adj.argsort()
            return pv_adj, ind

        from fastlmm.association.tests import Cv                
        return Cv.pv_adj_and_ind(nperm, pv_adj, nullfit, lrt, lrtperm,
                                 alteqnull, alteqnullperm, qmax, nullfitfile, nlocalperm) # call the shared version of this method

    def write(self, fp,ind, result_dict, pv_adj, detailed_table, signal_ratio=True):
        
        if "pv-local-aUD" in result_dict[0].test:
            # in this case, for p_adj, we use pv-local-aUD if it exists, and otherwise
            # pv-local. So don't know which is which in the "P-value adjusted" column. To
            # disambiguate, also print out "pv-local" here
            colnames = ["SetId", "LogLikeAlt", "LogLikeNull", "P-value_adjusted","P-value-local",
                        "P-value(50/50)", "#SNPs_in_Set", "#ExcludedSNPs", "chrm", "pos. range"]
        else:
            colnames = ["SetId", "LogLikeAlt", "LogLikeNull", "P-value_adjusted",
                        "P-value(50/50)", "#SNPs_in_Set", "#ExcludedSNPs", "chrm", "pos. range"]
        if signal_ratio:
            colnames.append("Alt_h2")
            colnames.append("Alt_h2_1")
        
        head = "\t".join(colnames)

        if detailed_table:
            lik1Info = result_dict[0].lik1Details
            lik0Info = result_dict[0].lik0Details

            altNames = list(lik1Info.keys())
            altIndices = sorted(list(range(len(altNames))), key=lambda k: altNames[k])
            altNames.sort()

            altNames = ['Alt'+t for t in altNames]
            head += "\t" + "\t".join( altNames )

            nullNames = list(lik0Info.keys())
            nullIndices = sorted(list(range(len(nullNames))), key=lambda k: nullNames[k])
            nullNames.sort()

            nullNames = ['Null'+t for t in nullNames]
            head += "\t" + "\t".join( nullNames )

        head += "\n"

        fp.write(head)
   
        for i in range(len(ind)):
            ii = ind[i]
            result = result_dict[ii]
            ll0=str( -(result.stat/2.0+result.test['lik1']['nLL'][0]) )

            if "pv-local-aUD" in result_dict[0].test:
                rowvals = [result.setname, str(-result.test['lik1']['nLL'][0]), ll0,
                           str(pv_adj[ii]),str(result.test['pv-local']),str(result.pv), str(result.setsize),
                           str(result.nexclude), result.ichrm, result.iposrange]
            else:
                rowvals = [result.setname, str(-result.test['lik1']['nLL'][0]), ll0,
                           str(pv_adj[ii]), str(result.pv), str(result.setsize),
                           str(result.nexclude), result.ichrm, result.iposrange]

            if signal_ratio:
                rowvals.append(str(result.h2))
                rowvals.append(str(result.h2_1))

            row = "\t".join(rowvals)

            if detailed_table:
                lik1Info = result.lik1Details
                lik0Info = result.lik0Details

                vals = list(lik1Info.values())
                vals = [vals[j] for j in altIndices]
                row += "\t" + "\t".join([str(v) for v in vals])

                vals = list(lik0Info.values())
                vals = [vals[j] for j in nullIndices]
                row += "\t" + "\t".join([str(v) for v in vals])

            row += "\n"
            fp.write(row)

    def pv_etc(self, filenull, G0_to_use, G1, y, x, null_model, varcomp_test, forcefullrank):
        if self.filenull is not None:
            return lr.twokerneltest(G0=G0_to_use, G1=G1, y=y, covar=x, appendbias=False,lik0=null_model,forcefullrank = forcefullrank)
        else:
            return lr.onekerneltest(G1=G1, y=y, covar=x, appendbias=False,lik0=varcomp_test,forcefullrank = self.forcefullrank)
