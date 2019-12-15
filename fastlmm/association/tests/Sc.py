from __future__ import absolute_import
import fastlmm.association.lrt as lr
import scipy as SP
import fastlmm.util.stats.chi2mixture as c2
import fastlmm.association.score as score
import scipy.linalg as LA
import scipy.stats as ST
from . import tests_util as tu
from six.moves import range

class Sc(object):
    """description of class"""

    __slots__ = ["score"]


    def check_nperm(self,nperm):
        return 0 # ignore request for permuations

    def __init__(self,score):
        self.score=score

    def __str__(self):
        return "sc_{0}".format(self.score)

    def construct(self, Y, X=None, forcefullrank = False, SNPs0 = None, i_exclude = None, nullModel = None, altModel = None,
                  scoring = None, greater_is_better = None):
        '''
        The same code gets executed for both linear and logistic, because the logistic is an approximation.
        '''
        assert nullModel['effect'] == 'mixed' and altModel['effect'] == 'mixed',\
               'You have not used mixed effects for the two kernel case.'

        assert nullModel['link'] == 'linear'
        assert altModel['link'] == 'linear'

        if self.score == 'mom':
            assert nullModel['link'] == 'linear' and altModel['link'] == 'linear', 'You are allowed to use '\
                   'only the linear link for sc mom test.'        
        G0,K0=tu.set_snps0(SNPs0=SNPs0,sample_size=Y.shape[0],i_exclude=i_exclude,forcefullrank=forcefullrank)          
        return score.scoretest2K(Y=Y[:,SP.newaxis],X=X,K=K0,G0=G0)

    def construct_no_backgound_kernel(self, Y,X, forcefullrank, nullModel, altModel, scoring,
                                      greater_is_better):
        if nullModel['link']=='logistic':

            assert nullModel['effect'] == 'fixed' and altModel['link'] == 'logistic' and\
                   altModel['effect'] == 'mixed', 'By choosing the logistic link for the '\
                                                  'null model, you should also select '\
                                                  'fixed effect for null model, '\
                                                  'mixed effect and logistic link for alt '\
                                                  'model.'
            return score.scoretest_logit(Y=Y[:,SP.newaxis],X=X)

        elif nullModel['link']=='linear':

            assert nullModel['effect'] == 'fixed' and altModel['link'] == 'linear' and\
                   altModel['effect'] == 'mixed', 'By choosing the linear link for the '\
                                                  'null model, you should also select '\
                                                  'fixed effect for null model, '\
                                                  'mixed effect and linear link for alt '\
                                                  'model.'            
            return score.scoretest(Y=Y[:,SP.newaxis],X=X)
        else:
            raise Exception("Invalid link for score null model: " + link)

    def pv(self,squaredform,expectationsqform,varsqform,GPG):
        if self.score == "davies":
            return Sc.pv_davies(squaredform,expectationsqform,varsqform,GPG)
        elif self.score == "mom":
            return Sc.pv_mom(squaredform,expectationsqform,varsqform,GPG)
        else:
            raise Exception("Don't know sc score " + self.score)

    @staticmethod
    def pv_davies_eig(squaredform,eigvals):
            import fastlmm.util.stats.quadform as qf
            #result = qf.qf(squaredform, eigvals,acc=1e-04,lim=10000)    #settings to match R-based results
            result = qf.qf(squaredform, eigvals,acc=1e-07) #decided on 1e-7 after experimentation between -4 and -12. Thresh on exp in QFC.C seems to have no effect
            return result[0]

    @staticmethod
    def pv_davies(squaredform,expectationsqform,varsqform,GPG):
        eigvals=LA.eigh(GPG,eigvals_only=True)
        pv = Sc.pv_davies_eig(squaredform,eigvals)
        return pv

    @staticmethod
    def pv_mom(squaredform,expectationsqform,varsqform,GPG):
        '''
        Do moment of matching on the scaled chi square approximation to the null, as in Li and Cui
        '''
        dofchi2= 2.0*varsqform*expectationsqform*expectationsqform
        scalechi2 = 0.5/(expectationsqform*varsqform)
        pv = ST.chi2.sf(squaredform/scalechi2,dofchi2)
        return pv

    @property
    def npvals(self):
        return 1 # return the 1 type of p-Value

    def w2(self, G0, result):
        return 0.0

    def lrt_method(self, result):
        return 0.0

    def pv_adj_from_result(self, result):
        return result.pv

    def pv_adj_and_ind(self, nperm, pv_adj, nullfit, lrt, lrtperm,
                       alteqnull, alteqnullperm, qmax, nullfitfile, nlocalperm):
        ind = pv_adj.flatten().argsort()
        return pv_adj, ind

    def write(self, fp, ind, result_dict, pv_adj, detailed_table):
        fp.write("\t".join(("SetId", "P-value", "#SNPs_in_Set", "#ExcludedSNPs","test stat", "chrm", "pos. range")) + "\n")
        for i in range(len(ind)):
            ii = ind[i]
            result = result_dict[ii]
            fp.write("\t".join((result.setname, str(pv_adj[ii]), 
                                str(result.setsize),str(result.nexclude), 
                                str(result.stat), result.ichrm, result.iposrange + "\n")))

    def pv_etc(self, filenull, G0_to_use, G1, y, x, null_model, varcomp_test, forcefullrank):
        if forcefullrank: raise Exception("full rank score not implemented here")
        if self.filenull is not None:
            pv, garbage = score.twokerneltest(G0=G0_to_use, G1=G1, y=y, nulldistrib=self, covar=x, appendbias=False,test2K=null_model)
            return pv, SP.nan, SP.nan
        else:
            pv = score.onekerneltest(G0=G1, y=y, nulldistrib=self, covar=x, appendbias=False)
            return pv, SP.nan, SP.nan
