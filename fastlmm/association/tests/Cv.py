from __future__ import absolute_import
import fastlmm.association.lrt as lr
import scipy as SP
import fastlmm.util.stats.chi2mixture as c2
import fastlmm.association.testCV as testCV
import logging
from six.moves import range

class Cv(object):
    """description of class"""

    __slots__ = []

    def check_nperm(self,nperm):
        return nperm #cv handles nperm>0 so just return whatever value gomes in

    def __init__(self):
        pass

    def __str__(self):
        #return "cv_{0}".format(self.penalty) horta20130609
        return "cv"

    def construct(self, Y, X=None, forcefullrank = None, SNPs0 = None, nullModel = None, altModel = None,
                  scoring = None, greater_is_better = None):
        return testCV.testCV(Y=Y[:,SP.newaxis],X=X,G0=G0,
                             nullModel=nullModel,altModel=altModel,
                             scoring=scoring, greater_is_better=greater_is_better)

    def construct_no_backgound_kernel(self, Y,X, forcefullrank, nullModel, altModel, scoring,
                                      greater_is_better):
        return construct(Y,X,forcefullrank,None,nullModel,altModel,scoring,greater_is_better)

    def pv(squaredform,expectationsqform,varsqform,GPG):
        raise Exception("'pv' doesn't apply to cv only to davies")

    @property
    def npvals(self):
        return 1 #returns 1 type of p-value

    def w2(self, G0, result):
        return SP.nan

    def lrt_method(self, result):
        return result.test['stat'] #iset is the index of the test (irrespective of permutation)

    def pv_adj_from_result(self, result):
        return SP.nan;

    def write(self, fp,ind, result_dict, pv_adj, detailed_table):
        fp.write("\t".join(("SetId", "stat","scoreAlt", "scoreNull", "P-value_adjusted", "#SNPs_in_Set", "#ExcludedSNPs", "Chrm", "Pos. range")) + "\n")
        for i in range(len(ind)):
            ii = ind[i]
            result = result_dict[ii]
            lik1=result.test['scores'].mean()
            lik0=result.test['scores0'].mean()
            fp.write("\t".join((result.setname, str(result.test['stat']), str(lik1), str(lik0),
                                str(pv_adj[ii]), str(result.setsize),
                                str(result.nexclude), result.ichrm, result.iposrange)
                                + "\n"))


    def pv_etc(self, filenull, G0_to_use, G1, y, x, null_model, varcomp_test, forcefullrank):
        return [SP.nan,SP.nan,SP.nan]

    #!! could these three methods be move to __init__.py?
    @staticmethod
    def pv_adj_and_ind(nperm, pv_adj, nullfit, lrt, lrtperm,
                       alteqnull, alteqnullperm, qmax, nullfitfile,nlocalperm,sort=True):
        
        if nullfit=="ml":
            logging.info("using ml fit for parameterized null distrib")
            assert nullfitfile is None, "not implemented: nullfitfile with ml fit"
            pv_adj,mixture,scale,dof=Cv.lrtpvals_mlfit(nperm, lrt, lrtperm)
        elif nullfit=="qq":
            logging.info("using qq fit for parameterized null distrib")
            pv_adj,mixture,scale,dof=Cv.lrtpvals_qqfit(nperm, lrt, lrtperm,
                                                       alteqnull, alteqnullperm,qmax=qmax,nullfitfile=nullfitfile)
        elif nullfit=="abs":
            assert nullfitfile is None, "not implemented: nullfitfile with abs fit"
            assert nperm==0, "currently can only use abs fit with no permutations"
            logging.info("using qq fit for parameterized null distrib, without using any permutations")
            pv_adj,mixture,scale,dof=Cv.lrtpvals_qqfit(nperm, lrt, lrtperm,
                                                       alteqnull, alteqnullperm,
                                                       abserr=True,fitdof=False,dof=1,qmax=qmax)
        else:
            raise Exception("dont' know nullfit='{0}'".format(nullfit))
        logging.info("mixture (non-zero dof)="+str(mixture))
        logging.info("dof="+str(dof))
        logging.info("scale="+str(scale))
        if sort:
            ind = pv_adj.argsort()
        else:
            ind = None
        return pv_adj, ind

  
    @staticmethod
    def lrtpvals_qqfit(nperm, lrt, lrtperm, alteqnull, alteqnullperm, qmax=None, abserr=False,fitdof=True, dof=None, nullfitfile=None):
        '''
        Fit the parameters of the null distribution using "quantile regession" on some fraction of the most signficant data points
        '''
        if (nperm > 0):            
            logging.info("estimating mixture parameters for permuted data\nusing quantile regression of log-pvalues with qmax=" + str(qmax) + " from permutations...")
            mix = c2.chi2mixture( lrt = lrtperm, qmax = qmax, alteqnull = alteqnullperm,abserr=abserr,fitdof=fitdof,dof=dof)
            
                        
            res = mix.fit_params_Qreg() # paramter fitting             
         
            imax=res['imax']
            mse=res['mse']
            logging.info("# of pvals used for nullfit=" + str(imax))
            pv_adj = mix.sf(lrt=lrt,alteqnull=alteqnull) # getting p-values for real data
            logging.info(" Done")
            logging.info("adjusting the observed p-values ...")

            logging.info(" Done")
        elif nullfitfile is not None:
            logging.info("estimating mixture parameters for permuted data\nusing quantile regression of log-pvalues with qmax=" 
                      + str(qmax) + " from STORED permutations in " + nullfitfile)           
                        
            #read in p-vals and alteqnull for from file                        
            colnames={"2*(LL(alt)-LL(null))","alteqnull","setsize"}
            import fastlmm.util.util as ut
            import numpy as np
            dat=ut.extractcols(nullfitfile,colnameset=colnames,dtypeset={"2*(LL(alt)-LL(null))": np.float64})
            lrtfile=dat["2*(LL(alt)-LL(null))"]
            alteqnullfile=dat["alteqnull"]
            
            mix = c2.chi2mixture( lrt = lrtfile, qmax=qmax, alteqnull=alteqnullfile,abserr=abserr,fitdof=fitdof,dof=dof)
                                    
            res = mix.fit_params_Qreg() # paramter fitting             
         
            imax=res['imax']
            mse=res['mse']
            logging.info("# of pvals used for nullfit=" + str(imax))          
            pv_adj = mix.sf(lrt=lrt,alteqnull=alteqnull) # getting p-values for real data
            logging.info(" Done")
            logging.info("adjusting the observed p-values ...")

            logging.info(" Done")
        else:
            logging.info("nperm = " + str(nperm) + " : No permutations were performed.")
            logging.info("estimating mixture parameters for non-permuted data\nusing quantile regression of log-pvalues with qmax=" + str(qmax) + "...")
            mix = c2.chi2mixture( lrt = lrt, qmax = qmax, alteqnull = alteqnull,abserr=abserr,fitdof=fitdof,dof=dof)
            res = mix.fit_params_Qreg() # paramter fitting
            imax=res['imax']
            logging.info("# of pvals used for nullfit=" + str(imax))
            pv_adj = mix.sf() #getting p-values
            logging.info(" Done")
        if mix.mixture==0:
            #raise Exception("only zero dof component items found")
            logging.info("*****WARNING*****: only zero dof component items found")
        return pv_adj,mix.mixture,mix.scale,mix.dof#,imax,mse

    @staticmethod
    def lrtpvals_mlfit(nperm, a2, a2perm, lrt, lrtperm):
        '''
        Fit the parameters of the null distribution using maximum likelihood on all the data points
        '''
        raise Exception("made changes to use alteqnull and did not modify this code as it looks obsolete")
        if (nperm > 0):
            logging.info("estimating mixture parameters for permuted data using maximum likelihood...")
            # estimating mixture parameters for permuted data, using
            [pvperm,mixture,scale,dof,i0] = c2.computePVmixtureChi2_old(lrtperm,a2perm, tol= 0.0, mixture=None, scale = None, dof = None)
            logging.info("        Done")
            logging.info("adjusting the observed p-values ...")
            # adjust the observed p-values with estimated null mixture, scale, dof parameters
            [pv_adj,mixture,scale,dof,i0] = c2.computePVmixtureChi2_old(lrt,a2, tol= 0.0, mixture=mixture, scale = scale, dof = dof)
            logging.info("        Done")
        else:
            logging.info("nperm = " + str(nperm) + " : No permutations were performed...")
            logging.info("adjusting the observed p-values ...")
            # adjust the observed p-values with estimated null mixture, scale, dof parameters
            [pv_adj,mixture,scale,dof,i0] = c2.computePVmixtureChi2_old(lrt,a2, tol= 0.0, mixture=None, scale = None, dof = None)
            logging.info("        Done")
        return pv_adj,mixture,scale,dof
