from __future__ import absolute_import
import fastlmm.association.lrt as lr
import scipy as sp
import fastlmm.util.stats.chi2mixture as c2

from . import tests_util as tu
from six.moves import range

class Lrt(object):
    """description of class"""
    def check_nperm(self,nperm):
        return nperm #permutations are fine, so just return

    def __str__(self):
        return "lrt"


    def construct(self, Y, X=None, forcefullrank = False, SNPs0 = None, i_exclude = None, nullModel = None, altModel = None,
                  scoring = None, greater_is_better = None):
        G0,K0=tu.set_snps0(SNPs0=SNPs0,sample_size=Y.shape[0],i_exclude=i_exclude)
        return lr.lrt(Y=Y, X=X, model0=None, appendbias=False, forcefullrank=forcefullrank, G0=G0,K0=K0, nullModel=nullModel, altModel=altModel)

    def construct_no_backgound_kernel(self, Y, X, forcefullrank, nullModel, altModel, scoring,
                                      greater_is_better):
        return self.construct(Y=Y,X=X,forcefullrank=forcefullrank,nullModel=nullModel,altModel=altModel,scoring=scoring,greater_is_better=greater_is_better)

    def pv(squaredform,expectationsqform,varsqform,GPG):
        raise Exception("'pv' doesn't apply to lrt only to davies")

    @property
    def npvals(self):
        return 1 # return only 1 type of p-value

    def w2(self, G0, result):
        if G0 is not None:
            return result.a2
        else:
            return result.h2

    def lrt_method(self, result):
        return result.stat

    def pv_adj_from_result(self, result):
        '''
        If local aUD exists, take that, if not, take the raw local.
        '''        
        if "pv-local-aUD" in result.test and not sp.isnan(result.test["pv-local-aUD"]):
            return result.test["pv-local-aUD"]
        elif "pv-local" in result.test:
            return result.test["pv-local"]
        else:
            return sp.nan

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
            colnames.append("Alt_a2")
        
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
            ll0=str( -(result.stat/2.0+result.test['lik1']['nLL']) )

            if "pv-local-aUD" in result_dict[0].test:
                rowvals = [result.setname, str(-result.test['lik1']['nLL']), ll0,
                           str(pv_adj[ii]),str(result.test['pv-local']),str(result.pv), str(result.setsize),
                           str(result.nexclude), result.ichrm, result.iposrange]
            else:
                rowvals = [result.setname, str(-result.test['lik1']['nLL']), ll0,
                           str(pv_adj[ii]), str(result.pv), str(result.setsize),
                           str(result.nexclude), result.ichrm, result.iposrange]

            if signal_ratio:
                rowvals.append(str(result.h2))
                rowvals.append(str(result.a2))

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
