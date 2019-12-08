from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import scipy.stats as st
import fastlmm.util.mingrid as mingrid
from six.moves import range


class MetaAnalysis(object):
    
    def __init__(self, beta, ste, tau=0):
        self.beta = beta
        self.ste = ste
        self.tau = tau
    
    def var_beta(self):
        var = self.ste * self.ste + self.tau
        return var

    def inverse_variance_weights(self):
        return 1.0 / self.var_beta()

    def z_score(self):
        z_score = self.mean_beta() / self.ste_mean()
        return z_score

    def meta_pvalue(self):
        z_score = self.z_score()        
        chi2 = z_score * z_score
        return st.chi2.sf(chi2, 1)

    def mean_beta(self):
        weights = self.inverse_variance_weights()
        mean_beta = self.beta.dot(weights) / (weights).sum()
        return mean_beta
        
    def ste_mean(self):
        return 1.0 / np.sqrt(self.inverse_variance_weights().sum())

    def log_likelihood(self, tau=None, mean_beta=None, reml=False):
        
        if tau is None:
            var = self.var_beta()
        else:
            var = self.ste * self.ste + tau

        determinant = np.log(var).sum()
        if mean_beta is None:
            # ML (equiv. REML) estimate of beta:
            mean_beta = (self.beta / var).sum() / (1.0 / var).sum()
            if reml:
                # perform REML
                determinant += np.log((1.0 / var).sum()) - np.log(self.beta.shape)
        residuals = (self.beta - mean_beta)
        rss = (residuals * residuals / var).sum()
        
        log_likelihood = - 0.5 * (determinant + rss)
        return log_likelihood


class FixefEffects(MetaAnalysis):

    def __init__(self, beta, ste):
        MetaAnalysis.__init__(self, beta=beta, ste=ste, tau=0)


class RandomEffects(MetaAnalysis):
    """
    We use REML to 


    Quantifying heterogeneity in a meta-analysis
    Julian P. T. Higgins and Simon G. Thompson
    MRC Biostatistics Unit; Institute of Public Health; Robinson Way; Cambridge CB2 2SR; U.K.
    """
    def __init__(self, beta, ste, reml=True):
        self.reml=reml
        tau = self.estimate_tau(beta=beta, ste=ste)
        MetaAnalysis.__init__(self, beta=beta, ste=ste, tau=tau)

    def tau_ml(self, beta, ste):
        meta = MetaAnalysis(beta=beta, ste=ste, tau=0)
        def f(x):
            return -meta.log_likelihood(tau=x, mean_beta=0)

        tau = mingrid.minimize1D(f, evalgrid=None, nGrid=10, minval=0.0, maxval=(beta*beta).mean(), verbose=False, brent=True,check_boundaries=True, resultgrid=None, return_grid=False)
        return tau[0]

    def estimate_tau(self, beta, ste):
        meta = MetaAnalysis(beta=beta, ste=ste, tau=0)
        def f(x):
            return -meta.log_likelihood(tau=x, mean_beta=None, reml=self.reml)

        tau = mingrid.minimize1D(f, evalgrid=None, nGrid=10, minval=0.0, maxval=(beta*beta).mean(), verbose=False, brent=True,check_boundaries=True, resultgrid=None, return_grid=False)
        return tau[0]


class HierarchicalRandomEffects(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    import pylab as plt
    plt.ion()

    N_repeats = 1000
    tau = 0.0
    beta_true = 0.0
    N_tests = 1000

    z_scores = np.zeros(N_repeats)
    p_values = np.zeros(N_repeats)
    z_scores_re = np.zeros(N_repeats)
    p_values_re = np.zeros(N_repeats)
    for i in range(N_repeats):
       
        var = np.random.uniform(size=N_tests)
        ste = np.sqrt(var)
        beta = np.random.normal(size=N_tests) * np.sqrt(var+tau) + beta_true
        
        fe = FixefEffects(beta=beta, ste=ste)
        p_values[i] = fe.meta_pvalue()
        mean_fe = fe.mean_beta()
        ste_fe = fe.ste_mean()
        var_beta = fe.var_beta()
        z_scores[i] = fe.z_score()

        print("Fixed effects:  mean=%.6f, ste=%.6f, pv=%.6f, z_score=%.6f" % (mean_fe, ste_fe, p_values[i], z_scores[i]))

        re = RandomEffects(beta=beta, ste=ste)
        p_values_re[i] = re.meta_pvalue()
        mean_fe = re.mean_beta()
        ste_fe = re.ste_mean()
        var_beta = re.var_beta()
        z_scores_re[i] = re.z_score()
        print("random effects:  mean=%.6f, ste=%.6f, pv=%.6f, z_score=%.6f" % (mean_fe, ste_fe, p_values_re[i], z_scores_re[i]))

    plt.figure(); plt.hist(z_scores, bins=50)
    plt.figure(); plt.hist(p_values)

    plt.figure(); plt.hist(z_scores_re, bins=50)
    plt.figure(); plt.hist(p_values_re)

    plt.figure(); plt.plot(z_scores_re*z_scores_re, z_scores*z_scores, '.')



