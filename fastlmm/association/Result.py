from __future__ import absolute_import
import logging

class Result:
    '''
    'lik0'          : null likelihood
                        'nLL'       : negative log-likelihood
                        'sigma2'    : the model variance sigma^2
                        'beta'      : [D*1] array of fixed effects weights beta
                        'h2'        : mixture weight between Covariance and noise
                        'REML'      : True: REML was computed, False: ML was computed
                        'a2'        : mixture weight between K0 and K1
    'lik1'          : alternative likelihood
                        'nLL'       : negative log-likelihood
                        'sigma2'    : the model variance sigma^2
                        'beta'      : [D*1] array of fixed effects weights beta
                        'h2'        : mixture weight between Covariance and noise
                        'REML'      : True: REML was computed, False: ML was computed
                        'a2'        : mixture weight between K0 and K1
    'nexclude'      : array of the number of excluded snps from null
    'test'          : "lrt", "sc_davies", sc_..."
    '''


    def __init__(self,setname,iset,iperm,ichrm,iposrange):
        self.setname = setname
        self.iset = iset
        self.iperm = iperm
        self.ichrm = ichrm
        self.iposrange = iposrange

    # computing observed lrt statistics and a2 parameters
    @property
    def stat(self):
        return self.test['stat']
        #return 2 * (self.lik0['nLL'] - self.lik1['nLL'])

    @property
    def a2(self):
        return self.test['lik1']['a2']

    @property
    def h2(self):
        try:
            return self.test['lik1']['h2'][0]
        except:
            logging.info("found a scalar h2")
            return self.test['lik1']['h2']

    @property
    def h2_1(self):
        try:
            return self.test['lik1']['h2_1'][0]
        except:
            logging.info("found a scalar h2_1")
            return self.test['lik1']['h2_1']


    @property
    def type(self):
        return self.test['type']

    @property
    def pv(self):
        return self.test['pv']
    
    @property
    def alteqnull(self):
        if 'alteqnull' in self.test:
            return self.test['alteqnull']
        return None

    @property
    def lik0Details(self):
        return self.test['lik0']

    @property
    def lik1Details(self):
        return self.test['lik1']
