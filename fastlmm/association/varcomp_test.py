from __future__ import absolute_import
import numpy as np
from six.moves import range

class varcomp_test(object):
    __slots__ = ["Y","X"]
  
    def __init__(self, Y, X=None, appendbias = False):
        self.Y=Y

        N=self.Y.shape[0]
        if X is None:
            self.X = np.ones((N,1))
        elif appendbias:
            assert self.hasBias() is False, ('You are trying to append a bias column in a dataset '
                                             'that already has one.')
            self.X=SP.hstack((np.ones((N,1)),X))
        else:
            self.X = X

    def _updateYX(self, Y, X):
        self.Y=Y        
        self.X=X

    def hasBias(self):

        for i in range(self.X.shape[1]):
            if np.all(self.X[:,i] == 1.0):
                return True

        return False

    def testG(self, G1, type = None,i_exclude=None):
        raise NotImplementedError
        pv = None
        stat = None
        return pv, stat