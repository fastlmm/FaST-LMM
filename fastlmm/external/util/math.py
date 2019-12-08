from __future__ import absolute_import
import numpy as np
import scipy as sp

def mvnormpdf(b, mean, cov, precision=False):
    k = b.shape[0]
    part1 = np.exp(-0.5*k*np.log(2.0*np.pi))
    dev = b-mean
    if precision is False:
        part2 = np.power(np.linalg.det(cov),-0.5)
        part3 = np.exp(-0.5*np.dot(np.dot(dev,np.linalg.inv(cov)),dev))
    else:
        part2 = np.power(np.linalg.det(cov),0.5)
        part3 = np.exp(-0.5*np.dot(np.dot(dev,cov),dev))
    return part1*part2*part3

# Performs np.tr(dot(A, B))
def trace2(A, B):
    assert len(A.shape)==2 and len(B.shape)==2
    assert A.shape[1]==B.shape[0] and A.shape[0]==B.shape[1]
    return np.sum(A.T*B)

def dotd(A, B):
    """Multiply two matrices and return the
    resulting diagonal.
    If A is nxp and B is pxn, it is done in O(pn).
    """
    return np.sum(A * B.T,1)

def ddot(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      ddot(d, mts, left=True) == dot(diag(d), mtx)
      ddot(d, mts, left=False) == dot(mtx, diag(d))
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx

def check_definite_positiveness(A):
    B = np.empty_like(A)
    B[:] = A
    B[np.diag_indices_from(B)] += np.sqrt(np.finfo(np.float).eps)
    try:
        np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        return False
    return True

def check_symmetry(A):
    return abs(A-A.T).max() < np.sqrt(np.finfo(np.float).eps)

def kl_divergence(p,q):
    return np.sum(np.log(p/q)*p)

stl = lambda a, b : sp.linalg.solve_triangular(a, b, lower=True, check_finite=False)
stu = lambda a, b : sp.linalg.solve_triangular(a, b, lower=False, check_finite=False)
