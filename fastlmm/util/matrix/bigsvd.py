from __future__ import absolute_import
from __future__ import print_function
import os
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import datetime
import fastlmm.util.matrix.cample as cample

def big_sdd(a):
    if a.flags['C_CONTIGUOUS']:
        #Could use https://pypi.org/project/fastremap/ to avoid a copy
        a = np.require(a,requirements=['F'])
    assert a.flags['F_CONTIGUOUS'],"expect a to be order 'F'"
    minmn = min(a.shape[0],a.shape[1])
    s = np.zeros(minmn,order='F') #!!! empty faster than zero? (and also for the next items)
    u = np.zeros((a.shape[0],a.shape[0]),order='F') 
    vt = np.zeros((a.shape[1],a.shape[1]),order='F')
    work = np.zeros(1,order='F')
    iwork = np.zeros(8*minmn,dtype=np.int64)
    info = cample.pydgesdd( b"A",
                            a.shape[0], #row count
                            a.shape[1], #col count
                            a,
                            a.shape[0], #1-based leading row
                            s,
                            u,
                            a.shape[0],   #LDU
                            vt,
                            a.shape[1],   #LDVT
                            work,
                            -1,
                            iwork)
    assert info==0
    work = np.zeros(int(work[0]),order='F') #!!! empty faster than zero? (and also for the next items)
    info = cample.pydgesdd( b"A",
                            a.shape[0], #row count
                            a.shape[1], #col count
                            a,
                            a.shape[0], #1-based leading row
                            s,
                            u,
                            a.shape[0],   #LDU
                            vt,
                            a.shape[1],   #LDVT
                            work,
                            work.shape[0],
                            iwork)
    assert info==0
    return u, s, vt


def lapack_svd(a):
    assert a.flags['F_CONTIGUOUS'],"expect a to be order 'F'"
    minmn = min(a.shape[0],a.shape[1])
    s = np.zeros(minmn,order='F') #!!! empty faster than zero? (and also for the next items)
    u = np.zeros((a.shape[0],a.shape[0]),order='F')
    vt = np.zeros((a.shape[1],a.shape[1]),order='F')
    work = np.zeros(1,order='F')
    info = cample.dgesvd(  b"A",
                            b"A",
                            a.shape[0], #row count
                            a.shape[1], #col count
                            a,
                            a.shape[0], #1-based leading row
                            s,
                            u,
                            a.shape[0],   #LDU
                            vt,
                            a.shape[1],   #LDVT
                            work,
                            -1)
    assert info==0
    work = np.zeros(int(work[0]),order='F') #!!! empty faster than zero? (and also for the next items)
    info = cample.dgesvd(  b"A", #Called twice. Once to get size and once to do the work.
                            b"A",
                            a.shape[0], #row count
                            a.shape[1], #col count
                            a,
                            a.shape[0], #1-based leading row
                            s,
                            u,
                            a.shape[0],   #LDU
                            vt,
                            a.shape[1],   #LDVT
                            work,
                            work.shape[0])
    assert info==0
    return u, s, vt


def lapack_inverse(a):
    b = identity.copy()
    cample.inverse(a,b,np.zeros(K,dtype=np.intc),K)
    return b


if __name__ == '__main__':


    K=10
    x = np.random.random((K,K))
    identity = np.eye(K)
    assert np.allclose(np.linalg.inv(x),lapack_inverse(x))


    for m_row in  [6, 1, 2, 4,10,500]:#[6,50000,100000]: #300,500,1000,2000,4000,6000,9000,15000]:# [6, 1, 2, 4,10,500]:#,15000,20000,50000]: #50000,500,100000,15000,9000,2000]:#
        #row_count, col_count = (9000,9000) -> 1G, 87%, 30 min

        if m_row == 6:
            n_col = 5
            #https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgesvd_col.c.htm
            a = np.array([[8.79,   9.93,   9.83,    5.45,   3.16],
                          [6.11,   6.91,   5.04,   -0.27,   7.98],
                          [-9.15,  -7.93,  4.86,    4.85,   3.01],
                          [ 9.57,  1.64,   8.83,    0.74,   5.80],
                          [-3.49,  4.02,   9.80,   10.00,   4.27],
                          [9.84,   0.15,   -8.99,  -6.02,  -5.31]],order="F")
        else:
            n_col =  m_row + 2
            logging.info("generating {0}x{1}".format(m_row,n_col))
            np.random.seed(0)
            a = np.random.randn(m_row, n_col).astype(np.float)# + np.random.randn(m_row, n_col)
            #for i in xrange(m_row):
            #    for j in xrange(i+1,n_col):
            #        a[i,j] = a[j,i]

        min_row_col = min(m_row,n_col)

        if True:
            now = datetime.datetime.now()
            logging.info("doing large big_sdd")
            ux, sx, vtx = big_sdd(np.array(a,order="F"))
            logging.info("done with big_sdd {0}x{1} in time {2}".format(m_row,n_col, datetime.datetime.now()-now))
            Sx = np.zeros((m_row,n_col))
            Sx[:min_row_col, :min_row_col] = np.diag(sx)
            assert np.allclose(a, np.dot(ux, np.dot(Sx, vtx)))

        if True:
            now = datetime.datetime.now()
            logging.info("doing lapack_svd")
            ux, sx, vtx = lapack_svd(np.array(a,order="F"))
            logging.info("done with lapack_svd {0}x{1} in time {2}".format(m_row,n_col, datetime.datetime.now()-now))
            Sx = np.zeros((m_row,n_col))
            Sx[:min_row_col, :min_row_col] = np.diag(sx)
            assert np.allclose(a, np.dot(ux, np.dot(Sx, vtx)))

        if True:
            now = datetime.datetime.now()
            logging.info("doing small np.linalg.svd")
            U, s, V = np.linalg.svd(a, full_matrices=False,compute_uv=True)
            logging.info("done with np.linalg.svd {0}x{1} in time {2}".format(m_row,n_col, datetime.datetime.now()-now))
            print(U.shape, s.shape, V.shape, end=' ') 
            S = np.zeros((min_row_col,min_row_col))
            S = np.diag(s)
            assert np.allclose(a, np.dot(U, np.dot(S, V)))
