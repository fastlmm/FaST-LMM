import cython
import numpy as np
cimport numpy as cnp
cimport scipy.linalg.cython_lapack as cython_lapack

cdef int ZERO = 0
ctypedef cnp.float64_t REAL_t
ctypedef long long int MKL_INT 

def inverse(mat,identity,pivots,k):
    cdef int K = k
    cdef REAL_t* mat_pointer = <REAL_t *>cnp.PyArray_DATA(mat)
    cdef REAL_t* iden_pointer = <REAL_t *>cnp.PyArray_DATA(identity)
    cdef int* piv_pointer = <int *>cnp.PyArray_DATA(pivots)
    #http://www.math.utah.edu/software/lapack/lapack-d/dgesv.html
    cython_lapack.dgesv(&K,&K,mat_pointer,&K,piv_pointer,iden_pointer,&K,&ZERO)

def dgesvd(jobu,jobvt,m,n,a,lda,s,u,ldu,vt,ldvt,work,lwork):
    #http://www.math.utah.edu/software/lapack/lapack-d/dgesvd.html
    #http://stackoverflow.com/questions/5047503/lapack-svd-singular-value-decomposition
    cdef char* JOBU = jobu
    cdef char* JOBVT = jobvt
    cdef int M = m
    cdef int N = n
    cdef REAL_t* A = <REAL_t *>cnp.PyArray_DATA(a)
    cdef int LDA = lda
    cdef REAL_t* S = <REAL_t *>cnp.PyArray_DATA(s)
    cdef REAL_t* U = <REAL_t *>cnp.PyArray_DATA(u)
    cdef int LDU = ldu
    cdef REAL_t* VT = <REAL_t *>cnp.PyArray_DATA(vt)
    cdef int LDVT = ldvt
    cdef REAL_t* WORK = <REAL_t *>cnp.PyArray_DATA(work)
    cdef int LWORK = lwork
    cdef int INFO = 0
	
    cython_lapack.dgesvd(JOBU,JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,&INFO)

    return INFO


cdef extern from "mkl.h":
    void dgesdd( const char* jobz, const MKL_INT* m, const MKL_INT* n, double* a, 
        const MKL_INT* lda, double* s, double* u, const MKL_INT* ldu, 
        double* vt, const MKL_INT* ldvt, double* work, 
        const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* info );

def pydgesdd(jobz,m,n,a,lda,s,u,ldu,vt,ldvt,work,lwork,iwork):
    #http://www.math.utah.edu/software/lapack/lapack-d/dgesvd.html
    #http://stackoverflow.com/questions/5047503/lapack-svd-singular-value-decomposition
    cdef char* JOBZ = jobz
    cdef MKL_INT M = m
    cdef MKL_INT N = n
    cdef REAL_t* A = <REAL_t *>cnp.PyArray_DATA(a)
    cdef MKL_INT LDA = lda
    cdef REAL_t* S = <REAL_t *>cnp.PyArray_DATA(s)
    cdef REAL_t* U = <REAL_t *>cnp.PyArray_DATA(u)
    cdef MKL_INT LDU = ldu
    cdef REAL_t* VT = <REAL_t *>cnp.PyArray_DATA(vt)
    cdef MKL_INT LDVT = ldvt
    cdef REAL_t* WORK = <REAL_t *>cnp.PyArray_DATA(work)
    cdef MKL_INT LWORK = lwork
    cdef MKL_INT* IWORK = <MKL_INT *>cnp.PyArray_DATA(iwork)
    cdef MKL_INT INFO = 0
	
    dgesdd(JOBZ,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,IWORK,&INFO)

    return INFO
