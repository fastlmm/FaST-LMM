import numpy as np

cimport numpy as np

cdef extern from "QFC.h":
	double _qf_swig "qf_swig"(double* lb1, int len_lb1, double* nc1, int len_nc1, int* n1, int len_n1, double sigma, double c1, int lim1, double acc, double* trace, int len_trace, int* ifault, int len_ifault)

def qf(np.ndarray[np.float64_t, ndim=1] lb1, np.ndarray[np.float64_t, ndim=1] nc1, np.ndarray[np.int32_t, ndim=1] n1, sigma, c1, lim1, acc, np.ndarray[np.float64_t, ndim=1] trace, np.ndarray[np.int32_t, ndim=1] ifault):
	len_lb1 = lb1.shape[0]
	len_nc1	= nc1.shape[0]
	len_n1 = n1.shape[0]
	len_trace = trace.shape[0]
	len_ifault = ifault.shape[0]
	#http://wiki.cython.org/tutorials/NumpyPointerToC

	qfval = _qf_swig(<double*> lb1.data, 
	<int> len_lb1, 
	<double*> nc1.data, 
	<int> len_nc1, 
	<int*> n1.data, 
	<int> len_n1, 
	sigma, 
	c1, 
	<int> lim1, 
	acc, 
	<double*> trace.data, 
	<int> len_trace, 
	<int*> ifault.data, 
	<int> len_ifault)
	return qfval



    