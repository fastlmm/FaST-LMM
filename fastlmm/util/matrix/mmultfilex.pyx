import cython
import numpy as np
cimport numpy as np

cdef extern from "mmultfile.h":
    int _mmultfile_b_less_aatbx "mmultfile_b_less_aatbx"(char* a_filename, long long offset, long long row_count, long long a_col_count, long long b_col_count, double* b1, double* aaTb, double* aTb, int num_threads, long long log_frequency)
    int _mmultfile_atax "mmultfile_atax"(char* a_filename, long long offset, long long row_count, long long col_count, long long work_index, long long work_count, double* ata_piece, int num_threads, long long log_frequency)


def mmultfile_b_less_aatbx(a_filename, long long offset, int row_count, int a_col_count, int b_col_count, np.ndarray[np.float64_t, ndim=2] b1, np.ndarray[np.float64_t, ndim=2] aaTb, np.ndarray[np.float64_t, ndim=2] aTb, int num_threads, int log_frequency):
   return _mmultfile_b_less_aatbx(a_filename, offset, row_count, a_col_count, b_col_count, <double*>b1.data, <double*> aaTb.data, <double*> aTb.data, num_threads, log_frequency)

def mmultfile_atax(a_filename, long long offset, int row_count, int col_count, int work_index, int work_count, np.ndarray[np.float64_t, ndim=2] ata_piece, int num_threads, int log_frequency):
   return _mmultfile_atax(a_filename, offset, row_count, col_count, work_index, work_count, <double*>ata_piece.data, num_threads, log_frequency)
