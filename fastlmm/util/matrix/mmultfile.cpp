#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mmultfile.h"
#include <iostream> 
#include <fstream> 
#include <vector>
#include <assert.h> 
#include <omp.h>
#include <ios>
#include <sys/stat.h>
#include "mkl.h"
using namespace std;

// could add code so that if the two blocks are the same, then only do half the *'s in the dot product
int mmultfile_atax(char* a_filename, long long offset, long long iid_count, long long sid_count, long long work_index, long long work_count, double* ata_piece, int num_threads, long long log_frequency)
{
	omp_set_num_threads(num_threads);
	long long start = sid_count * work_index / work_count;
	long long stop = sid_count * (work_index + 1) / work_count;
	long long max_space = sid_count / work_count + (sid_count % work_count != 0);

	std::vector<double> buffer_0(iid_count * max_space);
	std::vector<double> buffer_1(iid_count * max_space);
	std::vector<double> buffer_2(iid_count * max_space);
	std::vector<double>* ref_cur = &buffer_0;
	std::vector<double>* ref_next = &buffer_2;

	FILE* pFile;
	if ((pFile = fopen(a_filename, "rb")) == NULL)
	{
		PyErr_SetString(PyExc_IOError, "The file did not open.");
		return -1;
	}

	if (
#ifdef _WIN32
		_fseeki64(pFile, offset + start * iid_count * sizeof(double), SEEK_SET)
#elif __APPLE__
		fseeko(pFile, offset + start * iid_count * sizeof(double), SEEK_SET)
#else
		fseeko64(pFile, offset + start * iid_count * sizeof(double), SEEK_SET)
#endif
		!= 0)
	{
		PyErr_SetString(PyExc_IOError, "The file did not seek.");
		return -1;
	}

	size_t byte_count = fread((char*)&buffer_0[0], sizeof(double), iid_count * (stop - start), pFile);
	if (ferror(pFile) || feof(pFile) || byte_count==0)
	{
		PyErr_SetString(PyExc_IOError, "Buffer read failed.");
		return -1;
	}

	int return_value = 0;
	for (long long i = work_index; i < work_count; ++i) {
		if (log_frequency > 0 && i % log_frequency == 0)
		{
			printf("For work_index=%lld of %lld, processing i=%lld (in %lld..%lld) (iid_count=%lld, sid_count=%lld, num_threads=%d)\n", work_index, work_count, i, work_index, work_count, iid_count, sid_count, num_threads);
		}
		else if (log_frequency == -2)
		{
			printf("For work_index=%lld of %lld, processing i=%lld (in %lld..%lld) (iid_count=%lld, sid_count=%lld, num_threads=%d)\n", work_index, work_count, i, work_index, work_count, iid_count, sid_count, num_threads);
			printf("SKIPPING computation\n");
		}

		long long starti = sid_count * i / work_count;
		long long stopi = sid_count * (i + 1) / work_count;
		long long nexti = sid_count * (i + 2) / work_count;

		//ata_piece[starti - start:stopi - start, : ] = np.dot(ref_cur.T, buffer_0)
#pragma omp parallel default(none) shared(stop,start,iid_count,cerr,ref_cur,buffer_0,ata_piece,sid_count,pFile,ref_next,work_count,log_frequency,work_index,num_threads,buffer_1,starti,stopi,nexti,PyExc_IOError,return_value)
		{

			// See the equivalent Python code for a full explanation of the algorithm. In summary:
			// We are taking the matrix multiplication of one chunk against itself and all later chunks.
			// We loop on variable i, the index to the second chunk. Inside the loop we do
			// this in parallel:
			//        * On the main thread, read the data for the i+1 chunk, unless already at end of the file.
			//               This means our reading is always one chunk ahead. This works because the first
			//               matrix multiply is the first chunk with itself.
			//        * On all threads, do the matrix multiply (in parallel) for the first chuck with chunk i.
#pragma omp master
			if (nexti <= sid_count)
			{
				if (log_frequency > 0) printf("reading next chunk\n");

				size_t byte_count = fread((char*)&(*ref_next)[0], sizeof(double), iid_count * (nexti - stopi), pFile);
				if (ferror(pFile) || feof(pFile) || byte_count==0)
				{
					PyErr_SetString(PyExc_IOError, "Buffer read failed.");
					return_value = -1;
				}

				if (log_frequency > 0) printf("finished reading next chunk======================================\n");
			}


#pragma omp for schedule(dynamic)
			for (long long j = 0; j < stopi - starti; ++j) {
				if (log_frequency > 0)	printf("Doing computation %lld\n", j);
				long long j_iid_count = j * iid_count;
				for (long long k = 0; k < stop - start; ++k) {
					long long k_iid_count = k * iid_count;
					double temp = 0.0;
					for (long long m = 0; m < iid_count; ++m) {
						temp += (*ref_cur)[j_iid_count + m] * buffer_0[k_iid_count + m];
					}
					ata_piece[(j + starti - start) * (stop - start) + k] = temp;
				}
				if (log_frequency > 0) printf("done with computation %lld\n", j);
			}
			if (log_frequency > 0) printf("done with parallel loop\n");
		}
		if (log_frequency > 0) printf("done with parallel computation\n");

		if (i == work_index) {  // We just finished the first loop, so before the swap, point at buffer #0
			ref_cur = &buffer_1;
		}

		//Swap ref_cur and ref_next
		std::vector<double>* slicei_temp = ref_cur;
		ref_cur = ref_next;
		ref_next = slicei_temp;
	}

	fclose(pFile);
	if (log_frequency > 0) printf("finished all computation\n");

	return return_value;
}

int mmultfile_b_less_aatbx(char* a_filename, long long offset, long long iid_count, long long train_sid_count, long long test_sid_count, double* b1, double* aaTb, double* aTb, int num_threads, long long log_frequency)
{
	//speed idea: compile for release (and optimize)
	//use MKL to multiply???
	//remove the assert???
	//Are copies really needed?
	//is F, vc C order the best?
	//would bigger snp blocks be better
	std::fstream fs;
	fs.open(a_filename, std::fstream::in | std::fstream::binary);
	fs.seekg(offset, ios::beg);

	//We double buffer
	std::vector<double> buffer0(iid_count);
	std::vector<double> buffer1(iid_count);
	std::vector<double>* ref_write = &buffer0;
	std::vector<double>* ref_read = &buffer1;
	fs.read((char*)&buffer1[0], sizeof(double) * iid_count);
	if (!fs.good())
	{
		PyErr_SetString(PyExc_IOError, "Buffer read failed.");
		return -1;
	}

	int return_value = 0;
	for (long long train_sid_index = 0; train_sid_index < train_sid_count; ++train_sid_index) {
		if (log_frequency > 0 && train_sid_index % log_frequency == 0)
		{
			printf("\rProcessing column train_sid_index=%lld of %lld (iid_count=%lld, test_sid_count=%lld)               ", train_sid_index, train_sid_count, iid_count, test_sid_count);
		}

		omp_set_num_threads(num_threads);
		long long test_sid_index;
#pragma omp parallel private(test_sid_index) shared(fs, test_sid_count,train_sid_index,iid_count,aTb,ref_write,ref_read,b1,aaTb) //default(none) removed to work around an 'error' on Linux and Python2
		{
#pragma omp for
			for (test_sid_index = -1; test_sid_index < test_sid_count; ++test_sid_index) {
				if (test_sid_index == -1) { //While most work items calculate, the first fills the next buffer.
					if (train_sid_index + 1 < train_sid_count)
					{
						fs.read((char*)&(*ref_write)[0], sizeof(double) * iid_count);
						if (!fs.good())
						{
							PyErr_SetString(PyExc_IOError, "Buffer read failed.");
							return_value = -1;
						}
					}
				}
				else {
					long long i = train_sid_index * test_sid_count + test_sid_index;
					long long j = iid_count * test_sid_index;
					double aTbi = 0;
					for (long long iid_index = 0; iid_index < iid_count; ++iid_index) {
						aTbi += (*ref_read)[iid_index] * b1[iid_index + j];
					}
					aTb[i] = aTbi;
					for (long long iid_index = 0; iid_index < iid_count; ++iid_index) {
						aaTb[iid_index + j] -= (*ref_read)[iid_index] * aTbi;
					}
				}
			}
		}

		// Swap buffers
		std::vector<double>& buffer_temp = (*ref_read);
		(*ref_read) = (*ref_write);
		(*ref_write) = buffer_temp;
	}
	if (log_frequency > 0) printf("\n");
	return return_value;
}