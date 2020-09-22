from __future__ import absolute_import
from __future__ import print_function
import os
import logging
import numpy as np
import datetime
from fastlmm.util.matrix.mmultfilex import mmultfile_b_less_aatbx, mmultfile_atax #May need to install "Microsoft Visual C++ 2008 SP1 Redistributable Package (x64)"
import multiprocessing
from pysnptools.util.mapreduce1 import map_reduce
from pysnptools.kernelreader import KernelData, KernelNpz
import time
from pysnptools.util import format_delta
from pysnptools.snpreader import SnpMemMap
from six.moves import range

def get_num_threads():
    if 'MKL_NUM_THREADS' in os.environ:
        return int(os.environ['MKL_NUM_THREADS'])
    else:
        return multiprocessing.cpu_count()

def mmultfile_ata(memmap_lambda,writer,sid,work_count,name,runner,force_python_only=False):
    sid_count = len(sid)
    piece_count = work_count * 2
    log_frequency = 1 if logging.getLogger().level <= logging.INFO else 0

    def debatch_closure(piece_index):
        return sid_count * piece_index // piece_count

    def mapper_closure(work_index):
        memmap = memmap_lambda()
        piece_index0 = work_index
        piece_index1 = piece_count-work_index-1
        gtg_piece0 = mmultfile_ata_piece(memmap.filename,memmap.offset,piece_index0,piece_count,log_frequency=log_frequency,force_python_only=force_python_only)
        gtg_piece1 = mmultfile_ata_piece(memmap.filename,memmap.offset,piece_index1,piece_count,log_frequency=log_frequency,force_python_only=force_python_only)
        return [[piece_index0, gtg_piece0],[piece_index1, gtg_piece1]]

    def reducer_closure(result_result_sequence):
        logging.info("starting ata reducer")
        iid = [[value, value] for value in sid]
        gtg_data = KernelData(iid=iid, val=np.zeros((sid_count,sid_count)))
        for result_result in result_result_sequence:
            for piece_index, gtg_piece in result_result:
                logging.info("combining ata reducer {0}".format(piece_index))
                start = debatch_closure(piece_index)
                stop = debatch_closure(piece_index+1)
                gtg_data.val[start:,start:stop] = gtg_piece
                gtg_data.val[start:stop,start+gtg_piece.shape[1]:] = gtg_piece[gtg_piece.shape[1]:,:].T
        result = writer(gtg_data)
        return result

    gtg_npz_lambda = map_reduce(range(work_count),
               mapper=mapper_closure,
               reducer=reducer_closure,
               runner=runner,
               name=name,
               input_files=[],
               output_files=[]
               )

def mmultfile_ata_piece(a_filename, offset, work_index=0, work_count=1,log_frequency=-1, force_python_only=False):
    t0_gtg = time.time()
    if log_frequency > 0:
        logging.info("ata_piece: Working on piece {0} of {1}.".format(work_index, work_count))

    a = SnpMemMap(a_filename)

    def debatch_closure(work_index2):
        return a.sid_count * work_index2 // work_count
    start = debatch_closure(work_index)
    stop = debatch_closure(work_index+1)

    ata_piece = np.zeros((a.sid_count-start,stop-start),order='C')

    do_both = False
    if force_python_only or do_both:
        with open(a.filename,"rb") as fp:
            fp.seek(a.offset+start*a.iid_count*8)
            slice = np.fromfile(fp, dtype=np.float64, count=a.iid_count*(stop-start)).reshape(a.iid_count,stop-start,order="F")
            for i in range(work_index,work_count):
                starti = debatch_closure(i)
                stopi = debatch_closure(i+1)
                if i > work_index:
                    slicei = np.fromfile(fp, dtype=np.float64, count=a.iid_count*(stopi-starti)).reshape(a.iid_count,stopi-starti,order="F")
                else:
                    slicei = slice
                if log_frequency > 0 and i%log_frequency == 0:
                    logging.info("{0}/{1}".format(i,work_count))
                ata_piece[starti-start:stopi-start,:] = np.dot(slicei.T,slice)
        if do_both:
            ata_piece_python = ata_piece
            ata_piece = np.zeros((a.sid_count-start,stop-start),order='C')
    if not force_python_only or do_both:
        try:
            retval = mmultfile_atax(a_filename.encode('ascii'),a.offset,a.iid_count,a.sid_count,
                            work_index,work_count,
                            ata_piece,
                            num_threads = get_num_threads(),
                            log_frequency=log_frequency)
            assert retval==0
        except SystemError as system_error:
            raise system_error.__cause__

    if do_both:
        if not np.abs(ata_piece_python-ata_piece).max() > 1e-13:
           raise AssertionError("Expect Python and C++ to get the same mmultfile_atax answer")

    if log_frequency > 0:
        logging.info("ata_piece {0} of {1}: clocktime {2}".format(work_index, work_count,format_delta(time.time()-t0_gtg)))
    return ata_piece

def mmultfile_b_less_aatb(a_snp_mem_map, b, log_frequency=-1, force_python_only=False):
    if force_python_only:
        aTb = np.zeros((a_snp_mem_map.sid_count,b.shape[1])) #b can be destroyed. Is everything is in best order, i.e. F vs C
        aaTb = b.copy()
        b_mem = np.array(b,order="F") #!!! if we want this in "F" how about just creating that way instead of copying it
        with open(a_filename,"rb") as U_fp:
            U_fp.seek(a_snp_mem_map.offset)
            for i in range(a_snp_mem_map.sid_count):
                a_mem = np.fromfile(U_fp, dtype=np.float64, count=a_snp_mem_map.iid_count)
                if log_frequency > 0 and i%log_frequency == 0:
                    logging.info("{0}/{1}".format(i,a_snp_mem_map.sid_count))
                aTb[i,:] = np.dot(a_mem,b_mem)
                aaTb -= np.dot(a_mem.reshape(-1,1),aTb[i:i+1,:])
        return aTb, aaTb
    else:
        b1 = np.array(b,order="F")
        aTb = np.zeros((a_snp_mem_map.sid_count,b.shape[1]))
        aaTb = np.array(b1,order="F")
        try:
            mmultfile_b_less_aatbx(
                            a_snp_mem_map.filename.encode('ascii'),
                            a_snp_mem_map.offset,
                            a_snp_mem_map.iid_count, #row count
                            a_snp_mem_map.sid_count, #col count
                            b.shape[1], #col count
                            b1,   #B copy 1 in "F" order
                            aaTb, #B copy 2 in "F" order
                            aTb, # result
                            num_threads = get_num_threads(),
                            log_frequency=log_frequency,
                            )
        except SystemError as system_error:
            raise system_error.__cause__
        return aTb, aaTb

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_postsvd = True
    test_ata = False
    assert test_postsvd+test_ata <= 1, "Can test at most one thing"


    do_fast = False
    do_medium = False
    do_mid16 = True
    assert do_fast+do_medium <=1, "Can test as most one speed (aka size)"


    if test_postsvd:
        if do_fast:
            filename = r"D:\deldir\test\_Storage\very_small_0.8\G0_data.memmap"
        elif do_medium:
            filename = r"D:\deldir\test\_Storage\mid_0.8\G0_data.memmap"
        elif do_mid16:
            filename = r"D:\deldir\test\_Storage\mid_1.6\G0_data.memmap"
        else:
            filename = r"D:\deldir\scratch\escience\carlk\cachebio\genetics\onemil\fc\bigsyn0\U1.memmap"
        

        do_original = False
        force_python_only = False
        memory_factor = 1
        log_frequency = 100

        G0_memmap = SnpMemMap(filename)
        idx = G0_memmap.pos[:,0]!=2
        inonzero = np.array([True]*sum(idx))
        inonzero[0] = False
        logging.info("Generating random {0}x{0}".format(sum(idx)))
        if sum(idx) < 10000:
            np.random.seed(0)
            SVinv3 = np.random.random((sum(idx),sum(idx)))
        else:
            SVinv3 = np.zeros((sum(idx),sum(idx)))
            np.fill_diagonal(SVinv3,val=1)
        logging.info("Done Generating random")
        local_fn_U = r"d:\deldir\local_fn_U.memmap"
        t0 = time.time()
        U_memmap = post_svd(local_fn_U, G0_memmap, idx, SVinv3, inonzero, memory_factor, runner, do_original=do_original,force_python_only=force_python_only,log_frequency=log_frequency)
        print(U_memmap.val)
        logging.info("clocktime {0}".format(format_delta(time.time()-t0)))
        print("done")
        


    elif test_ata:
        if do_fast:
            filename = r"D:\deldir\test\_Storage\very_small_0.8\G0_data.memmap"
        else:
            filename = r"D:\deldir\scratch\escience\carlk\cachebio\genetics\onemil\fc\bigsyn0\U1.memmap"

        work_count = 3
        force_python_only = False

        a = SnpMemMap(filename)
        def memmap_lambda():
            return a
        result = []
        def writer(kernel_data):
            result.append(kernel_data)


        mmultfile_ata(memmap_lambda,writer,a.sid,work_count,name=None,runner=Local(),force_python_only=force_python_only)
        result = result[0]
        print(result.val)
        logging.info("Done")

    else:
        if do_fast:
            u1_filename = r"D:\deldir\scratch\escience\carlk\cachebio\genetics\onemil\fcvery_small_5\U1.memmap"
            test_snps_count = 100
            log_frequency=100
        else:
            u1_filename = r"D:\deldir\scratch\escience\carlk\cachebio\genetics\onemil\fc\bigsyn0\U1.memmap"
            test_snps_count = 1000
            log_frequency=10
        a = SnpMemMap(u1_filename)
        np.random.seed(0)
        logging.info("Generating random {0}x{1}".format(a.iid_count,a.sid_count))
        x = np.random.random((a.iid_count,test_snps_count))
        logging.info("Done Generating random")
        logging.info("Starting C++ run")
        cc1, cc2 = mmultfile_b_less_aatb(u1_filename,x,log_frequency=log_frequency,force_python_only=False)
        logging.info("Starting python run")
        py1, py2 = mmultfile_b_less_aatb(u1_filename,x,log_frequency=log_frequency,force_python_only=True)
        np.testing.assert_array_almost_equal(py1,cc1, decimal=3)
        np.testing.assert_array_almost_equal(py2,cc2, decimal=3)
        logging.info("Done")

