import matplotlib
import pylab
#matplotlib.use('TkAgg',warn=False) #This lets it work even on machines without graphics displays
import os
import logging
logging.basicConfig(level=logging.INFO)
from fastlmm.util.runner import Local, Hadoop, Hadoop2, HPC, LocalMultiProc, LocalInParts
from pysnptools.snpreader import Bed, SnpData
import numpy as np
import pysnptools.util as pstutil

def get_val(x_sample,dist,iid_count,sid_start,sid_stop,seed):
    missing_rate = .218
    sid_batch_size_a = sid_stop-sid_start
    sid_batch_size_b = sid_stop-sid_start

    np.random.seed(seed+sid_start)
    val_a = np.empty((iid_count,sid_batch_size_a),order="F")
    sid_so_far = 0
    
    while sid_so_far < sid_batch_size_a:
        logging.debug(sid_so_far)
        sid_index_to_freq = np.random.choice(x_sample,size=sid_batch_size_b,replace=True,p=dist) #For each sid, pick its minor allele freq
        val_b = (np.random.rand(iid_count,2,sid_batch_size_b) < sid_index_to_freq).sum(axis=1).astype(float) #Sample each allele and then sum the minor alleles #!!!slowest part
        missing = np.random.rand(iid_count,sid_batch_size_b)<missing_rate
        val_b[missing] = np.nan

        sid_index_to_minor_allele_count = np.nansum(val_b,axis=0) #Find the # of minor alleles for each allele. By design this will be mostly 0's
    
        #Keep only the sids with a minor allele count between 1 and iid_count.
        legal = (sid_index_to_minor_allele_count != 0) * (sid_index_to_minor_allele_count <= iid_count)
        legal_count = legal.sum()
        val_a[:,sid_so_far:sid_so_far+legal_count]=val_b[:,legal][:,:sid_batch_size_a-sid_so_far] #!!! takes about 1/3 of the time. Faster if both were order="F"?
        sid_so_far += legal_count
    
    return val_a

def get_dist(iid_count):
    w = np.array([-0.6482249 , -8.49790398])
    
    def dist_fit(x):
        log_y = w[0] * np.log(x) + w[1]
        return np.exp(log_y)
    
    x_sample = np.logspace(np.log10(.1/iid_count),np.log10(.5),100,base=10) #discretize from 1e-7 (assuming iid_count=1e6) to .5, logarithmically
    y_sample = np.array([dist_fit(x) for x in x_sample])                    #Find the relative weight of each point
    dist = y_sample/y_sample.sum()
    return x_sample, dist


def pick_reference_alternate_alleles(sid_batch_size):
    reference_allele = np.random.choice(['A','C','T','G'],size=sid_batch_size,replace=True)
    alternate_allele = reference_allele.copy()
    while True: #Loop until each sid's minor allele is different from its major allele
        same = (reference_allele == alternate_allele)
        same_sum = same.sum()
        if same_sum == 0:
            return reference_allele, alternate_allele # all different, so leave loop
        logging.debug(same_sum)
        alternate_allele[same] = np.random.choice(['A','C','T','G'],size=same_sum,replace=True)
    

def get_iid(iid_count):
    iid = [('0','iid_{0}'.format(i)) for i in xrange(iid_count)]
    return iid

def get_sid(sid_start, sid_stop):
    sid = ["sid_{0}".format(i) for i in xrange(sid_start,sid_stop)]
    return sid

_chrom_size = np.array([263,255,214,203,194,183,171,155,145,144,144,143,114,109,106,98,92,85,67,72,50,56],dtype=long)*int(1e6) #The approximate size of human chromosomes in base pairs
_chrom_total = _chrom_size.sum()

def get_pos(sid_start, sid_stop, sid_count, chrom_count=22):
    chrom_size = _chrom_size[:chrom_count]
    chrom_total = chrom_size.sum()
    step = chrom_total // sid_count

    pos = np.empty(((sid_stop-sid_start),3))
    for sid_index in xrange(sid_start,sid_stop):
        full_index = sid_index * step
        chrom_index, offset_index = find_chrom(full_index, chrom_size)
        pos[sid_index-sid_start,:] = [chrom_index+1,0,offset_index+1]
    return pos

def find_chrom(full_index,chrom_size):
    for chrom_index, chrom_one in enumerate(chrom_size):
        if full_index < chrom_one:
            return chrom_index, full_index
        full_index -= chrom_one
    assert False


def get_val2(seed, iid_count, sid_start, sid_stop):
    x_sample, dist = get_dist(iid_count) # The discrete distribution of minor allele frequencies (based on curve fitting to real data)
    val = get_val(x_sample,dist,iid_count,sid_start,sid_stop, seed)
    return val

def data_gen(iid_count, sid_start, sid_stop, sid_count, seed=0):
    logging.info("data_gen(iid_count={0},sid_start={1},sid_stop={2},sid_count={3},seed={4})".format(iid_count,sid_start,sid_stop,sid_stop,seed))
    val = get_val2(seed, iid_count, sid_start, sid_stop)
    sid_batch_size = sid_stop-sid_start
    iid = get_iid(iid_count)
    val = np.zeros((iid_count,sid_batch_size),order="F")
    pos = get_pos(sid_start, sid_stop, sid_count)
    sid = get_sid(sid_start, sid_stop)
    snpdata = SnpData(iid=iid,sid=sid,val=val,pos=pos,name="random gen")
    reference_allele, alternate_allele = pick_reference_alternate_alleles(sid_batch_size)

    return snpdata, reference_allele, alternate_allele

def chrom_number_to_letter(chrom_number):
    assert 1 <= chrom_number and chrom_number <= 25, "Expect chrom number to be between 1 and 25 (inclusive)"
    if chrom_number <= 22:
        return str(chrom_number)
    elif chrom_number == 23:
        return 'X'
    elif chrom_number == 24:
        return 'Y'
    elif chrom_number == 25:
        return 'M'
    else:
        raise Exception("unexpected chrom_number '{0}'".format(chrom_number))

def double_to_val(d, ref_char, alt_char):
    assert ref_char != alt_char, "real assert"
    if d != d: #NaN mising
        return ''
    elif d == 0:
        return ref_char+ref_char
    elif d == 2:
        return alt_char+alt_char
    elif d == 1:
        if ref_char < alt_char:
            return ref_char+alt_char
        else:
            return alt_char+ref_char
    else:
        raise Exception("unexpected double '{0}'".format(d))

def data_write(dir_name,snpdata, reference_allele, alternate_allele):
    filename = "{0}/seed{1}.sid{2}-{3}.txt".format(dir_name,seed,sid_start,sid_start+snpdata.sid_count)
    logging.info(filename)

    pstutil.create_directory_if_necessary(filename)
    with open(filename+".temp","w") as f_out:
        f_out.write("\t".join(["sid","chrom","gen_dist","bp_pos","ref","alt","iid","value"]))
        f_out.write("\n")
    
        for sid_index2 in xrange(snpdata.sid_count):
            ref_char = reference_allele[sid_index2]
            alt_char = alternate_allele[sid_index2]
    
            sid_line = "\t".join([snpdata.sid[sid_index2],                              # sid (string)
                                  chrom_number_to_letter(snpdata.pos[sid_index2,0]),    # chromosome letter (char)
                                  str(snpdata.pos[sid_index2,1]),                       # genetic distance (double)
                                  str(long(snpdata.pos[sid_index2,2])),                 # base-pair position (long)
                                  ref_char,                                             # reference allele (char)
                                  alt_char                                              # alternate_allele (char)
                                 ])
            for iid_index in xrange(snpdata.iid_count):
                line = "\t".join([sid_line,
                                  snpdata.iid[iid_index,1],                             # iid
                                  double_to_val(snpdata.val[iid_index,sid_index2],      # value (string or null)
                                               ref_char,alt_char)
                                 ])
                f_out.write(line)
                f_out.write("\n")
    os.rename(filename+".temp",filename)

if __name__ == '__main__':
    dir_name = "output"
    seed = 0

    # Number of rows and colunmns
    iid_count = 1000 # int(1e6)
    sid_count = int(10e6)

    sid_batch_size = 1000

    for sid_start in xrange(0,sid_count,sid_batch_size):
        sid_stop = sid_start+sid_batch_size
        snpdata, reference_allele, alternate_allele = data_gen(iid_count, sid_start, sid_stop, sid_count, seed=seed)
        data_write(dir_name, snpdata, reference_allele, alternate_allele)

    logging.info("done")
