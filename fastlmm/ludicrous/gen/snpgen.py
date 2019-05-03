import numpy as np
from fastlmm.ludicrous.gen.generate import get_iid, get_sid, get_pos, get_val2
from pysnptools.snpreader import SnpReader
import pysnptools.util as pstutil
import logging
import os
from pysnptools.pstreader import PstData

class SnpGen(SnpReader):

    def __init__(self, seed, iid_count, sid_count, chrom_count=22, sid_batch_size=1000, cache_file=None):
        self._ran_once = False
        self.cache_file = cache_file

        self.seed = seed
        self._iid_count = iid_count
        self._sid_count = sid_count
        self.chrom_count = chrom_count
        self.sid_batch_size = sid_batch_size


        if cache_file is not None:
            if not os.path.exists(cache_file):
                pstutil.create_directory_if_necessary(cache_file)
                self.run_once()
                np.savez(cache_file, _row=self._row, _col=self._col, _col_property=self._col_property)
            else:
                with np.load(cache_file) as data:
                    self._row = data['_row']
                    assert len(self._row) == iid_count, "The iid in the cache file has a different length than iid_count"
                    self._col = data['_col']
                    assert len(self._col) == sid_count, "The sid in the cache file has a different length than sid_count"
                    self._col_property = data['_col_property']
                    assert len(self._col_property) == sid_count, "The pos in the cache file has a different length than sid_count"
                    self._ran_once = True


    def __repr__(self): 
        return "{0}(seed={1},iid_count={2},sid_count={3},chrom_count={4},sid_batch_size={5},cache_file={6})".format(self.__class__.__name__,self.seed,self._iid_count,self._sid_count,self.chrom_count,self.sid_batch_size,self.cache_file)

    @property
    def row(self):
        self.run_once()
        return self._row

    @property
    def col(self):
        self.run_once()
        return self._col

    @property
    def col_property(self):
        self.run_once()
        return self._col_property

    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True
        self._row = np.array(get_iid(self._iid_count))
        self._col = np.array(get_sid(0, self._sid_count))
        self._col_property = get_pos(0,self._sid_count,self._sid_count,chrom_count=self.chrom_count)

    def copyinputs(self, copier):
        self.run_once()
        copier.input(self.cache_file)

    # Most _read's support only indexlists or None, but this one supports Slices, too.
    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        self.run_once()


        row_index_count = len(row_index_or_none) if row_index_or_none is not None else self._iid_count # turn to a count of the index positions e.g. all of them
        col_index = col_index_or_none if col_index_or_none is not None else np.arange(self._sid_count) # turn to an array of index positions, e.g. 0,1,200,2200,10
        batch_index = col_index // self.sid_batch_size  #find the batch index of each index position, e.g. 0,0,0,2,0
        val = np.empty((row_index_count,len(col_index))) #allocate memory for result
        list_batch_index = list(set(batch_index))
        for i in list_batch_index:  #for each distinct batch index, generate snps
            logging.info("working on snpgen batch {0} of {1}".format(i,len(list_batch_index)))
            start = i*self.sid_batch_size  #e.g. 0 (then 2000)
            stop = start + self.sid_batch_size #e.g. 1000, then 3000
            batch_val = get_val2(self.seed,self._iid_count,start,stop) # generate whole batch
            a = (batch_index==i) #e.g. [True,True,True,False,True], then [False,False,False,True,False]
            b = col_index[a]-start #e.g.  0,1,200,10, then 200
            val[:,a] = batch_val[:,b] if row_index_or_none is None else pstutil.sub_matrix(batch_val, row_index_or_none, b)

        return val

if __name__ == '__main__':
    seed = 0
    snpgen = SnpGen(seed=seed,iid_count=1000,sid_count=int(10e6))
    snpdata = snpgen[:,[0,1,200,2200,10]].read()
    snpdata2 = snpgen[:,[0,1,200,2200,10]].read()
    np.testing.assert_equal(snpdata.val,snpdata2.val)
    snpdata3 = snpgen[::10,[0,1,200,2200,10]].read()
    np.testing.assert_equal(snpdata3.val,snpdata2.val[::10,:])
    np.random.seed(seed)
    kernel_snps = np.arange(50000)# np.array(list(set(np.random.randint(0,int(10e6),size=60000))))[:50000]
    #kernel_snps.sort()

    snpgen3 = SnpGen(seed=seed,iid_count=int(1e6),sid_count=int(10e6))
    batch3 = 1000
    for start in xrange(0,len(kernel_snps),batch3):
        kernel_batch = kernel_snps[start:start+batch3]
        snpdata3 = snpgen3[:,kernel_batch].read()
    print snpdata.val