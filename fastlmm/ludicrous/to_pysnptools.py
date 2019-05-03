import logging
import os
import numpy as np
from pysnptools.pstreader import PstReader
from pysnptools.snpreader import Bed, Pheno, SnpData, SnpReader, SnpNpz

class PstMemMap(PstReader):


    def __init__(self, filename):
        '''
        filename    : string of the name of the memory mapped file.
        '''
        #!!!there is also a NPZ file the way that bed has multiple files
        #!!!should/could they be one file using memmaps offset feature?

        super(PstMemMap, self).__init__()
        self._ran_once = False

        self._filename = filename

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self._filename)

    @property
    def row(self):
        self.run_once()
        return self._row


    @property
    def col(self):
        self.run_once()
        return self._col

    @property
    def row_property(self):
        self.run_once()
        return self._row_property

    @property
    def col_property(self):
        self.run_once()
        return self._col_property

    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True

        npzfile = SnpReader._name_of_other_file(self._filename,"dat","npz")
        logging.debug("np.load('{0}')".format(npzfile))
        with np.load(npzfile) as data: #!! similar code in epistasis
            self._row = data['row']
            self._col = data['col']
            if np.array_equal(self._row, self._col): #If it's square, mark it so by making the col and row the same object
                self._col = self._row
            self._row_property = data['row_property']
            self._col_property = data['col_property']
            self._dtype = data['dtype'][0]
            self._order = data['order'][0]

        return self

    def copyinputs(self, copier):
        # doesn't need to self.run_once()
        copier.input(self._filename)
        npzfile = SnpReader._name_of_other_file(self._filename,"dat","npz")
        copier.input(npzfile)

    # Most _read's support only indexlists or None, but this one supports Slices, too.
    _read_accepts_slices = True
    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        assert view_ok, "Expect view_ok to be True" #!!! good assert?
        self.run_once()
        mode = "r" if view_ok else "w+"
        logging.debug("val = np.memmap('{0}', dtype={1}, mode={2}, order={3}, shape=({4},{5}))".format(self._filename,self._dtype,mode,self._order,self.row_count,self.col_count))
        val = np.memmap(self._filename, dtype=self._dtype, mode=mode, order=self._order, shape=(self.row_count,self.col_count))
        val, _ = self._apply_sparray_or_slice_to_val(val, row_index_or_none, col_index_or_none, self._order, self._dtype, force_python_only) #!!! must confirm that this doesn't copy of view_ok
        return val

    @staticmethod
    def write(filename, pstdata):
        npzfile = SnpReader._name_of_other_file(filename,"dat","npz")
        if pstdata.val.flags['F_CONTIGUOUS']:
            order = "F"
        elif pstdata.val.flags['C_CONTIGUOUS']:
            order = "C"
        else:
            raise Exception("Don't know order of PstData's value")

        np.savez(npzfile, row=pstdata.row, col=pstdata.col, row_property=pstdata.row_property, col_property=pstdata.col_property,dtype=np.array([pstdata.val.dtype]),order=np.array([order]))
        if isinstance(pstdata.val,np.memmap):
            pstdata.val.flush()
        else:
            val = np.memmap(filename, dtype=pstdata.val.dtype, mode="w+", order=order, shape=(pstdata.row_count,pstdata.col_count))
            val[:,:] = pstdata.val
            val.flush()
        logging.debug("Done writing " + filename)

        return SnpMemMap(filename) #!!! shouldn't all writers in pysnpsdata return their reader


class SnpMemMap(PstMemMap,SnpReader):

    def __init__(self, *args, **kwargs):
        super(SnpMemMap, self).__init__(*args, **kwargs)

    @staticmethod
    def write(filename, snpdata):
        return PstMemMap.write(filename,snpdata)

    @staticmethod #!!!PstMemMap should have something like this, too
    def snp_data(iid,sid,filename,pos=None,order="F",dtype=np.float64):
        shape = (len(iid),len(sid))
        logging.info("About to start allocating memmap '{0}'".format(filename))
        fp = np.memmap(filename, dtype=dtype, mode="w+", order=order, shape=shape)
        logging.info("Finished allocating memmap '{0}'. Size is {1}".format(filename,os.path.getsize(filename)))
        result = SnpData(iid=iid,sid=sid,val=fp,pos=pos,name="np.memmap('{0}')".format(filename))
        return result


class _MergeRows(PstReader): #!!!move to PySnptools
    def __init__(self,reader_list,cache_file=None,skip_check=False):
        super(_MergeRows, self).__init__()
        assert len(reader_list) > 0, "Expect at least one reader"
        self.skip_check = skip_check

        self.reader_list = reader_list
        self._repr_string = "_MergeRows({0})".format(",".join([str(s) for s in reader_list]))

        if cache_file is not None:
            if not os.path.exists(cache_file):
                self._run_once()
                np.savez(cache_file, _row=self._row, _row_property=self._row_property, _row_count_list=self._row_count_list)
            else:
                with np.load(cache_file) as data:
                    self._row = data['_row']
                    self._row_property = data['_row_property']
                    self._row_count_list = data['_row_count_list']
                    self._has_run_once = True


    def _run_once(self):
        if hasattr(self,'_has_run_once'):
            return
        self._has_run_once = True
        #Check that all iids are distinct and that all sids and pos are the same and in the same order

        row_list = []
        row_property_list = []
        row_set = set()
        col = self.reader_list[0].col
        col_property = self.reader_list[0].col_property
        for reader_index,reader in enumerate(self.reader_list):
            if reader_index % 10 == 0: logging.info("_MergeRows looking at reader #{0}: {1}".format(reader_index,reader))
            if not self.skip_check:
                assert np.array_equal(col,reader.col), "Expect columns to be the same across all files"
                np.testing.assert_equal(col_property,reader.col_property) #"Expect column_property to be the same across all files"
                size_before = len(row_set)
                row_set.update((tuple(item) for item in reader.row))
                assert len(row_set) == size_before + reader.row_count, "Expect rows to be distinct in all files"
            row_list.append(reader.row)
            row_property_list.append(reader.row_property)
        self._row = np.concatenate(row_list)
        self._row_property = np.concatenate(row_property_list)
        self._row_count_list = [len(row) for row in row_list]

    @property
    def row(self):
        self._run_once()
        return self._row

    @property
    def col(self):
        self._run_once()
        return self.reader_list[0].col

    @property
    def row_property(self):
        self._run_once()
        return self._row_property

    @property
    def col_property(self):
        self._run_once()
        return self.reader_list[0].col_property

    def __repr__(self): 
        #Don't need _run_once because based only on initial info
        return self._repr_string

    def copyinputs(self, copier):
        self._run_once()
        for reader in self.reader_list:
            copier.input(reader)

    def _create_reader_and_iid_index_list(self,iid_index):
        result = []
        start = 0
        for reader_index in xrange(len(self.reader_list)):
            stop = start + self._row_count_list[reader_index]
            is_here = (iid_index >= start) * (iid_index < stop)
            if any(is_here):
                iid_index_rel = iid_index[is_here]-start
                result.append((reader_index,is_here,iid_index_rel))
            start = stop
        return result

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        #!!!tests to do: no iid's
        #!!!tests to do: no sid's
        #!!!test to do: from file 1, file2, and then file1 again

        iid_index = iid_index_or_none if iid_index_or_none is not None else np.arange(self.iid_count) #!!!might want to special case reading all
        sid_index_or_none_count = self.sid_count if sid_index_or_none is None else len(sid_index_or_none)
        reader_and_iid_index_list = self._create_reader_and_iid_index_list(iid_index)

        if len(reader_and_iid_index_list) == 0:
            return self.reader_list[0]._read(iid_index,sid_index_or_none,order,dtype,force_python_only, view_ok)
        elif len(reader_and_iid_index_list) == 1:
            reader_index,iid_index_in,iid_index_rel = reader_and_iid_index_list[0]
            reader = self.reader_list[reader_index]
            return reader._read(iid_index_rel,sid_index_or_none,order,dtype,force_python_only, view_ok)
        else:
            logging.info("Starting read from {0} subreaders".format(len(reader_and_iid_index_list)))
            if order == 'A' or order is None:
                order = 'F'
            val = np.empty((len(iid_index),sid_index_or_none_count),dtype=dtype,order=order)
            for reader_index,is_here,iid_index_rel in reader_and_iid_index_list:
                reader = self.reader_list[reader_index]
                if reader_index % 1 == 0: logging.info("Reading from #{0}: {1}".format(reader_index,reader))
                val[is_here,:] = reader._read(iid_index_rel,sid_index_or_none,order,dtype,force_python_only, view_ok=True)
            logging.info("Ended read from {0} subreaders".format(len(reader_and_iid_index_list)))
            return val

class _MergeIIDs(_MergeRows,SnpReader): #!!! move to PySnptools
    def __init__(self, *args, **kwargs):
        super(_MergeIIDs, self).__init__(*args, **kwargs)

#!!! would be be better to make a Transpose class that could term _mergerows into mergecols? Be sure special Bed code is still there.
class _MergeCols(PstReader): #!!! move to PySnptools
    def __init__(self,reader_list,cache_file=None,skip_check=False):
        super(_MergeCols, self).__init__()
        assert len(reader_list) > 0, "Expect at least one reader"
        self.skip_check = skip_check

        self.reader_list = reader_list
        self._repr_string = "_MergeCols({0})".format(",".join([str(s) for s in reader_list]))

        if cache_file is not None:
            #!!!add warning if cache_file doesn't end with .npz
            if not os.path.exists(cache_file):
                self._run_once()
                np.savez(cache_file, _row=self._row, _row_property=self._row_property, _col=self._col, _col_property=self._col_property,sid_count_list=self.sid_count_list)
            else:
                with np.load(cache_file) as data:
                    self._col = data['_col']
                    self._col_property = data['_col_property']
                    self.sid_count_list = data['sid_count_list']
                    assert ('_row' in data) == ('_row_property' in data)
                    self._row = data['_row']
                    self._row_property = data['_row_property']
                self._has_run_once = True



    def _run_once(self):
        if hasattr(self,'_has_run_once'):
            return
        self._has_run_once = True
        #Check that all iids are distinct and that all sids and pos are the same and in the same order

        if self.skip_check and all(isinstance(reader,Bed) for reader in self.reader_list): #Special code if all Bed readers
            l = [SnpReader._read_map_or_bim(reader.filename,remove_suffix="bed", add_suffix="bim") for reader in self.reader_list]
            self.sid_count_list = np.array([len(ll[0]) for ll in l])
            self._row = self.reader_list[0].row
            self._row_property = self.reader_list[0].row_property
            self._col = np.concatenate([ll[0] for ll in l])
            self._col_property = np.concatenate([ll[1] for ll in l])
        else:
            col_list = []
            col_property_list = []
            col_set = set()
            self.sid_count_list = []
            self._row = self.reader_list[0].row
            self._row_property = self.reader_list[0].row_property
            for reader_index,reader in enumerate(self.reader_list):
                if reader_index % 10 == 0: logging.info("_MergeCols looking at reader #{0}: {1}".format(reader_index,reader))
                if not self.skip_check:
                    assert np.array_equal(self._row,reader.row), "Expect rows to be the same across all files"
                    np.testing.assert_equal(self._row_property,reader.row_property) #"Expect column_property to be the same across all files"
                    size_before = len(col_set)
                    col_set.update((tuple(item) for item in reader.col))
                    assert len(col_set) == size_before + reader.col_count, "Expect cols to be distinct in all files"
                col_list.append(reader.col)
                col_property_list.append(reader.col_property)
                self.sid_count_list.append(reader.sid_count)
            self._col = np.concatenate(col_list)
            self._col_property = np.concatenate(col_property_list)
            self.sid_count_list = np.array(self.sid_count_list)

    @property
    def col(self):
        self._run_once()
        return self._col

    @property
    def row(self):
        self._run_once()
        return self._row

    @property
    def col_property(self):
        self._run_once()
        return self._col_property

    @property
    def row_property(self):
        self._run_once()
        return self._row_property

    def __repr__(self): 
        #Don't need _run_once because based only on initial info
        return self._repr_string

    def copyinputs(self, copier):
        self._run_once()
        for reader in self.reader_list:
            copier.input(reader)

    def _create_reader_and_sid_index_list(self,sid_index):
        result = []
        start = 0
        for reader_index in xrange(len(self.reader_list)):
            stop = start + self.sid_count_list[reader_index] #!!! shouldn't this be col_count (and check _mergerows, too)
            is_here = (sid_index >= start) * (sid_index < stop)
            if any(is_here):
                sid_index_rel = sid_index[is_here]-start
                result.append((reader_index,is_here,sid_index_rel))
            start = stop
        return result

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        #!!!tests to do: no iid's
        #!!!tests to do: no sid's
        #!!!test to do: from file 1, file2, and then file1 again

        sid_index = sid_index_or_none if sid_index_or_none is not None else np.arange(self.sid_count) #!!!might want to special case reading all
        iid_index_or_none_count = self.iid_count if iid_index_or_none is None else len(iid_index_or_none)

        #Create a list of (reader,sid_index)
        reader_and_sid_index_list = self._create_reader_and_sid_index_list(sid_index)
        if len(reader_and_sid_index_list) == 0:
            return self.reader_list[0]._read(iid_index_or_none,sid_index,order,dtype,force_python_only, view_ok)
        elif len(reader_and_sid_index_list) == 1:
            reader_index,sid_index_in,sid_index_rel = reader_and_sid_index_list[0]
            reader = self.reader_list[reader_index]
            return reader._read(iid_index_or_none,sid_index_rel,order,dtype,force_python_only, view_ok)
        else:
            logging.info("Starting read from {0} subreaders".format(len(reader_and_sid_index_list)))
            if order == 'A' or order is None:
                order = 'F'
            val = np.empty((iid_index_or_none_count,len(sid_index)),dtype=dtype,order=order)
            for reader_index,is_here,sid_index_rel in reader_and_sid_index_list:
                reader = self.reader_list[reader_index]
                if reader_index % 1 == 0: logging.info("Reading from #{0}: {1}".format(reader_index,reader))
                val[:,is_here] = reader._read(iid_index_or_none,sid_index_rel,order,dtype,force_python_only, view_ok=True)
            logging.info("Ended read from {0} subreaders".format(len(reader_and_sid_index_list)))
            return val

class _MergeSIDs(_MergeCols,SnpReader): #!!! move to PySnptools
    def __init__(self,  *args, **kwargs):
        super(_MergeSIDs, self).__init__(*args, **kwargs)
