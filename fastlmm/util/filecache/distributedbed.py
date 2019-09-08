import os
import numpy as np
import logging
import unittest
from pysnptools.snpreader import _MergeSIDs
from pysnptools.snpreader import SnpReader, Bed
from pysnptools.pstreader import PstReader
from pysnptools.util import _multiopen
from pysnptools.snpreader import _snps_fixup
from pysnptools.util import log_in_place
from pysnptools.util.mapreduce1 import map_reduce


#!!!cmk update everything and confirm testing

class DistributedBed(SnpReader): #!!!cmk is this tested anywhere
    #!!!cmk update
    '''
    A class that implements the :class:`SnpReader` interface. It stores BED-like data in pieces on storage. When requested, it retrieves requested parts of the data. 

    **Constructor:**
        :Parameters: * **storage** (*FileCache*) -- The :class:`LocalCache` or other :class:`FileCache` where the data should or is stored.
    '''
    def __init__(self, storage):
        super(DistributedBed, self).__init__()

        self._ran_once = False

        self._storage = storage
        self._merge = None


    def __repr__(self): 
        return "{0}({1})".format(self.__class__.__name__,self._storage)

    @property
    def row(self):
        """*same as* :attr:`iid`
        """
        self._run_once()
        return self._merge.row

    @property
    def col(self):
        """*same as* :attr:`sid`
        """
        self._run_once()
        return self._merge.col

    @property
    def col_property(self):
        """*same as* :attr:`pos`
        """
        self._run_once()
        return self._merge.col_property

    def _run_once(self):
        if self._ran_once:
            return
        self._ran_once = True

        _metadatanpz = "metadata.npz"
        with self._storage.open_read(_metadatanpz) as handle_metadatanpz_file_name:
            #self._file_dict["metadatanpz"] = handle_metadatanpz
            _reader_name_listnpz = "reader_name_list.npz"
            with self._storage.open_read(_reader_name_listnpz) as handle_reader_name_listnpz_file_name:
                reader_name_list = np.load(handle_reader_name_listnpz_file_name)['reader_name_list']
                #self._file_dict["reader_name_listnpz"] = handle_reader_name_listnpz

                reader_list = [_Distributed1Bed(reader_name,self._storage) for reader_name in reader_name_list]

                self._merge = _MergeSIDs(reader_list,cache_file=handle_metadatanpz_file_name,skip_check=True)

                for reader in reader_list:
                    reader._row = self._merge.row
            

    #def __del__(self):
    #    for handle in self._file_dict.itervalues():
    #        handle.close()

    def copyinputs(self, copier):
        pass

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        self._run_once()
        return self._merge._read(iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok)

    @staticmethod
    def write(storage, snpreader, piece_per_chrom_count, updater=None, runner=None): #!!!cmk might want to set pieces_per_chrom such that it is a certain size
        '''
        Uploads from any BED-like data to cluster storage for efficient retrieval later.
        If some of the content already exists in storage, it skips uploading that part of the content. (To avoid this behavior,
        clear the storage.)

        Additional data formats can be supported by creating new readers that follow that :class:`.SnpReader` interface.

        :param storage: A :class:`.LocalCache` or other :class:`.FileCache` to upload to.
        :type storage: :class:`.FileCache`

        :param snpreader: A :class:`.Bed` or other :class:`.SnpReader` that with values of 0,1,2, or missing.
        :type snpreader: :class:`.SnpReader`

        :param piece_per_chrom_count: The number of pieces in which to store the data from each chromosome. Data is split across
            SNPs. If set to 100 and 22 chromosomes are uploaded, then data will be stored in 2200 pieces. Later, when data is requested
            only the pieces necessary for the request will be downloaded.
        :type piece_per_chrom_count: A number

        :param updater: A single argument function to write logging message to.
        :type updater: A lambda such as created by :func:`log_in_place`

        :param runner: a runner, optional: Tells how to run locally or multi-processor are good options.
            If not given, the function is run locally.
        :type runner: a runner.

        :rtype: DistributedBed

        '''
        from pysnptools.util import _progress_reporter

        count_A1 = True #Make all these's the same for reading and writing so that nothing will change.
        snpreader = _snps_fixup(snpreader, count_A1=count_A1)

        with _progress_reporter("DistributedBed.write", size=0, updater=updater) as updater2:
            chrom_set = list(set(snpreader.pos[:,0]))
            def mapper_closure(chrom):
                chrom_reader = snpreader[:,snpreader.pos[:,0]==chrom]
                def nested_closure(piece_per_chrom_index):
                    start = chrom_reader.sid_count * piece_per_chrom_index // piece_per_chrom_count
                    stop = chrom_reader.sid_count * (piece_per_chrom_index+1) // piece_per_chrom_count
                    piece_reader = chrom_reader[:,start:stop]
                    _piece_name_list = ["chrom{0}.piece{1}of{2}.{3}".format(int(chrom),piece_per_chrom_index,piece_per_chrom_count,suffix) for suffix in ['bim','fam','bed']]
                    exist_list = [storage.file_exists(_piece_name) for _piece_name in _piece_name_list]
                    if sum(exist_list) < 3: #If all three of the BIM/FAM/BED files are already there, then skip the upload, otherwise do the upload
                        for i in xrange(3): #If one or two of BIM/FAM/BED are there, remove them
                            if exist_list[i]:
                                storage.remove(_piece_name_list[i])
                        _Distributed1Bed.write(_piece_name_list[-1],storage,piece_reader.read(),count_A1=count_A1,updater=updater2)
                    return _piece_name_list[-1]
                return map_reduce(xrange(piece_per_chrom_count),
                    mapper=nested_closure,
                    )
            list_list_pair = map_reduce(chrom_set,
                nested = mapper_closure,
                runner=runner,
                )                

        reader_name_list = []
        reader_list = []
        for chrom_result in list_list_pair:
            for _piece_name in chrom_result:
                reader_name_list.append(_piece_name)
                reader_list.append(_Distributed1Bed(_piece_name,storage))
                

        _metadatanpz = "metadata.npz"
        with storage.open_write(_metadatanpz) as local_metadatanpz:
            _reader_name_listnpz = "reader_name_list.npz"
            with storage.open_write(_reader_name_listnpz) as local_reader_name_listnpz:
                reader_name_list = np.savez(local_reader_name_listnpz,reader_name_list=reader_name_list)
                if os.path.exists(local_metadatanpz):
                    os.remove(local_metadatanpz)
                _MergeSIDs(reader_list,cache_file=local_metadatanpz,skip_check=True)

        return DistributedBed(storage)

class _Distributed1Bed(SnpReader):
    '''
    An atomic set of bed/bim/fam files stored somewhere. Can answer metadata questions without downloading the *.bed file.
    But does download the whole *.bed file when any SNP value is requested.
    '''
    def __init__(self,path,storage):
        super(_Distributed1Bed, self).__init__()

        self._ran_once = False
        self._file_dict = {}

        self._storage = storage
        self.path = path
        self.local = None

    def __repr__(self): 
        return "{0}('{1}','{2}')".format(self.__class__.__name__,self.path,self._storage)


    @property
    def row(self):
        """*same as* :attr:`iid`
        """
        if not hasattr(self,"_row"):
            _fam = SnpReader._name_of_other_file(self.path,remove_suffix="bed", add_suffix="fam")
            local_fam = self._storage.open_read(_fam)
            self._row = SnpReader._read_fam(local_fam.__enter__(),remove_suffix="fam")
            self._file_dict["fam"] = local_fam
        return self._row

    @property
    def col(self):
        """*same as* :attr:`sid`
        """
        if not hasattr(self,"_col"):
            _bim = SnpReader._name_of_other_file(self.path,remove_suffix="bed", add_suffix="bim")
            local_bim = self._storage.open_read(_bim)
            self._col, self._col_property = SnpReader._read_map_or_bim(local_bim.__enter__(),remove_suffix="bim", add_suffix="bim")
            self._file_dict["bim"] = local_bim
        return self._col

    @property
    def col_property(self):
        """*same as* :attr:`pos`
        """
        if not hasattr(self,"_col"):
            self.col #get col info
        return self._col_property

    def _run_once(self):
        if self._ran_once:
            return
        self._ran_once = True
        self.row # get row info
        self.col # get col info

        _bed = SnpReader._name_of_other_file(self.path,remove_suffix="bed", add_suffix="bed")
        local_bed = self._storage.open_read(_bed)
        self.local = Bed(local_bed.__enter__(),count_A1=True,iid=self.row,sid=self.col,pos=self.col_property,skip_format_check=True)
        self._file_dict["bed"] = local_bed

    def __del__(self):
        for handle in self._file_dict.itervalues():
            handle.__exit__(None,None,None)
        self._file_dict = {}

    def copyinputs(self, copier):
        pass

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        self._run_once()
        return self.local._read(iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok)
    
    @staticmethod
    def write(path, storage, snpdata,count_A1=True,updater=None):
        file_list = [SnpReader._name_of_other_file(path,remove_suffix="bed", add_suffix=new_suffix) for new_suffix in ["bim","fam","bed"]] #'bed' should be last
        with _multiopen(lambda file_name:storage.open_write(file_name,updater=updater),file_list) as local_file_name_list:
            Bed.write(local_file_name_list[-1],snpdata,count_A1=count_A1)

        return _Distributed1Bed(path,storage)


    #!!!cmk get this working and running as part of main tests
class TestDistributedBedDocStrings(unittest.TestCase):
    def test_distributed_bed(self):
        import pysnptools.snpreader.snpdata
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/snpreader")
        result = doctest.testmod(pysnptools.snpreader.distributed_bed)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__
