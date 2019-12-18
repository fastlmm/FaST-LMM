from __future__ import absolute_import
import numpy as SP
import subprocess, sys, os.path
from itertools import *
from fastlmm.pyplink.snpset import *
from fastlmm.pyplink.altset_list import *
import pandas as pd
import logging
from six.moves import range

class Dat(object):
    '''
    This is a class that reads into memory from DAT/FAM/MAP files.
    '''


    def __init__(self,dat_filename):
        '''
        filename    : string of the name of the Dat file.
        '''

        self.dat_filename = dat_filename
        self._ran_once = False

    #!! similar code in fastlmm
    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.dat_filename)

    def names_of_other_files(self):
        return Dat.names_of_other_files_static(self.dat_filename)

    @property
    def snp_to_index(self):
        self.run_once()
        return self._snp_to_index

    @staticmethod
    def names_of_other_files_static(dat_filename):
        base_filename = (".").join(dat_filename.split(".")[:-1])
        famfile = base_filename + '.fam'
        mapfile = base_filename + '.map'
        return famfile, mapfile

    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True

        famfile, mapfile = self.names_of_other_files()

        #!!similar code in BED reader
        logging.info("Loading fam file {0}".format(famfile))
        self._original_iids = SP.loadtxt(famfile,dtype = 'str',usecols=(0,1),comments=None)

        #!!similar code in BED reader
        logging.info("Loading map file {0}".format(mapfile))
        self.bimfields = pd.read_csv(mapfile,delimiter = '\s',usecols = (0,1,2,3),header=None,index_col=False,engine='python')
        self.rs = SP.array(self.bimfields[1].tolist(),dtype='str')
        self.pos = self.bimfields[[0,2,3]].values
        self._snp_to_index = {}
        logging.info("indexing snps");
        for i in range(self.snp_count):
            snp = self.rs[i]
            if snp in self._snp_to_index : raise Exception("Expect snp to appear in bim file only once. ({0})".format(snp))
            self._snp_to_index[snp]=i

        #!!could change to just create/find an index to the file position of each row. Instead, reading all into memory
        datfields = pd.read_csv(self.dat_filename,delimiter = '\s',header=None,index_col=False)
        if not sp.array_equal(sp.array(datfields[0],dtype="string"), self.rs) : raise Exception("Expect snp list in map file to exactly match snp list in dat file")
        self.start_column = 3
        if len(self._original_iids) != datfields.shape[1]-self.start_column : raise Exception("Expect # iids in fam file to match dat file")
        self.datfields = datfields.T


        return self

    #def __del__(self):
    #    if self._filepointer != None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
    #        self._filepointer.close()

    def copyinputs(self, copier):
        # doesn't need to self.run_once() because creates name of all files itself
        famfile, mapfile = self.names_of_other_files()
        copier.input(self.dat_filename)
        copier.input(famfile)
        copier.input(mapfile)


    @property
    def snp_count(self):
        self.run_once()
        return len(self.bimfields);

    def read(self,snp_set = AllSnps(), order="F", dtype=SP.float64, force_python_only=False):
        '''
        Input: a snp_set. Choices include
            AllSnps() [the default],
            PositionRange(snpIndex,nSNPs)
            SnpAndSetName(groupname,snplist),

        Output dictionary:
        'rs'     : [S] array rs-numbers
        'pos'    : [S*3] array of positions [chromosome, genetic dist, basepair dist]
        'snps'   : [N*S] array of snp-data
        'iid'    : [N*2] array of family IDs and individual IDs
        '''
        self.run_once()
        snpset_withbed = snp_set.addbed(self)
        return self.read_with_specification(snpset_withbed, order=order, dtype=dtype, force_python_only=force_python_only)

    ##!! This property is ugly
    @property
    def ind_used(self):
        # doesn't need to self.run_once() because only uses original inputs
        return self._ind_used

    @ind_used.setter
    def ind_used(self, value):
        '''
        Tell the Dat reader to return data for only a subset (perhaps proper) of the individuals in a particular order
        e.g. 2,10,0 says to return data for three users: the user at index position 2, the user at index position 10, and the user at index position 0.
        '''
        # doesn't need to self.run_once() because only uses original inputs
        self._ind_used = value

    @property
    def original_iids(self):
        self.run_once()
        return self._original_iids

    def counts_and_indexes(self, snpset_withdat):
        #!!similar code in BED
        iid_count_in = len(self.original_iids)
        snp_count_in = self.snp_count
        if hasattr(self,'_ind_used'):
            iid_count_out = len(self.ind_used)
            iid_index_out = self.ind_used
        else:
            iid_count_out = iid_count_in
            iid_index_out = list(range(0,iid_count_in))
        snp_count_out = len(snpset_withdat)
        snp_index_out = list(snpset_withdat)  #make a copy, in case it's in some strange format, such as HDF5
        return iid_count_in, iid_count_out, iid_index_out, snp_count_in, snp_count_out, snp_index_out


    @staticmethod
    def read_with_specification(snpset_withdat, order="F", dtype=SP.float64, force_python_only=False):
        dat = snpset_withdat.bed #!!bed is a misnomer
        iid_count_in, iid_count_out, iid_index_out, snp_count_in, snp_count_out, snp_index_out = dat.counts_and_indexes(snpset_withdat)

        SNPs = SP.zeros((iid_count_out,snp_count_out),order=order, dtype=dtype)
        for SNPsIndex, bimIndex in enumerate(snp_index_out):
            row = sp.array(dat.datfields[bimIndex])[dat.start_column:,]
            SNPs[:,SNPsIndex] = row[iid_index_out]

        ret = {
                'rs'     :dat.rs[snp_index_out],
                'pos'    :dat.pos[snp_index_out,:],
                'snps'   :SNPs,
                'iid'    :dat.original_iids[iid_index_out,:]
                }
        return ret


    @staticmethod
    def write(snpMatrix, datfile):
        snpsarray = snpMatrix['snps']
        rslist = snpMatrix['rs']
        posarray = snpMatrix['pos']
        iidarray = snpMatrix['iid']

        famfile, mapfile = Dat.names_of_other_files_static(datfile)

        with open(famfile,"w") as fam_filepointer:
            for iid_row in iidarray:
                fam_filepointer.write("{0} {1} 0 0 0 0\n".format(iid_row[0],iid_row[1]))

        with open(mapfile,"w") as map_filepointer:
            for snp_index,rs in enumerate(rslist):
                posrow = posarray[snp_index]
                map_filepointer.write("{0}\t{1}\t{2}\t{3}\n".format(posrow[0], rs, posrow[1], posrow[2]))

        with open(datfile,"w") as dat:
            for snp_index,rs in enumerate(rslist):
                dat.write("{0}\tj\tn\t".format(rs)) #use "j" and "n" as the major and minor allele
                row = snpsarray[:,snp_index]
                dat.write("\t".join([str(i) for i in row]) + "\n")

#if __name__ == "__main__":
#    logging.basicConfig(level=logging.INFO)
#    import doctest
#    doctest.testmod()
