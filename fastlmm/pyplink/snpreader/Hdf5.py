from __future__ import absolute_import
from six.moves import range
try:
    import h5py
except:
    pass

import logging
import scipy as sp
from fastlmm.pyplink.snpset import *
from fastlmm.pyplink.altset_list import *

#!!document the format

class Hdf5(object):


    def __init__(self,filename, order = 'F',blocksize=5000):
        self._ran_once = False
        self.h5 = None

        ##!! copy relevent comments from Bed reader
        self.filename=filename
        self.order = order
        self.blocksize = blocksize

    def copyinputs(self, copier):
        copier.input(self.filename)

    @property
    def snp_to_index(self):
        self.run_once()
        return self._snp_to_index

    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True
        try:
            self.h5 = h5py.File(self.filename, "r")
        except IOError as e:
            raise IOError("Missing or unopenable file '{0}' -- Native error message: {1}".format(self.filename,e))

        
        self._original_iids = sp.array(sp.array(self.h5['iid']),dtype=str)
        self.rs = sp.array(sp.array(self.h5['rs']),dtype='str')
        self.pos = sp.array(self.h5['pos'])

        ## similar code in bed
        self._snp_to_index = {}
        logging.info("indexing snps");
        for i,snp in enumerate(self.rs):
            if snp in self._snp_to_index : raise Exception("Expect snp to appear in bim file only once. ({0})".format(snp))
            self._snp_to_index[snp]=i

        self.snpsInFile = self.h5['snps']

        if "SNP-major" not in self.snpsInFile.attrs: raise Exception("In Hdf5 the 'snps' matrix must have a Boolean 'SNP-major' attribute")
        self.is_snp_major = self.snpsInFile.attrs["SNP-major"]

        S_original = len(self.rs)
        N_original = len(self.original_iids)
        if self.is_snp_major:
            if not self.snpsInFile.shape == (S_original, N_original) : raise Exception("In Hdf5, snps matrix dimensions don't match those of 'rs' and 'iid'")
        else:
            if not self.snpsInFile.shape == (N_original, S_original) : raise Exception("In Hdf5, snps matrix dimensions don't match those of 'rs' and 'iid'")


    @property
    def snp_count(self):
        self.run_once()
        return len(self.rs);

    @property
    def original_iids(self):
        self.run_once()
        return self._original_iids

    #same code is in Bed. Could this be moved to an abstract class?
    def read(self,snp_set = AllSnps(), order="F", dtype=SP.float64, force_python_only=False):
        self.run_once()
        snpset_withbed = snp_set.addbed(self)
        return self.read_with_specification(snpset_withbed, order=order, dtype=dtype, force_python_only=force_python_only)

    @staticmethod
    #should move into utils
    def is_sorted_without_repeats(list):
        if len(list) < 2:
            return True
        for i in range(1,len(list)):
            if not list[i-1] < list[i]:
                return False
        return True
    

    def __del__(self):
        if self.h5 != None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
            self.h5.close()

    def read_direct(self, snps, selection=sp.s_[:,:]):

        if self.is_snp_major:
            selection = tuple(reversed(selection))

        if snps.flags["F_CONTIGUOUS"]:
            self.snpsInFile.read_direct(snps.T,selection)
        else:
            self.snpsInFile.read_direct(snps,selection)

    #!! much code the same as for Bed
    def create_block(self, blocksize, dtype, order):
        N_original = len(self.original_iids) #similar code else where -- make a method
        matches_order = self.is_snp_major == (order =="F") #similar code else where -- make a method
        opposite_order = "C" if order == "F" else "F"#similar code else where -- make a method
        if matches_order:
            return sp.empty([N_original,blocksize], dtype=dtype, order=order)
        else:
            return sp.empty([N_original,blocksize], dtype=dtype, order=opposite_order)

    def read_with_specification(self, snpset_with_snpreader, order="F", dtype=SP.float64, force_python_only=False): 
        self.run_once()

        order = order.upper()
        opposite_order = "C" if order == "F" else "F"

        snp_index_list = sp.array(list(snpset_with_snpreader)) # Is there a way to create an array from an iterator without putting it through a list first?
        S = len(snp_index_list)
        S_original = self.snp_count
        N_original = len(self.original_iids)

        # Check if snps and iids indexes are in order and in range
        snps_are_sorted = Hdf5.is_sorted_without_repeats(snp_index_list)
        if hasattr(self,'_ind_used'):
            iid_index_list = self._ind_used
            iid_is_sorted = Hdf5.is_sorted_without_repeats(iid_index_list)
        else:
            iid_index_list = sp.arange(N_original)
            iid_is_sorted = True

        N = len(iid_index_list)

        SNPs = sp.empty([N, S], dtype=dtype, order=order)

        matches_order = self.is_snp_major == (order =="F")
        is_simple = not force_python_only and iid_is_sorted and snps_are_sorted and matches_order #If 'is_simple' may be able to use a faster reader

        # case 1 - all snps & all ids requested
        if is_simple and S == S_original and N == N_original:
            self.read_direct(SNPs)

        # case 2 - some snps and all ids
        elif is_simple and N == N_original:
            self.read_direct(SNPs, sp.s_[:,snp_index_list])

        # case 3 all snps and some ids
        elif is_simple and S == S_original:
            self.read_direct(SNPs, sp.s_[iid_index_list,:])

        # case 4 some snps and some ids -- use blocks
        else:
            blocksize = min(self.blocksize, S)
            block = self.create_block(blocksize, dtype, order)

            if not snps_are_sorted:
                snp_index_index_list = sp.argsort(snp_index_list)
                snp_index_list_sorted = snp_index_list[snp_index_index_list]
            else:
                snp_index_index_list = sp.arange(S)
                snp_index_list_sorted = snp_index_list

            for start in range(0, S, blocksize):
                #print start
                end = min(start+blocksize,S)
                if end-start < blocksize:  #On the last loop, the buffer might be too big, so make it smaller
                    block = self.create_block(end-start, dtype, order)
                snp_index_list_forblock = snp_index_list_sorted[start:end]
                snp_index_index_list_forblock = snp_index_index_list[start:end]
                self.read_direct(block, sp.s_[:,snp_index_list_forblock])
                SNPs[:,snp_index_index_list_forblock] = block[iid_index_list,:]

        rs = self.rs[snp_index_list]
        pos = self.pos[snp_index_list,:]
        iids = sp.array(self.original_iids[iid_index_list],dtype="str") #Need to make another copy of to stop it from being converted to a list of 1-d string arrays

        has_right_order = (order=="C" and SNPs.flags["C_CONTIGUOUS"]) or (order=="F" and SNPs.flags["F_CONTIGUOUS"])
        #if SNPs.shape == (1, 1):
        assert(SNPs.shape == (N, S) and SNPs.dtype == dtype and has_right_order)

        ret = {
                'rs'     :rs,
                'pos'    :pos,
                'snps'   :SNPs,
                'iid'    :iids
                }
        return ret



    @property
    def ind_used(self):
        # doesn't need to self.run_once() because only uses original inputs
        return self._ind_used

    @ind_used.setter
    def ind_used(self, value):
        '''
        Tell the Bed reader to return data for only a subset (perhaps proper) of the individuals in a particular order
        e.g. 2,10,0 says to return data for three users: the user at index position 2, the user at index position 10, and the user at index position 0.
        '''
        # doesn't need to self.run_once() because only uses original inputs
        self._ind_used = value

    @staticmethod
    def write(snpMatrix, hdf5file, dtype='f8',snp_major=True,compression=None):
        if not isinstance(dtype, str) or len(dtype) != 2 or dtype[0] != 'f' : raise Exception("Expect dtype to start with 'f', e.g. 'f4' for single, 'f8' for double")
        data = (snpMatrix['snps'].T) if snp_major else snpMatrix['snps']
        with h5py.File(hdf5file, "w") as h5:
            h5.create_dataset('snps', data=data,dtype=dtype,compression=compression,shuffle=True)
            h5['snps'].attrs["SNP-major"] = snp_major
            h5.create_dataset('iid', data=snpMatrix['iid'])
            h5.create_dataset('pos', data=snpMatrix['pos'])
            h5.create_dataset('rs', data=snpMatrix['rs'])
