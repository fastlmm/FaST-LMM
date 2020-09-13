from __future__ import absolute_import
import numpy as SP
import subprocess, sys, os.path
from itertools import *
from fastlmm.pyplink.snpset import *
from fastlmm.pyplink.altset_list import *
import pandas as pd
import logging
from six.moves import range

WRAPPED_PLINK_PARSER_PRESENT = None

def decide_once_on_plink_reader():
    #This is now done in a method, instead of at the top of the file, so that messages can be re-directed to the appropriate stream.
    #(Usually messages go to stdout, but when the code is run on Hadoop, they are sent to stderr)

    global WRAPPED_PLINK_PARSER_PRESENT
    if WRAPPED_PLINK_PARSER_PRESENT == None:
        # attempt to import wrapped plink parser
        try:
            from pysnptools.snpreader import wrap_plink_parser
            WRAPPED_PLINK_PARSER_PRESENT = True #!!does the standardizer work without c++
            logging.info("using c-based plink parser")
        except Exception as detail:
            logging.warn(detail)
            WRAPPED_PLINK_PARSER_PRESENT = False


class Bed(object):
    '''
    This is a class that does random-access reads of a BED file. For examples of its use see its 'read' method.
    '''


    def __init__(self,basefilename):
        '''
        basefilename    : string of the basename of [basename].bed, [basename].bim,
                          and [basename].fam
        '''

        self._ran_once = False
        self._filepointer = None

        self.basefilename = basefilename

    #!! similar code in fastlmm
    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.basefilename)

    @property
    def snp_to_index(self):
        self.run_once()
        return self._snp_to_index

    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True

        famfile = self.basefilename+ '.fam'
        bimfile = self.basefilename+'.bim'

        logging.info("Loading fam file {0}".format(famfile))
        self._original_iids = SP.loadtxt(famfile,dtype = 'str',usecols=(0,1),comments=None)
        logging.info("Loading bim file {0}".format(bimfile))

        self.bimfields = pd.read_csv(bimfile,delimiter = '\s',usecols = (0,1,2,3),header=None,index_col=False,engine='python')
        self.rs = SP.array(self.bimfields[1].tolist(),dtype='str')
        self.pos = self.bimfields[[0,2,3]].values
        self._snp_to_index = {}
        logging.info("indexing snps");
        for i in range(self.snp_count):
            snp = self.rs[i]
            if snp in self._snp_to_index : raise Exception("Expect snp to appear in bim file only once. ({0})".format(snp))
            self._snp_to_index[snp]=i

        bedfile = self.basefilename+ '.bed'
        self._filepointer = open(bedfile, "rb")
        mode = self._filepointer.read(2)
        if mode != b'l\x1b': raise Exception('No valid binary BED file')
        mode = self._filepointer.read(1) #\x01 = SNP major \x00 = individual major
        if mode != b'\x01': raise Exception('only SNP-major is implemented')
        logging.info("bed file is open {0}".format(bedfile))
        return self

    def __del__(self):
        if self._filepointer != None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
            self._filepointer.close()

    def copyinputs(self, copier):
        # doesn't need to self.run_once() because only uses original inputs
        copier.input(self.basefilename + ".bed")
        copier.input(self.basefilename + ".bim")
        copier.input(self.basefilename + ".fam")


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

        Examples:

        >>> from fastlmm.util import example_file # Download and return local file name
        >>> bed_file = example_file('tests/datasets/all_chr.maf0.001.N300.*','*.bed').replace('.bed','')
        >>> bed = Bed(bed_file)
        >>> ret = bed.read()
        >>> len(ret['rs'])
        1015
        >>> ret = bed.read(AllSnps())
        >>> len(ret['rs'])
        1015
        >>> ret = bed.read(SnpAndSetName('someset',['23_9','23_2']))
        >>> ",".join(ret['rs'])
        '23_9,23_2'
        >>> ret = bed.read(PositionRange(0,10))
        >>> ",".join(ret['rs'])
        '1_12,1_34,1_10,1_35,1_28,1_25,1_36,1_39,1_4,1_13'


        >>> altset_list1 = SnpAndSetNameCollection(example_file('tests/datasets/set_input.small.txt')) # get the list of snpsets defined in the file
        >>> altset_list2 = Subset(altset_list1,['set1','set5'])                       # only use a subset of those snpsets
        >>> altset_list3 = MinMaxSetSize(altset_list2, minsetsize=2, maxsetsize=15)   # only use the subset of subsets that contain between 2 & 15 snps (inclusive)
        >>> bed_file = example_file('tests/datasets/all_chr.maf0.001.N300.*','*.bed').replace('.bed','')
        >>> bed = Bed(bed_file)
        >>> altsetlist_plusbed = altset_list3.addbed(bed)                             # apply altset_list3 to this bed file
        >>> len(altsetlist_plusbed)                                                   # tell how many snpsets there will be
        1
        >>> snpset_plusbed = list(altsetlist_plusbed)[0]
        >>> str(snpset_plusbed)                                                       # the name of the snpset
        'set5'
        >>> len(snpset_plusbed)                                                       # the number of snps in the snpset
        13
        >>> ret = snpset_plusbed.read()
        >>> ",".join(ret['rs'])
        '5_12,5_28,5_32,5_5,5_11,5_1,5_9,5_3,5_19,5_7,5_21,5_15,5_23'

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
        Tell the Bed reader to return data for only a subset (perhaps proper) of the individuals in a particular order
        e.g. 2,10,0 says to return data for three users: the user at index position 2, the user at index position 10, and the user at index position 0.
        '''
        # doesn't need to self.run_once() because only uses original inputs
        self._ind_used = value

    @property
    def original_iids(self):
        self.run_once()
        return self._original_iids

    def counts_and_indexes(self, snpset_withbbed):
        iid_count_in = len(self.original_iids)
        snp_count_in = self.snp_count
        if hasattr(self,'_ind_used'):
            iid_count_out = len(self.ind_used)
            iid_index_out = self.ind_used
        else:
            iid_count_out = iid_count_in
            iid_index_out = list(range(0,iid_count_in))
        snp_count_out = len(snpset_withbbed)
        snp_index_out = list(snpset_withbbed)  #make a copy, in case it's in some strange format, such as HDF5
        return iid_count_in, iid_count_out, iid_index_out, snp_count_in, snp_count_out, snp_index_out


    @staticmethod
    def read_with_specification(snpset_withbbed, order="F", dtype=SP.float64, force_python_only=False):
        # doesn't need to self.run_once() because it is static
        decide_once_on_plink_reader()
        global WRAPPED_PLINK_PARSER_PRESENT

        bed = snpset_withbbed.bed
        iid_count_in, iid_count_out, iid_index_out, snp_count_in, snp_count_out, snp_index_out = bed.counts_and_indexes(snpset_withbbed)

        if WRAPPED_PLINK_PARSER_PRESENT and not force_python_only:
            from pysnptools.snpreader import wrap_plink_parser
            SNPs = SP.zeros((iid_count_out, snp_count_out), order=order, dtype=dtype)
            bed_fn = bed.basefilename + ".bed"
            count_A1 = False

            if dtype == SP.float64:
                if order=="F":
                    wrap_plink_parser.readPlinkBedFile2doubleFAAA(bed_fn.encode('ascii'), iid_count_in, snp_count_in, count_A1, iid_index_out, snp_index_out, SNPs)
                elif order=="C":
                    wrap_plink_parser.readPlinkBedFile2doubleCAAA(bed_fn.encode('ascii'), iid_count_in, snp_count_in, count_A1, iid_index_out, snp_index_out, SNPs)
                else:
                    raise Exception("order '{0}' not known, only 'F' and 'C'".format(order));
            elif dtype == SP.float32:
                if order=="F":
                    wrap_plink_parser.readPlinkBedFile2floatFAAA(bed_fn.encode('ascii'), iid_count_in, snp_count_in, count_A1, iid_index_out, snp_index_out, SNPs)
                elif order=="C":
                    wrap_plink_parser.readPlinkBedFile2floatCAAA(bed_fn.encode('ascii'), iid_count_in, snp_count_in, count_A1, iid_index_out, snp_index_out, SNPs)
                else:
                    raise Exception("dtype '{0}' not known, only float64 and float32".format(dtype))
            
        else:
            # An earlier version of this code had a way to read consecutive SNPs of code in one read. May want
            # to add that ability back to the code. 
            # Also, note that reading with python will often result in non-contigious memory, so the python standardizers will automatically be used, too.       
            logging.warn("using pure python plink parser (might be much slower!!)")
            SNPs = SP.zeros(((int(SP.ceil(0.25*iid_count_in))*4),snp_count_out),order=order, dtype=dtype) #allocate it a little big
            for SNPsIndex, bimIndex in enumerate(snpset_withbbed):

                startbit = int(SP.ceil(0.25*iid_count_in)*bimIndex+3)
                bed._filepointer.seek(startbit)
                nbyte = int(SP.ceil(0.25*iid_count_in))
                bytes = SP.array(bytearray(bed._filepointer.read(nbyte))).reshape((int(SP.ceil(0.25*iid_count_in)),1),order='F')

                SNPs[3::4,SNPsIndex:SNPsIndex+1][bytes>=64]=SP.nan
                SNPs[3::4,SNPsIndex:SNPsIndex+1][bytes>=128]=1
                SNPs[3::4,SNPsIndex:SNPsIndex+1][bytes>=192]=2
                bytes=SP.mod(bytes,64)
                SNPs[2::4,SNPsIndex:SNPsIndex+1][bytes>=16]=SP.nan
                SNPs[2::4,SNPsIndex:SNPsIndex+1][bytes>=32]=1
                SNPs[2::4,SNPsIndex:SNPsIndex+1][bytes>=48]=2
                bytes=SP.mod(bytes,16)
                SNPs[1::4,SNPsIndex:SNPsIndex+1][bytes>=4]=SP.nan
                SNPs[1::4,SNPsIndex:SNPsIndex+1][bytes>=8]=1
                SNPs[1::4,SNPsIndex:SNPsIndex+1][bytes>=12]=2
                bytes=SP.mod(bytes,4)
                SNPs[0::4,SNPsIndex:SNPsIndex+1][bytes>=1]=SP.nan
                SNPs[0::4,SNPsIndex:SNPsIndex+1][bytes>=2]=1
                SNPs[0::4,SNPsIndex:SNPsIndex+1][bytes>=3]=2
            SNPs = SNPs[iid_index_out,:] #reorder or trim any extra allocation

        ret = {
                'rs'     :bed.rs[snp_index_out],
                'pos'    :bed.pos[snp_index_out,:],
                'snps'   :SNPs,
                'iid'    :bed.original_iids[iid_index_out,:]
                }
        return ret

if __name__ == "__main__":
    if False:
        from fastlmm.util import example_file # Download and return local file name
        bed_file = example_file('tests/datasets/all_chr.maf0.001.N300.*','*.bed')
        bed = Bed(bed_file)
        ret = bed.read()
        len(ret['rs'])
        #1015
        ret = bed.read(AllSnps())
        len(ret['rs'])
        #1015
        ret = bed.read(SnpAndSetName('someset',['23_9','23_2']))
        ",".join(ret['rs'])
        #'23_9,23_2'
        ret = bed.read(PositionRange(0,10))
        ",".join(ret['rs'])
        #'1_12,1_34,1_10,1_35,1_28,1_25,1_36,1_39,1_4,1_13'


    logging.basicConfig(level=logging.WARNING)
    import doctest
    doctest.testmod()
