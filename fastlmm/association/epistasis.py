from __future__ import absolute_import
from __future__ import print_function
from pysnptools.util.mapreduce1.runner import *
import logging
import fastlmm.pyplink.plink as plink
import pysnptools.util as pstutil
import pysnptools.util.pheno as pstpheno
import numpy as np
from fastlmm.inference import LMM
import scipy.stats as stats
from pysnptools.snpreader import Bed
from fastlmm.util.pickle_io import load, save
import time
import pandas as pd
from six.moves import range

def epistasis(test_snps,pheno,G0, G1=None, mixing=0.0, covar=None,output_file_name=None,sid_list_0=None,sid_list_1=None,
                 log_delta=None, min_log_delta=-5, max_log_delta=10, 
                 cache_file = None,
                 runner=None, count_A1=None):
    """
    Function performing epistasis GWAS with ML (never REML).  See http://www.nature.com/srep/2013/130122/srep01099/full/srep01099.html.

    :param test_snps: SNPs from which to test pairs. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type test_snps: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string

    :param pheno: A single phenotype: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type pheno: a 'pheno dictionary' or a string

    :param G0: SNPs from which to construct a similarity matrix.
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G0: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string

    :param G1: SNPs from which to construct a second similarity kernel, optional. Also, see 'mixing').
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G1: a `SnpReader <http://fastlmm.github.io/PySnpTools/#snpreader-snpreader>`__ or a string

    :param mixing: Weight between 0.0 (inclusive, default) and 1.1 (inclusive) given to G1 relative to G0.
            If you give no mixing number, G0 will get all the weight and G1 will be ignored.
    :type mixing: number

    :param covar: covariate information, optional: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type covar: a 'pheno dictionary' or a string

    :param sid_list_0: list of sids, optional:
            All unique pairs from sid_list_0 x sid_list_1 will be evaluated.
            If you give no sid_list_0, all sids in test_snps will be used.
    :type sid_list_0: list of strings

    :param sid_list_1: list of sids, optional:
            All unique pairs from sid_list_0 x sid_list_1 will be evaluated.
            If you give no sid_list_1, all sids in test_snps will be used.
    :type sid_list_1: list of strings

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created. The output format is tab-delimited text.
    :type output_file_name: file name

    :param log_delta: A parameter to LMM learning, optional
            If not given will search for best value.
    :type log_delta: number

    :param min_log_delta: (default:-5)
            When searching for log_delta, the lower bounds of the search.
    :type min_log_delta: number

    :param max_log_delta: (default:-5)
            When searching for log_delta, the upper bounds of the search.
    :type max_log_delta: number

    :param cache_file: Name of  file to read or write cached precomputation values to, optional.
                If not given, no cache file will be used.
                If given and file does not exists, will write precomputation values to file.
                If given and file does exists, will read precomputation values from file.
                The file contains the U and S matrix from the decomposition of the training matrix. It is in Python's np.savez (\*.npz) format.
                Calls using the same cache file should have the same 'G0' and 'G1'
    :type cache_file: file name

    :param runner: a `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: `Runner <http://fastlmm.github.io/PySnpTools/#util-mapreduce1-runner-runner>`__

    :param count_A1: If it needs to read SNP data from a BED-formatted file, tells if it should count the number of A1
         alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
    :type count_A1: bool


    :rtype: Pandas dataframe with one row per SNP pair. Columns include "PValue"

    :Example:

    >>> import logging
    >>> from pysnptools.snpreader import Bed
    >>> from fastlmm.util import example_file # Download and return local file name
    >>> from fastlmm.association import epistasis
    >>> logging.basicConfig(level=logging.INFO)
    >>> bed_file = example_file('tests/datasets/all_chr.maf0.001.N300.*','*.bed')
    >>> test_snps = Bed(bed_file,count_A1=True)
    >>> pheno = example_file('tests/datasets/phenSynthFrom22.23.N300.randcidorder.txt')
    >>> covar = example_file('tests/datasets/all_chr.maf0.001.covariates.N300.txt')
    >>> results_dataframe = epistasis(test_snps, pheno, G0=test_snps, covar=covar, 
    ...                                 sid_list_0=test_snps.sid[:10], #first 10 snps
    ...                                 sid_list_1=test_snps.sid[5:15], #Skip 5 snps, use next 10
    ...                                 count_A1=False)
    >>> print(results_dataframe.iloc[0].SNP0, results_dataframe.iloc[0].SNP1,round(results_dataframe.iloc[0].PValue,5),len(results_dataframe))
    1_12 1_9 0.07779 85

    """

    if runner is None:
        runner = Local()

    epistasis = _Epistasis(test_snps, pheno, G0, G1, mixing, covar, sid_list_0, sid_list_1, log_delta, min_log_delta, max_log_delta, output_file_name, cache_file, count_A1=count_A1)
    logging.info("# of pairs is {0}".format(epistasis.pair_count))
    epistasis.fill_in_cache_file()
    result = runner.run(epistasis)
    return result

def write(sid0_list, sid1_list, pvalue_list, output_file):
    """
    Given three arrays of the same length [as per the output of epistasis(...)], writes a header and the values to the given output file.
    """
    with open(output_file,"w") as out_fp:
        out_fp.write("{0}\t{1}\t{2}\n".format("sid0","sid1","pvalue"))
        for i in range(len(pvalue_list)):
            out_fp.write("{0}\t{1}\t{2}\n".format(sid0_list[i],sid1_list[i],pvalue_list[i]))


# could this be written without the inside-out of IDistributable?
class _Epistasis(object) : #implements IDistributable

    def __init__(self, test_snps, pheno, G0, G1=None, mixing=0.0, covar=None,sid_list_0=None,sid_list_1=None,
                 log_delta=None, min_log_delta=-5, max_log_delta=10, output_file=None, cache_file=None, count_A1=None):
        self._ran_once = False

        self.test_snps = test_snps
        self.pheno = pheno
        self.output_file_or_none = output_file
        self.cache_file = cache_file
        self.count_A1 = count_A1
        self.covar = covar
        self.sid_list_0 = np.array(sid_list_0,dtype='str') if sid_list_0 is not None else None
        self.sid_list_1 = np.array(sid_list_1,dtype='str') if sid_list_1 is not None else None
        self.G0=G0
        self.G1_or_none=G1
        self.mixing=mixing
        self.external_log_delta=log_delta
        self.min_log_delta = min_log_delta
        self.max_log_delta = max_log_delta
        self._str = "{0}({1},{2},G0={6},G1={7},mixing={8},covar={3},output_file={12},sid_list_0={4},sid_list_1{5},log_delta={9},min_log_delta={10},max_log_delta={11},cache_file={13})".format(
            self.__class__.__name__, self.test_snps,self.pheno,self.covar,self.sid_list_0,self.sid_list_1,
                 self.G0, self.G1_or_none, self.mixing, self.external_log_delta, self.min_log_delta, self.max_log_delta, output_file, cache_file)
        self.block_size = 1000

    def order_by_test_snps(self, sid_sequence):
        return self.test_snps.sid[sorted(self.test_snps.sid_to_index(sid_sequence))]

    def set_sid_sets(self):
        sid_set_0 = set(self.sid_list_0)
        self.intersect = self.order_by_test_snps(sid_set_0.intersection(self.sid_list_1))
        self.just_sid_0 = self.order_by_test_snps(sid_set_0.difference(self.intersect))
        self.just_sid_1 = self.order_by_test_snps(set(self.intersect).symmetric_difference(self.sid_list_1))
        self._pair_count = len(self.just_sid_0)*len(self.intersect) + len(self.just_sid_0)*len(self.just_sid_1) + len(self.intersect)*len(self.just_sid_1) + len(self.intersect) * (len(self.intersect)-1)//2
        self.test_snps, self.pheno, self.covar, self.G0, self.G1_or_none = pstutil.intersect_apply([self.test_snps, self.pheno, self.covar, self.G0, self.G1_or_none]) #should put G0 and G1 first

    def _run_once(self):
        if self._ran_once:
            return
        self._ran_once = None

        if isinstance(self.test_snps, str):
            self.test_snps = Bed(self.test_snps,count_A1=self.count_A1)

        if isinstance(self.G0, str):
            self.G0 = Bed(self.G0,count_A1=self.count_A1)

        if isinstance(self.pheno, str):
            self.pheno = pstpheno.loadOnePhen(self.pheno,vectorize=True,missing='NaN')

        if self.covar is not None and isinstance(self.covar, str):
            self.covar = pstpheno.loadPhen(self.covar,missing='NaN')

        if self.G1_or_none is not None and isinstance(self.G1_or_none, str):
            self.G1_or_none = Bed(self.G1_or_none,count_A1=self.count_A1)

        if self.sid_list_0 is None:
            self.sid_list_0 = self.test_snps.sid

        if self.sid_list_1 is None:
            self.sid_list_1 = self.test_snps.sid

        self.set_sid_sets()

        #!!Should fix up to add only of no constant columns - will need to add a test case for this
        if self.covar is None:
            self.covar = np.ones((self.test_snps.iid_count, 1))
        else:
            self.covar = np.hstack((self.covar['vals'],np.ones((self.test_snps.iid_count, 1))))
        self.n_cov = self.covar.shape[1] 


        if self.output_file_or_none is None:
            self.__tempdirectory = ".working"
        else:
            self.__tempdirectory = self.output_file_or_none + ".working"

        self._ran_once = True
        

 #start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        self._run_once()
        block_count = self.div_ceil(self._pair_count, self.block_size)
        return block_count



    def work_sequence(self):
        self._run_once()

        return self.work_sequence_range(0,self.work_count)

    def work_sequence_range(self, start, end):
        self._run_once()

        lmm = self.lmm_from_cache_file()
        lmm.sety(self.pheno['vals'])

        for sid0_list, sid1_list in self.pair_block_sequence_range(start,end):
            yield lambda lmm=lmm,sid0_list=sid0_list,sid1_list=sid1_list : self.do_work(lmm,sid0_list,sid1_list)  # the 'lmm=lmm,...' is need to get around a strangeness in Python

    def reduce(self, result_sequence):
        #doesn't need "run_once()"

        frame = pd.concat(result_sequence)
        frame.sort_values(by="PValue", inplace=True)
        frame.index = np.arange(len(frame))

        if self.output_file_or_none is not None:
            frame.to_csv(self.output_file_or_none, sep="\t", index=False)

        return frame

        #!!Find a place to output info like this near the end of the run
        #logging.info("PhenotypeName\t{0}".format(pheno['header']))
        #logging.info("SampleSize\t{0}".format(test_snps.iid_count))
        #logging.info("SNPCount\t{0}".format(test_snps.sid_count))
        #logging.info("Runtime\t{0}".format(time.time()-t0))


    @property
    def tempdirectory(self):
        self._run_once()
        return self.__tempdirectory

    #optional override -- the str name of the instance is used by the cluster as the job name
    def __str__(self):
        #Doesn't need run_once
        return self._str


    def copyinputs(self, copier):
        self._run_once()
        if isinstance(self.test_snps, str):
            copier.input(self.test_snps + ".bed")
            copier.input(self.test_snps + ".bim")
            copier.input(self.test_snps + ".fam")
        else:
            copier.input(self.test_snps)

        copier.input(self.pheno)
        copier.input(self.covar)

        if isinstance(self.G0, str):
            copier.input(self.G0 + ".bed")
            copier.input(self.G0 + ".bim")
            copier.input(self.G0 + ".fam")
        else:
            copier.input(self.G0)

        copier.input(self.G1_or_none)
        copier.input(self.cache_file)

    def copyoutputs(self,copier):
        #Doesn't need run_once
        copier.output(self.output_file_or_none)

 #end of IDistributable interface---------------------------------------

    @staticmethod
    def div_ceil(num, den): #!!move to utils?
        return -(-num//den) #The -/- trick makes it do ceiling instead of floor. "//" will do integer division even in the future and on floats.
    
    def pair_block_sequence_range(self,block_start,block_end):
        self._run_once()
        assert 0 <= block_start  <= block_end <= self.work_count, "real assert"

        block_index = block_start
        start = block_index * self.pair_count // self.work_count
        next_start = (block_index+1) * self.pair_count // self.work_count
        size_goal = next_start - start
        end = block_end * self.pair_count // self.work_count

        sid0_list = []
        sid1_list = []
        for sid0, sid1 in self.pair_sequence_range(start,end):
            sid0_list.append(sid0)
            sid1_list.append(sid1)
            if len(sid0_list) == size_goal:
                yield sid0_list, sid1_list
                block_index += 1
                if block_index == block_end:
                    return
                sid0_list = []
                sid1_list = []
                start = next_start
                next_start = (block_index+1) * self.pair_count // self.work_count
                size_goal = next_start - start
        assert len(sid0_list) == 0, "real assert"

    #If start == end, then returns without yielding anything 
    def pair_sequence_range(self,start,end):
        self._run_once()
        assert 0 <= start <= end <= self._pair_count, "real assert"

        i = start
        for sid0, sid1 in self.pair_sequence_with_start(start):
            yield sid0, sid1
            i = i + 1
            if i == end:
                break
        assert i == end, "Not enough items found. Didn't get to the end"


    def pair_sequence_with_start(self,start):
        self._run_once()

        skip_ref = [start]

        just_sid_0_list = list(self.just_sid_0)
        just_sid_1_list = list(self.just_sid_1)
        intersect_list = list(self.intersect)

        for sid0, sid1 in self.combo_distinct(just_sid_0_list, intersect_list, skip_ref):
            yield sid0, sid1
        for sid0, sid1 in self.combo_distinct(just_sid_0_list, just_sid_1_list, skip_ref):
            yield sid0, sid1
        for sid0, sid1 in self.combo_distinct(intersect_list, just_sid_1_list, skip_ref):
            yield sid0, sid1
        for sid0, sid1 in self.combo_same(intersect_list, skip_ref):
            yield sid0, sid1
        assert skip_ref[0] == 0, "real assert"


    def combo_distinct(self, distinct__list0, distinct__list1, skip_ref):
        row_count = len(distinct__list0)
        col_count = len(distinct__list1)

        if skip_ref[0] >= row_count * col_count:
            skip_ref[0] = skip_ref[0] - row_count * col_count
            assert skip_ref[0] >=0, "real assert"
            return

        row_start = skip_ref[0] // col_count
        skip_ref[0] = skip_ref[0] - row_start * col_count
        assert skip_ref[0] >=0, "real assert"

        for row_index in range(row_start, row_count):
            sid0 = distinct__list0[row_index]
            if row_index == row_start:
                col_start = skip_ref[0]
                skip_ref[0] = 0
            else:
                col_start = 0
            for col_index in range(col_start, col_count):
                sid1 = distinct__list1[col_index]
                yield sid0, sid1

    def combo_same(self, list, skip_ref):
        count = len(list)
        full_size = count * (count + 1) // 2
        if skip_ref[0] >= full_size:
            skip_ref[0] = skip_ref[0] - full_size
            assert skip_ref[0] >=0, "real assert"
            return

        row_start = int((-1 + 2*count - np.sqrt(1 - 4*count + 4*count**2 - 8*skip_ref[0]))//2)
        skip_ref[0] = skip_ref[0] - (count*row_start - (row_start*(1 + row_start))//2)
        assert skip_ref[0] >=0, "real assert"

        for row_index in range(row_start, count):
            sid0 = list[row_index]
            if row_index == row_start:
                col_start = skip_ref[0]
                skip_ref[0] = 0
            else:
                col_start = 0
            for col_index in range(col_start + 1 + row_index, count):
                sid1 = list[col_index]
                assert sid0 is not sid1, "real assert"
                yield sid0, sid1



    @property
    def pair_count(self):
        self._run_once()
        return self._pair_count

    def lmm_from_cache_file(self):
        logging.info("Loading precomputation from {0}".format(self.cache_file))
        lmm = LMM()
        with np.load(self.cache_file) as data:
            lmm.U = data['arr_0']
            lmm.S = data['arr_1']
        return lmm

    def fill_in_cache_file(self):
        self._run_once()

        logging.info("filling in the cache_file and log_delta, as needed")

        if self.G1_or_none is None:
            self.G1val_or_none = None
        else:
            self.G1val_or_none = self.G1_or_none.read().val

        # The S and U are always cached, in case they are needed for the cluster or for multi-threaded runs
        if self.cache_file is None:
            self.cache_file = os.path.join(self.__tempdirectory, "cache_file.npz")
            if os.path.exists(self.cache_file): # If there is already a cache file in the temp directory, it must be removed because it might be out-of-date
                os.remove(self.cache_file)

        lmm = None
        if not os.path.exists(self.cache_file):
            logging.info("Precomputing eigen")
            lmm = LMM()
            G0_standardized = self.G0.read().standardize()
            lmm.setG(G0_standardized.val, self.G1val_or_none, a2=self.mixing)
            logging.info("Saving precomputation to {0}".format(self.cache_file))
            pstutil.create_directory_if_necessary(self.cache_file)
            np.savez(self.cache_file, lmm.U,lmm.S) #using np.savez instead of pickle because it seems to be faster to read and write

        if self.external_log_delta is None:
            if lmm is None:
                lmm = self.lmm_from_cache_file()

            logging.info("searching for internal delta")
            lmm.setX(self.covar)
            lmm.sety(self.pheno['vals'])
            #log delta is used here. Might be better to use findH2, but if so will need to normalized G so that its K's diagonal would sum to iid_count
            result = lmm.find_log_delta(REML=False, sid_count=self.G0.sid_count, min_log_delta=self.min_log_delta, max_log_delta=self.max_log_delta  ) #!!what about findA2H2? minH2=0.00001
            self.external_log_delta = result['log_delta']

        self.internal_delta = np.exp(self.external_log_delta) * self.G0.sid_count
        logging.info("internal_delta={0}".format(self.internal_delta))
        logging.info("external_log_delta={0}".format(self.external_log_delta))


    do_pair_count = 0
    do_pair_time = time.time()

    def do_work(self, lmm, sid0_list, sid1_list):
        dataframe = pd.DataFrame(
            index=np.arange(len(sid0_list)),
            columns=('SNP0', 'Chr0', 'GenDist0', 'ChrPos0', 'SNP1', 'Chr1', 'GenDist1', 'ChrPos1', 'PValue', 'NullLogLike', 'AltLogLike')
            )
        #!!Is this the only way to set types in a dataframe?
        dataframe['Chr0'] = dataframe['Chr0'].astype(np.float)
        dataframe['GenDist0'] = dataframe['GenDist0'].astype(np.float)
        dataframe['ChrPos0'] = dataframe['ChrPos0'].astype(np.float)
        dataframe['Chr1'] = dataframe['Chr1'].astype(np.float)
        dataframe['GenDist1'] = dataframe['GenDist1'].astype(np.float)
        dataframe['ChrPos1'] = dataframe['ChrPos1'].astype(np.float)
        dataframe['PValue'] = dataframe['PValue'].astype(np.float)
        dataframe['NullLogLike'] = dataframe['NullLogLike'].astype(np.float)
        dataframe['AltLogLike'] = dataframe['AltLogLike'].astype(np.float)


        #This is some of the code for a different way that reads and dot-products 50% more, but does less copying. Seems about the same speed
        #sid0_index_list = self.test_snps.sid_to_index(sid0_list)
        #sid1_index_list = self.test_snps.sid_to_index(sid1_list)
        #sid_index_union_dict = {}
        #sid0_index_index_list = self.create_index_index(sid_index_union_dict, sid0_index_list)
        #sid1_index_index_list = self.create_index_index(sid_index_union_dict, sid1_index_list)
        #snps0_read = self.test_snps[:,sid0_index_list].read().standardize()
        #snps1_read = self.test_snps[:,sid1_index_list].read().standardize()

        sid_union = set(sid0_list).union(sid1_list)
        sid_union_index_list = sorted(self.test_snps.sid_to_index(sid_union))
        snps_read = self.test_snps[:,sid_union_index_list].read().standardize()

        sid0_index_list = snps_read.sid_to_index(sid0_list)
        sid1_index_list = snps_read.sid_to_index(sid1_list)

        products = snps_read.val[:,sid0_index_list] * snps_read.val[:,sid1_index_list] # in the products matrix, each column i is the elementwise product of sid i in each list
        X = np.hstack((self.covar, snps_read.val, products))
        UX = lmm.U.T.dot(X)
        k = lmm.S.shape[0]
        N = X.shape[0]
        if (k<N):
            UUX = X - lmm.U.dot(UX)
        else:
            UUX = None

        for pair_index, sid0 in enumerate(sid0_list):
            sid1 = sid1_list[pair_index]
            sid0_index = sid0_index_list[pair_index]
            sid1_index = sid1_index_list[pair_index]

            index_list = np.array([pair_index]) #index to product
            index_list = index_list + len(sid_union_index_list) #Shift by the number of snps in the union
            index_list = np.hstack((np.array([sid0_index,sid1_index]),index_list)) # index to sid0 and sid1
            index_list = index_list + self.covar.shape[1] #Shift by the number of values in the covar
            index_list = np.hstack((np.arange(self.covar.shape[1]),index_list)) #indexes of the covar

            index_list_less_product = index_list[:-1] #index to everything but the product

            #Null -- the two additive SNPs
            lmm.X = X[:,index_list_less_product]
            lmm.UX = UX[:,index_list_less_product]
            if (k<N):
                lmm.UUX = UUX[:,index_list_less_product]
            else:
                lmm.UUX = None
            res_null = lmm.nLLeval(delta=self.internal_delta, REML=False)
            ll_null = -res_null["nLL"]

            #Alt -- now with the product feature
            lmm.X = X[:,index_list]
            lmm.UX = UX[:,index_list]
            if (k<N):
                lmm.UUX = UUX[:,index_list]
            else:
                lmm.UUX = None
            res_alt = lmm.nLLeval(delta=self.internal_delta, REML=False)
            ll_alt = -res_alt["nLL"]

            test_statistic = ll_alt - ll_null
            degrees_of_freedom = 1
            pvalue = stats.chi2.sf(2.0 * test_statistic, degrees_of_freedom)
            logging.debug("<{0},{1}>, null={2}, alt={3}, pvalue={4}".format(sid0,sid1,ll_null,ll_alt,pvalue))

            dataframe.iloc[pair_index] = [
                 sid0, snps_read.pos[sid0_index,0],  snps_read.pos[sid0_index,1], snps_read.pos[sid0_index,2],
                 sid1, snps_read.pos[sid1_index,0],  snps_read.pos[sid1_index,1], snps_read.pos[sid1_index,2],
                 pvalue, ll_null, ll_alt]

            self.do_pair_count += 1
            if self.do_pair_count % 100 == 0:
                start = self.do_pair_time
                self.do_pair_time = time.time()
                logging.info("do_pair_count={0}, time={1}".format(self.do_pair_count,self.do_pair_time-start))

        return dataframe

if __name__ == "__main__":

    import doctest
    doctest.testmod()

    print("done")

