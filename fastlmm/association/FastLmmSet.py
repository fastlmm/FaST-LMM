from __future__ import absolute_import
from __future__ import print_function
import fastlmm.association.lrt as lr
import fastlmm.association.score as score
import fastlmm.association.testCV as testCV
import fastlmm.inference as fastlmm
from fastlmm.pyplink.plink import *
from pysnptools.util.pheno import *
from fastlmm.pyplink.altset_list import *
import subprocess, sys, os.path
import fastlmm.util.stats.chi2mixture as c2
from pysnptools.util.mapreduce1.distributable import *
from pysnptools.util.mapreduce1.runner import *
import pdb
import pysnptools.util as pstutil
import fastlmm.util.util as utilx
import fastlmm.util.stats as ss
import time
from .Result import *
from .PairResult import *
from fastlmm.association.tests import *
import fastlmm.util.genphen as gp
import scipy as sp
from itertools import *
from fastlmm.pyplink.snpreader.Bed import *
import logging
import warnings
from tempfile import TemporaryFile
import fastlmm.util.preprocess as util
import os
from six.moves import map
from six.moves import range
from six.moves import zip

class FastLmmSet: # implements IDistributable
    '''
    A class for specifying a FastLmmSet and then running it.
    '''


    def addpostifx_to_outfile(self):
        if self.datestamp is not None:
            if self.datestamp=="auto":
                self.datestamp=pstutil._datestamp()
            self.outfile=utilx.appendtofilename(self.outfile,self.datestamp,"_")

    def __init__(self, **entries):
        '''
        outfile         : string of the filename to which to write out results (two files will be generated from this, on a record of the run)
        phenofile       : string of the filename containing phenotype
        alt_snpreader   : A snpreader for the SNPs for alternative kernel. If just a file name is given, the Bed reader (with standardization) is used.
        bedfilealt      : (deprecated) same as alt_snpreader
        filenull        : string of the filename (with .bed or .ped suffix) containing SNPs for null kernel. Should be Bed format if autoselect is true.
        extractSim      : string of a filename containing a list of whitespace-delmited SNPs. Only these SNPs will be used when filenull is read.
                               By default all SNPs in filenull are used.
                               It is an error to specify extractSim and not filenull.
                               Currently assumes that filenull is in bed format and that autoselect is not being used.
        altset_list     : list of the altsets
                               By default this is a file that will be read by SnpAndSetNameCollection
                               but a file of nuc ranges can be given again via 'NucRangeSet(filename)'
        altset_list2    : if this is present, then does an interaction test between genes in altset_list and alt_setlist2
        covarfile       : string of the filename containing covariate
        mpheno=1        : integer representing the index of the testing phenotype (starting at 1)
        ipheno=0        : same as mpheno, but starts at 0 (deprecated)
        mindist=2.0     : SNPs within mindist from the alternative SNPs will be removed from
                          the null kernel computation
        idist=1         : the index of the position index to use with mindist
                            1 : genomic distance
                            2 : base-pair distance
        autoselect = True   : Should autoselect to used?
        test="lrt"      :['sc_davies', 'sc_mom', 'sc_liuskat','sc_daviesskat','sc_all','lrt','lrt_up']
        nperm = 10       : number of pemutations per test
        npermabs = None : absolutely number of permutations
        calseed =   : the int that gets added to the random seed 34343 used to generate permutations for lrt null fitting.
        write_lrtperm   : If true, write the lrtperm vector (dictated by calseed) to a file.
        forcefullrank   :  If true, the covariance is always computed for lrt (default False)
        minsetsize      : If non-None, only include sets at least this large (inclusive)
        maxsetsize      : If non-None, only include sets no more than this large (inclusive)
        datestamp       : Defaults to None. Use "auto" to have it generate a unique date and time string
                          to append to filenames. Otherwise, it appends whatever you set it to.
        log             : (Defaults to not changing logging level) Level of log messages, e.g. logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO
        '''

        # member variables with default values, or those not defined in the input file    
        self.mpheno=1
        self.mindist = 2.0
        self.idist=1
        self.autoselect = True
        self.extractSim = None
        self.nperm = 10 # used to permute the data in order to get empirical distrib of null for param fitting of null; these are global
        self.npermabs = None #used instead of nperm to compute an at least this many permutations (I say at least because it is # of perms per test)
        self.cache_from_perm=False #If true, then actually does the permutations with the real, all in run_test() so can cache expensive things
        self.nlocalperm=None #same as nperm, but local to each test (doesn't do pooling)
        self.fitlocal=False #if true, fit aUD to each local set
        self.write_lrtperm = False  #when set to True, writes a third file containing only the p-values resulting from nperm
        self.nullfitfile = None   #file from which to grab the null stats from. should be generated using write_lrtperm
        self.calseed = None# was 0 # the int that gets added to the random seed 34343 used to generate permutations for lrt null fitting
        self.sets=None
        self.permute=None       # permute the SNPs being tested (or, equivalently, the compliment) using this with 1,2,3,4, etc. for different seeds, used for type I error tables
        self.forcefullrank = False
        self.nullfit="qq"       #alternative is "ml"    
        self.qmax=0.1
        self.minsetsize = 1
        self.maxsetsize = 10000000
        self.datestamp = None #use auto to have it auto-generate a date stamp
        self.covarimp = 'standardize' #use 'standardize' for mean and variance standardization, with mean imputation
        self.altset_list2=None#for doing pairs of sets
        self.nullModel = None
        self.altModel = None
        self.genphen=None#MUST LEAVE THIS AS THE DEFAULT#{"varE":1.0,"varG":1.0, "varBack":1.0,"varCov":1.0,"link":'linear',"casefrac":0.5,"once":false}      #generate synthetic phen, using LMM, and real SNPs
       
        self.scoring = None
        self.greater_is_better = None
        self.log = None
        self.detailed_table = False
        self.signal_ratio = True

        self._synthphenfile=None

        self.alt_snpreader = None
        self.show_pvalue_5050 = False

        
        if hasattr(self,"ipheno"):
            assert not hasattr(self,"mpheno")
            self.mpheno = self.ipheno+1

        self.__dict__.update(entries)
        self._check_entries(entries)           
        self._ran_once = False
        self.addpostifx_to_outfile()

        if self.hasNonNoneAttr("alt_snpreader") and self.hasNonNoneAttr("bedfilealt") : raise Exception("alt_snpreader or bedfilealt may be given, but not both")
        if self.hasNonNoneAttr("bedfilealt"):
            warnings.warn("bedfilealt is deprecated. alt_snpreader instead", DeprecationWarning)
            self.alt_snpreader = self.bedfilealt
            delattr(self,"bedfilealt")
        if isinstance(self.alt_snpreader, str):
            self.alt_snpreader = Bed(self.alt_snpreader)


        # convert deprecated "verbose" into "log"        
        if self.hasNonNoneAttr("log") and self.hasNonNoneAttr("verbose") :
            raise Exception("log or verbose may be given, but not both")
        if self.hasNonNoneAttr("verbose"):
            if self.verbose:
                self.log = logging.INFO
            else:
                self.log = logging.CRITICAL
            delattr(self, "verbose")

        if self.hasNonNoneAttr("log"): #If neither were set
            logger = logging.getLogger()
            logger.setLevel(self.log)


        if not hasattr(self, "test") : raise Exception("FastLmmSet must have 'test' set")
        #allowed_tests=['sc_davies', 'sc_mom', 'sc_liuskat','sc_daviesskat','sc_all','lrt','cv_l1','cv_l2']
        #if self.test not in allowed_tests: raise Exception("test must be one of " + str(allowed_tests) +", but found " + self.test)

    # =================================================================
    # BEGIN Implementation of the IDistributable interface described in pysnptools.util.mapreduce1.distributable
    # =================================================================

    def hasNonNoneAttr(self, attr):
        return hasattr(self,attr) and getattr(self,attr) != None


    # required by IDistributable
    @property
    def work_count(self):
        '''
        Tells how many work items there are
        '''
        self.run_once() #load files, etc. -- stuff we only want to do once per task (e.g. on the cluster)
                        #no matter how many times we call 'run_test' (and then 'reduce' which of course
                        #only gets run one time, and calls this too, but doesn't actually do the work).

        if self.altset_list2 is None: #singleton sets
            return len(self.altsetlist_filtbysnps) * (1+self.nperm)
        else: #pairs of sets
            return len(self.altsetlist_filtbysnps)*len(self.__altsetlist2_filtbysnps)* (1+self.nperm)   

    def printPhenToFile(self, nInd, SNPsalt, y):
        self._synthphenfile=utilx.appendtofilename(self.outfile,"synthPhenVarBack")
        np.savetxt(self._synthphenfile,y)
        with open(self._synthphenfile,"w") as fp: 
            for n in sp.arange(0,nInd):
                 fp.write(SNPsalt['iid'][n][0] +'\t'+ SNPsalt['iid'][n][1]+ "\t" + '%f' % y[n] + "\n")

    def work_sequence(self):
        '''
        Enumerates a sequence of work items
        Each work item is a lambda expression (i.e. function pointer) that calls 'run_test', returning a list of Results (often just one)
        '''
        self.run_once() #load files, etc. -- stuff we only want to do once per task (e.g. on the cluster)
                        #no matter how many times we call 'run_test' (and then 'reduce' which of course
                        #only gets run one time, and calls this too, but doesn't actually do the work).
        ttt0=time.time()
        y=None        
        y_back=None
        nInd=len(self.__y)               
        haswrittenphen=False;                            
                
        if self.altset_list2 is None: #singleton sets            
            for iset, altset in enumerate(self.altsetlist_filtbysnps):
                for iperm in range(-1, self.nperm):   #note that self.nperm is the 'stop', not the 'count'
                    SNPsalt=altset.read()
                    SNPsalt['snps'] = util.standardize(SNPsalt['snps'])
                    G1 = SNPsalt['snps']/sp.sqrt(SNPsalt['snps'].shape[1])  
                    ichrm =  ",".join(sp.array(sp.unique(SNPsalt['pos'][:,0]),dtype=str)) 
                    minpos= str(sp.min(SNPsalt['pos'][:,2]))
                    maxpos= str(sp.max(SNPsalt['pos'][:,2]))
                    iposrange = minpos + "-" + maxpos                     
                    
                    if self.genphen is None:
                        y=self.__y
                    else: 
                        assert self.permute is None, "Error: using permute with genphen--there should be no need, use genphen['seed'] instead"
                        self.__y=None      

                        #if (self.__G0 is not None) and y_G0 is None:  #cache so only do it once
                        if (self.__SNPs0 is not None) and y_G0 is None:  #cache so only do it once   
                            newseed = self.mainseed ^ self.genphen['seed']
                            from numpy.random import RandomState
                            randomstate = RandomState(newseed)
                            nSnp=self.__SNPs0['data']['snps'].shape[1]
                            #good for low rank, other wise, use the dual, as in gp.genphen
                            raise Exception("Bug below. should be randn, is rand")
                            y_G0=sp.sqrt(self.genphen["varBack"]/nSnp)*self.__SNPs0['data']['snps'].dot(randomstate.rand(nSnp,1))    #TODO: CL This seems to be a bug. Should be randn
                            #y_G0=sp.sqrt(self.genphen["varBack"])*self.__SNPs0['data']['snps'].dot(randomstate.rand(nSnp,1))    #TODO: CL This seems to be a bug. Should be randn
                        elif self.__SNPs0 is None:
                            y_G0=0      

                        #always have the same background signal                                                
                        if self.genphen["varBackNullFileGen"] is not None and y_back is None:                                                      
                            nSnp=self.__SNPs0['data']['snps'].shape[1]
                            y_back=sp.sqrt(self.genphen["varBack"]/nSnp)*self.__varBackNullSnpsGen['snps'].dot(sp.random.randn(self.__varBackNullSnpsGen['snps'].shape[1],1))/sp.sqrt(self.__varBackNullSnpsGen['snps'].shape[1]) 
                            #self.printPhenToFile(nInd, SNPsalt, y_back);
                        elif "varBackNullPhenGen" in self.genphen and self.genphen["varBackNullPhenGen"] is not None and y_back is None:
                            y_back=loadPhen(filename = self.genphen["varBackNullPhenGen"])['vals']                            
                        elif y_back is None:          
                            y_back=0                          
                        
                        if self.genphen["once"] and y is None: # only generate the phenotype once for entire run                            
                            newseed = self.mainseed ^ self.genphen['seed']  
                            assert  self.genphen["varG"]==0, "doesn't make sense to have varG>0 and only one phen--not sure why I put this code there"       
                            if self.genphen["varG"]>0 and nInd<=SNPsalt['snps'].shape[1]:                                                       
                                Kall = self.KfromAltSnps(nInd) #useful for full rank
                                y=gp.genphen(y_G0=y_G0+y_back,G1=None,covDat=self.__X,options=self.genphen,nInd=nInd,
                                             K1=Kall,randseed=newseed)                          
                            else:     
                                Kall=None                                     
                                y=gp.genphen(y_G0=y_G0+y_back,G1=G1,covDat=self.__X,options=self.genphen,nInd=nInd,randseed=newseed)
                        elif (not self.genphen["once"]):  #generate for each set in turn                           
                            eachseed=utilx.combineseeds(iset,self.genphen['seed'])                            
                            newseed = self.mainseed ^ eachseed                              
                            y=gp.genphen(y_G0=y_G0+y_back,G1=G1,covDat=self.__X,options=self.genphen,nInd=nInd,randseed=newseed) 
                                                                
                    assert y is not None, "y is None"                                   
                    yield lambda altset=altset,iset=iset,iperm=iperm,y=y,ichrm=ichrm, iposrange=iposrange : self.run_test(SNPs1=SNPsalt,G1=G1, y=y, altset=altset, iset=iset, iperm=iperm, ichrm=ichrm, iposrange=iposrange)
        else: #pairs of sets
            raise Exception("not implemented, started a long time ago and never finished")
            #for iperm in xrange(-1, self.nperm):   #note that self.nperm is the 'stop', not the 'count'
            #    for iset, altset in enumerate(self.__altsetlist_filtbysnps):
            #         for iset2, altset2 in enumerate(self.__altsetlist2_filtbysnps):
            #            if iset!=iset2:
            #                yield lambda altset=altset,iset=iset,altset2=altset2,iset2=iset2,iperm=iperm : self.run_interactiontest(altset,iset,altset2,iset2,iperm)

        ttt1=time.time()
        logging.info("---------------------------------------------------")
        logging.info("Elapsed time for all tests is %.2f seconds" % (ttt1-ttt0))
        logging.info("---------------------------------------------------")
    
    def check_for_None(self, alteqnull, alteqnullperm):
        none_ind = []
        for i in range(len(alteqnull)):
            if alteqnull[i] is None:
                none_ind.append(i)
        none_ind_perm = []
        for i in range(len(alteqnullperm)):
            if alteqnullperm[i] is None:
                none_ind_perm.append(i)
        if len(none_ind)>0:
            raise Exception("found None entries in alteqnull during reduce, which means some Result objects were missing")
        if len(none_ind_perm)>0:
            raise Exception("found None entries in alteqnullperm during reduce, which means some Result objects were missing")

    # required by IDistributable
    def reduce(self, result_list_sequence):
        '''
        Given a sequence of results from 'run_test', create the output report.

        '''
        logging.info("last modified: %s" % time.ctime(os.path.getmtime(__file__)))
        logging.info("-------------------------------------------------")

        # Create the info file and direct all 'stdout' messages to both stdout and to the info file
        infofile=utilx.appendtofilename(self.outfile,"info")
        outfiletab=self.outfile
        if self.datestamp=="auto": self.outfile=utilx.appendtofilename(self.outfile,"tab")

        try:
            pstutil.create_directory_if_necessary(infofile)
        except:
            logging.warn("Exception while creating directory for '{0}'. Assuming that other cluster task is creating it.".format(infofile))

        logging_handler=logging.FileHandler(infofile,"w",delay=False)
        logger = logging.getLogger()
        logger.addHandler(logging_handler)

        logging.info("distributable = " + self.__repr__())

        self.run_once() #load files, etc. -- doesn't actually do the work if it's already been done
                         #note, however, that before reduce is called, work_count calls run_once() anyhow

        result_dict = {}

        lrt = SP.nan*SP.ones(len(self.altsetlist_filtbysnps))
        # whether alt and null models give the same margll
        alteqnull = [None]*len(self.altsetlist_filtbysnps)

        lrtperm = SP.nan*SP.ones(len(self.altsetlist_filtbysnps)*self.nperm)
        alteqnullperm = [None]*len(self.altsetlist_filtbysnps)*self.nperm
        setsizeperm = SP.nan*SP.ones(len(self.altsetlist_filtbysnps)*self.nperm)

        npvals=self.test.npvals
        pv_adj= sp.nan*sp.ones((len(self.altsetlist_filtbysnps)))
              
        # results can come in any order, so we have to use iperm and iset to put them in the right place
        # there is one result instance for each combination of test and permutation, and here we are just gathering them
        # into the arrays from above

        for result_list in result_list_sequence:
            for result in result_list:
                if result.iperm < 0:
                    result_dict[result.iset] = result
                    #iset is the index of the test (irrespective of permutation)
                    lrt[result.iset] = self.test.lrt_method(result)
                    alteqnull[result.iset] = result.alteqnull #equiv to result["alteqnull"]
                    #pv_adj[result.iset,:] = self.test.pv_adj_from_result(result)                
                    pv_adj[result.iset] = self.test.pv_adj_from_result(result)                
                else:
                    isetiperm = result.iset+result.iperm*len(self.altsetlist_filtbysnps)
                    lrtperm[isetiperm] = self.test.lrt_method(result)
                    alteqnullperm[isetiperm] = result.alteqnull
                    setsizeperm[isetiperm] = result.setsize
                if result.alteqnull is None and str(self.test)[0:3]=="lrt":
                    raise Exception("self.alteqnull is None")

        #look for None to see if anything didn't get filled in, and if so, where
        if str(self.test)[0:3]=="lrt":
            self.check_for_None(alteqnull, alteqnullperm)
            
        if self.write_lrtperm and self.nperm>0:
            self._saveArray("lrtperm", ["2*(LL(alt)-LL(null))","alteqnull","setsize"], (lrtperm,alteqnullperm,setsizeperm))
                
        pv_adj,ind = self.test.pv_adj_and_ind(self.nperm, pv_adj, self.nullfit, 
                                              lrt, lrtperm, alteqnull, alteqnullperm,self.qmax, self.nullfitfile, self.nlocalperm)

        logging.info("writing the result files : " + self.outfile + "")

        #TODO: move the following to a function/object:
        with open(outfiletab,"w") as fp:
            if type(self.test) is Lrt: #dan: this is ugly, will fix soon                
                self.test.write(fp, ind, result_dict, pv_adj, self.detailed_table, self.signal_ratio)
            else:
                self.test.write(fp, ind, result_dict, pv_adj, self.detailed_table)

        if not self.show_pvalue_5050: #Remove the unwanted column if requested and necessary
            dataframe=pd.read_csv(outfiletab,delimiter='\t',comment=None) #Need \t instead of \s because the output has tabs by design and spaces in column names(?)
            # adjust the dataframe and then remove the original outfile and (if requested) create a final outfile
            header5050 = 'P-value(50/50)'
            if header5050 in dataframe.columns:
                dataframe.drop(header5050, axis=1, inplace=True)
            dataframe.rename(columns={'P-value_adjusted': 'P-value'}, inplace=True)
            dataframe.to_csv(outfiletab, sep="\t", index=False)

        logger.removeHandler(logging_handler)
        logging_handler.close()
        return self.outfile
        
    #def getRandSnpSignal(self, nSnp, nInd,genphen,newseed):
    #    from numpy.random import RandomState
    #    import pysnptools.util.gensnp as gp #!!! does this work?
    #    randomstate = RandomState(newseed) 
    #    nSnp=genphen["numBackSnps"]
    #    #randsnps=self.alt_snpreader.read(RandomSnpSet(nSnp,newseed))     #this appears to be VERY slow
    #    #snps=randsnps['snps']         
    #    snps=gp.gensnps(nInd,nSnp)                        
    #    randG = snps/sp.sqrt(nSnp) #pre-process for kernel
    #    y_randG=sp.sqrt(genphen["varBack"])*randG.dot(randomstate.rand(nSnp,1))            
    #    return y_randG

    def TESTBEFOREUSINGKfromAltSnps(self, N, SNPsalt=None):        
       
        print("constructing K from all SNPs")
        t0=time.time()

        Kall=sp.zeros([N,N])
        nSnpTotal=self.alt_snpreader.snp_count
        print("reading in " +str(nSnpTotal)+ " SNPs and adding up kernels")
        #altsnps = readBED(self.bedfilealt,standardizeSNPs=True)['snps'] #loads all in to memory        
        blocksize=100
        ct=0
        ts=time.time()
        if self.alt_snpreader.snp_count<N: raise Exception("need to adjust code to handle low rank")          
        for start in range(0,self.alt_snpreader.snp_count,blocksize):                                
            ct+=blocksize
            snpSet = PositionRange(start,blocksize)
            snps = self.alt_snpreader.read(snpSet)['snps']
            import fastlmm.util.preprocess as util
            snps = util.standardize(snps)

            #print "start = {0}".format(start)
            Kall=Kall + snps.dot(snps.T)    
            t1=time.time()
            if ct % 50000==0: 
                print("read %s SNPs in %.2f seconds" % (ct, t1-ts))
                ts=time.time()
            #if ct==2: 
            #    break
            #    Kall=sp.rand((N,N))
            #    Kall=Kall.dot(Kall.T)
        Kall=Kall/sp.sqrt(self.alt_snpreader.snp_count)  
        Kall=Kall + 1e-5*sp.eye(N,N)     
        t1=time.time()
        logging.info("%.2f seconds elapsed" % (t1-t0))        
                
        return Kall

    def _saveArray(self, appendNameFile, header, values):
        outfile = utilx.appendtofilename(self.outfile, appendNameFile)
        try:
            pstutil.create_directory_if_necessary(outfile)
        except:
            logging.warn("Exception while creating directory for '{0}'. Assuming that other cluster task is creating it.".format(outfile))

        logging.info("writing to file " + outfile + ".")

        assert type(values) == sp.ndarray or type(values) == tuple, 'values can only be sp.ndarray or a tuple of sp.ndarray.'

        if type(values) == sp.ndarray:
            values = (values,)

        values = list(map( list, list(zip(*values)) ))

        with open(outfile, "w") as fp:
            fp.write('\t'.join(header) + "\n")
            for v in values:
                fp.write("{0}\n".format( '\t'.join([str(x) for x in v]) ))
    
    #!! would be nice of this was optional and if not given the OS was asked
    # required by IDistributable
    @property
    def tempdirectory(self):
        return self.outfile + ".work_directory"


    #!! need comments
    def copyinputs(self, copier):
        copier.input(self.phenofile)
        copier.input(self.alt_snpreader)
        copier.input(self.altset_list)
        if (self.altset_list2 is not None):
            copier.input(self.altset_list2)
        copier.input(self.covarfile)
        # similar code elsewhere
        if (self.filenull is not None):
            root, ext = os.path.splitext(self.filenull)
            # allow other formats?
            if ext.lower() == ".bed":
                copier.input(root + ".bed")
                copier.input(root + ".bim")
                copier.input(root + ".fam")
            else:
                copier.input(root + ".ped")
                copier.input(root + ".map")
        if (self.extractSim is not None):
            copier.input(self.extractSim)
        if (self.nullfitfile is not None):                        
            copier.input(self.nullfitfile)
        if self.genphen is not None and "varBackNullFileGen" in self.genphen:
            copier.input(self.genphen["varBackNullFileGen"]+".fam")
            copier.input(self.genphen["varBackNullFileGen"]+".bim")
            copier.input(self.genphen["varBackNullFileGen"]+".bed")
        if self.genphen is not None and "varBackNullPhenGen" in self.genphen and self.genphen["varBackNullPhenGen"] is not None:
            copier.input(self.genphen["varBackNullPhenGen"])

    #Note that the files created are not automatically copied. Instead,
    # whenever we want another file to be created, a second change must be made here so that it will be copied.
    def copyoutputs(self,copier):
        outfiletab=self.outfile
        copier.output(outfiletab)
        infofilename = utilx.appendtofilename(self.outfile,"info")
        copier.output(infofilename)
        if self.write_lrtperm and self.nperm>0:
            lrtperm_outfile = utilx.appendtofilename(self.outfile,"lrtperm")
            copier.output(lrtperm_outfile)
        if self._synthphenfile is not None:
            copier.output(self._synthphenfile)

    # =================================================================
    # END Implementation the IDistributable interface described in pysnptools.util.mapreduce1.distributable
    # =================================================================

    def _check_entries(self, entries):
        assert 'nullModel' in entries and 'altModel' in entries
        assert 'effect' in entries['nullModel'] and 'link' in entries['nullModel']
        assert 'effect' in entries['altModel'] and 'link' in entries['altModel']

        assert entries['nullModel']['effect'] in set(['fixed', 'mixed'])
        assert entries['altModel']['effect'] in set(['fixed', 'mixed'])

        assert entries['nullModel']['link'] in set(['linear', 'logistic', 'erf'])
        assert entries['altModel']['link'] in set(['linear', 'logistic', 'erf'])

        assert all( [(v in set(['effect', 'link', 'penalty', 'approx'])) for v in entries['nullModel']] ),\
               'The only allow keys for nullModel are effect, link, penalty, and approx.'

        assert all( [(v in set(['effect', 'link', 'penalty', 'approx'])) for v in entries['altModel']] ),\
               'The only allow keys for altModel are effect, link, penalty, and approx.'

       

    def _check_params_after(self):        
        self.nperm = self.test.check_nperm(self.nperm)
        if self.minsetsize is not None and self.minsetsize<1: raise Exception("minsetsize must be 1 or greater")
        if self.nullfitfile is not None: assert (self.nperm is None or self.nperm==0), "if using nullfitfile then nperm should be 0"
        assert ((self.nlocalperm is None) or self.nlocalperm==0) or ((self.nperm is None) or self.nperm==0), "cannot use both nperm and nlocalperm"
        assert self.nullfitfile is None or self.nlocalperm==0, "cannot use nullfitfile with nlocalperm"

        if self.genphen is not None:
            if 'casefrac' not in list(self.genphen.keys()): self.genphen['casefrac']=None
            if 'fracCausal' not in list(self.genphen.keys()): self.genphen['fracCausal']=1.0
            if 'numBackSnps' not in list(self.genphen.keys()): self.genphen['numBackSnps']=0
            if 'varBackNullFileGen' not in list(self.genphen.keys()): self.genphen['varBackNullFileGen']=None  
    
    def _check_params_before(self):
        pass
        #if self.test== "lrt_up" and self.filenull is None:
        #    self.test = "lrt"
                   
    def pv_from_localperm(self, iperm, result, t0, varcomp_test, y):
        '''
        This is not test specific, but is currently only implemented for LRT. No code would 
        need to be changed here for other tests, only the test-specific files elsewhere.
        '''
        assert iperm==-1, "should not be permuting for global null fit here, only local"
        assert str(self.test)=="lrt", "can only use local permutations with lrt, not with " + str(self.test)            
        assert self.altModel["effect"]=="mixed" and self.altModel["link"]=="linear", "nlocalperm only set up for linear lrt at the moment"
        assert self.nullModel["effect"]=="fixed", "nlocalperm only set up for one-kernel" 
        #assert self.__G0 is None, "only 1-kernel implemented"
        assert self.__SNPs0 is None, "only 1-kernel implemented"
        logging.info("running up to " + str(self.nlocalperm) + " local permutations")
        permstatbetter=[]
        allstat=[]
        alteqnullperm=[]
        yorig=y
        tol=0.0
        from numpy.random import RandomState
        randomstate = RandomState(self.rseed)
        checkpoint=1000
        powerneeded=2
        for pm in range(self.nlocalperm):
            if pm==checkpoint:
                logging.info('checkpointing '  + str(pm))                    
                numbetter=sp.sum(permstatbetter)
                pv=numbetter/float(pm)                                
                t1=time.time()
                logging.info("Checkpoint time elapsed=%.2f seconds" % (t1-t0))
                if pv*checkpoint>=10**powerneeded:
                    logging.info("stopping due to checkpoint")
                    break;
                checkpoint=checkpoint*2
            
            permutationIndex = utilx.generate_permutation(y.shape[0],randomstate)
            yperm=y[permutationIndex]           
            Xperm=self.__X[permutationIndex]                
                                                                   
            permresult = varcomp_test.testGupdate(yperm,Xperm,self.test)  
            allstat.append(permresult['stat'])
            alteqnullperm.append(permresult['alteqnull'])
            permstatbetter.append(permresult['stat']-result.test['stat']>tol)  
                       
        numbetter=sp.maximum(0.99,sp.sum(permstatbetter))
        pv=numbetter/float(pm) 
        result.test['pv-local']=pv
        result.test['pv']=result.test['pv']
        
        if self.fitlocal and pv*pm<10**powerneeded:
            import fastlmm.association.tests.Cv as cv
            logging.info("fitting aUD to local with " + str(pm+1) + " permutations")
            pv_adj,mixture,scale,dof=cv.lrtpvals_qqfit(nperm=self.nlocalperm, lrt=sp.array([result.test['stat']]), lrtperm=sp.array(allstat), alteqnull=sp.array([result.test['alteqnull']]), alteqnullperm=sp.array(alteqnullperm), qmax=self.qmax)
            result.test['pv-local-aUD']=pv_adj[0]
            logging.info("mixture (non-zero dof)="+str(mixture))
            logging.info("dof="+str(dof))
            logging.info("scale="+str(scale))
        else:
            result.test['pv-local-aUD']=sp.NaN
        
        logging.info("    used " + str(pm+1) + " permutations to compute p=" + str(pv) + ", p50=" + str(result.test['pv']))
        
    def G_exclude(self, i_exclude):
        if "data" in self.__SNPs0:
            G_exclude = self.__SNPs0["data"]["snps"][:,i_exclude]
        else:
            snp_names = self.__SNPs0["reader"].rs[i_exclude]
            snp_set = SnpAndSetName('G_exclude', snp_names)
            G_exclude = self.__SNPs0["reader"].read(snp_set)['snps']
            G_exclude = util.standardize(G_exclude)
            #normalize
            pass
        G_exclude/=sp.sqrt(self.__SNPs0["num_snps"])
        return G_exclude

    def run_test(self, SNPs1, G1, y, altset, iset, ichrm, iposrange, iperm = -1, varcomp_test=None):
        '''
        This function does the main work of the class, and also reads in the SNPs for the alternative model.
        It is called (via a lambda) inside the loops found in 'generate_sequence'.

        Input:
            altset - a set of snps
            iset - index to altset
            iperm - index to permutation (-1 means no permutation)
            varcomp -if not None, assume that it is the correct one for this test, and does not re-compute anything
        Output:
            a list (often just one) of
                instances of the Result class, varcomp (for caching)
        '''
        t0=time.time()            

        self.run_once() #load files, etc. -- stuff we only want to do once no matter how many times we call 'run_test' and then 'reduce.        

        logging.info("Working on permutation index #{0} (where -1 means no permutation) out of {1} total permutations...".format(iperm, self.nperm))
        logging.info("\taltset {0}, {1} of {2}  ".format(altset, iset, len(self.altsetlist_filtbysnps)))
                      
        result = Result(iperm=iperm,iset=iset,setname=str(altset), ichrm=ichrm, iposrange=iposrange)
         
        if varcomp_test is None or not self.cache_from_perm:            
            if G1.shape[1]==0:
                logging.info( "no SNPS in set " + setname )
                result=None
                return [result]
            if sp.isnan(G1.sum()): raise Exception("found missing values in test SNPs that remain after intersection for " + str(altset))

            result.setsize = SNPs1['snps'].shape[1]
                
            logging.info(" (" + str(result.setsize) + " SNPs)")              

            if self.filenull is not None:
                if 'data' in self.__SNPs0:
                    pos0=self.__SNPs0['data']['pos']
                else:
                    pos0=self.__SNPs0['snp_set'].pos
                i_exclude =  utilx.excludeinds(pos0, SNPs1['pos'], mindist = self.mindist,idist = self.idist)
                result.nexclude = i_exclude.sum()
                logging.info("numExcluded=" + str(result.nexclude) + "")
            else:
                i_exclude = None
                result.nexclude=0;


        if self.permute is not None: #permute the data to test for type I error
                permutationIndex = utilx.generate_permutation(SNPs1['snps'].shape[0],self.mainseed ^ self.permute)      
        if iperm >= 0 :  #permute the data to create Null-only P-values for lrt null fitting
            eachseed=utilx.combineseeds(self.calseed,iperm)            
            newseed = self.mainseed ^ eachseed  

            if self.permute is not None :
                #Add a left shift so that iperm=1 and self.permute=2 gives a different newseed than 
                #iperm=2 and self.permute=1
                newseed = (newseed << 1) ^ self.permute
            permutationIndex = utilx.generate_permutation(SNPs1['snps'].shape[0], newseed)
        if varcomp_test is None or not self.cache_from_perm:
            #need to make this caching smarter for when background kernel changes in every test (e.g. interactions)
            if (self.permute is not None) or (iperm >=0):
                G1=G1[permutationIndex]

            #if result.nexclude:
            #    logging.info(" (computing needed null info anew) ")
            #    G0_to_use = self.__SNPs0#['data']['snps'][:,~i_exclude]/SP.sqrt(self.__SNPs0['data']['snps'][:,~i_exclude].shape[1])   
            #else: 
            #    G0_to_use=self.__G0
        
            if (str(self.test)!="lrt_up") and (self.genphen is not None) and (not self.genphen['once']) or ((str(self.test)!="lrt_up") and result.nexclude>0) or self.__varcomp_test is None:
                #need to re-construct the test (can't cache)
                #note that if result.nexclude kicked in, then this would happen in clause above anyhow                               
                varcomp_test = self.varcomp_test_setup(y,self.__SNPs0,i_exclude)
            else: #use cached values                
                #G0_to_use=self.__G0
                varcomp_test=self.__varcomp_test
                #varcomp_test.i_exclude=i_exclude
                
                logging.info(" (using cached null info)")
            
            #create G_exclude                         
            if (str(self.test)=="lrt_up" and result.nexclude>0):
                G_exclude = self.G_exclude(i_exclude)
            else:
                G_exclude = None            
         
            result.test = varcomp_test.testG(G1,self.test, i_exclude=i_exclude,G_exclude=G_exclude)             
            logging.info("p=%.2e",result.test['pv'])                   

            # do the permutations here, rather than as they would normally be done with a seperate call to             
            if self.nlocalperm and self.nlocalperm>0:
                self.pv_from_localperm(iperm, result, t0, varcomp_test, y)
        else: #caching for 1-kernel LRT (save expensive stuff from non-permuted run)
            #currently not used, because need to be able to pass back values to work_sequence for caching
            assert iperm!=-1, "should be a permutation run"
            assert str(self.test)=="lrt", "can only use lrt, here now"
            assert self.altModel["effect"]=="mixed" and self.altModel["link"]=="linear", "caching only set up for linear lrt at the moment"
            assert self.nullModel["effect"]=="fixed", "caching only set up for one-kernel" 
            assert self.__G0 is None, "caching only 1-kernel implemented, otherwise need to permute G0 as well here"

            #permute X, y instead of G1 so can use caching                  
            yperm=y[permutationIndex]                
            Xperm=self.__X[permutationIndex]       
            result = varcomp_test.testGupdate(yperm,Xperm,self.test)  
                                                  
        t1=time.time()
        logging.info("%.2f seconds elapsed in run_test" % (t1-t0))
                
        return [result]

    def TESTBEFOREUSING_run_interactiontest(self,altset,iset,altset2,iset2,iperm = -1):
        '''
        This function does the main work of the class (when interaction tests are running).
        It is called (via a lambda) inside the loops found in 'generate_sequence'

        Input:
            altset - a set of snps
            iset - index to altset
            iperm - index to permutation (-1 means no permutation)
        Output:
            an instance of the Result class
        '''

        t0=time.time()

        self.run_once() #load files, etc. -- stuff we only want to do once no matter how many times we call 'run_test' and then 'reduce.

        logging.info("Working on permutation index #{0} out of {1} total permutations...".format(iperm, self.nperm))
        logging.info("\taltset {0}, {1} of {2}  ".format(altset, iset, len(self.altsetlist_filtbysnps)))
        self.stdout.flush()

        result = PairResult(iperm=iperm,iset=iset,setname=str(altset),iset2=iset2,setname2=str(altset2))

        #read the alternative model SNPs (for all sets) from disk.
        SNPs1a = altset.read()      #first set
        import fastlmm.util.preprocess as util
        SNPs1a['snps'] = util.standardize(SNPs1a['snps'])

        SNPs1b = altset2.read()     #second set
        SNPs1b['snps'] = util.standardize(SNPs1b['snps'])

        SNPs1=dict(SNPs1a)
        SNPs1.update(SNPs1b) #concatenate the SNP dictionaries
  
        G1 = SNPs1['snps']/sp.sqrt(SNPs1['snps'].shape[1])
        if G1.shape[1]==0: raise Exception("no SNPs to test")
        result.setsize = SNPs1['snps'].shape[1]

        logging.info(" (" + str(result.setsize) + " SNPs)")

        if self.permute >= 0 :
            permutationIndex = utilx.generate_permutation(SNPs1['snps'].shape[0],self.rseed ^ self.permute)
            G1=G1[permutationIndex]

        if iperm >= 0 :
            newseed = self.rseed ^ iperm
            if self.permute >= 0 :
                #Add a left shift so that iperm=1 and self.permute=2 gives a different newseed than iperm=2 and self.permute=1
                newseed = (newseed << 1) ^ self.permute
            permutationIndex = utilx.generate_permutation(SNPs1['snps'].shape[0], newseed)
            #permute the data to create Null-only P-values
            G1=G1[permutationIndex]

        if self.filenull is not None:
            i_exclude =  utilx.excludeinds(self.__SNPs0['data']['pos'], SNPs1['pos'], mindist = self.mindist,idist = self.idist)
            result.nexclude = i_exclude.sum()
            logging.info("numExcluded=" + str(result.nexclude))
        else:
            result.nexclude=0;

        null_model_changed=(result.nexclude>0)

        #need to make this caching smarter for when background kernel changes in every test (e.g. interactions)
        if null_model_changed:
            logging.info(" (computing needed null info anew) ")
            G0_to_use = self.__SNPs0['data']['snps'][:,~i_exclude]/SP.sqrt(self.__SNPs0['data']['snps'][:,~i_exclude].shape[1])  #excluded SNPs
            null_model=None
        else: #use cached values
            G0_to_use=self.__G0
            null_model=self.__varcomp_test           
            logging.info(" (using cached null info)")

        [result.pv,result.lik0,result.lik1] = self.test.pv_etc(filenull, G0_to_use, G1, y, x, appendBias, null_model, self.__varcomp_test, forcefullrank)

        # where is result.test used?
        result.test=self.test #probably not the best way to do this, but hacking it for now

        t1=time.time()
        logging.info("%.2f seconds elapsed" % (t1-t0))

        return result
  
    def setSNPs0(self):        
        logging.info("Reading SNPs0")
        root, ext = os.path.splitext(self.filenull)
        if (self.autoselect):
            #raise Exception("this autoselect option has been removed. please use the stand-alone FastLmm-SELECT in this same download")
            assert not self.forcefullrank, "cannot currently force full rank with autoselect=True"
            if self.extractSim is not None : raise Exception("extractSim not supported with autoselect")
            #call C++ version of FaST-LMM select to determine background Covariance matrix of the LMM
            osname = sys.platform
            dir = os.path.split(__file__)[0] #__file__ is the pathname of the file from which the module
            os.environ["FastLmmUseAnyMklLib"] = "1"
            if (osname.find("win") >= 0):    #         was loaded, if it was loaded from a file
                fastlmmpath = os.path.join(dir,"Fastlmm_autoselect", "fastlmmc.exe")
            elif (osname.find("linux") >= 0):
                fastlmmpath = os.path.join(dir,"Fastlmm_autoselect", "fastlmmc")
                import stat
                st = os.stat(fastlmmpath)
                os.chmod(fastlmmpath, st.st_mode | stat.S_IEXEC)
            else:
                logging.info('\n\n unsupported operating system!')
            if not os.path.isfile(fastlmmpath) : raise Exception("Expect file {0}".format(fastlmmpath))
            logging.info("fastlmmC path=" + fastlmmpath)

            if (not os.path.exists('select')):
                subprocess.call("mkdir select", shell=True)

            with TemporaryFile() as stdin_temp: #An empty to file from which to read stdin (This allows this to run even when there is no normal stdin defined)
                with TemporaryFile() as stdout_temp:
                    if (self.covarfile == None):
                        subprocess.call(fastlmmpath + ' -autoselect select/auto -bfilesim ' + root + ' -pheno ' + self.phenofile + ' -pedOutFile ' + "select/null", shell=True, stdout=stdout_temp, stderr=subprocess.STDOUT, stdin=stdin_temp)
                    else:
                        subprocess.call(fastlmmpath + ' -autoselect select/auto -bfilesim ' + root + ' -pheno ' + self.phenofile + ' -covar ' + self.covarfile + ' -pedOutFile ' + "select/null", shell=True, stdout=stdout_temp, stderr=subprocess.STDOUT, stdin=stdin_temp)
                    stdout_temp.seek(0)
                    stdout_string = stdout_temp.read()
            logging.info(stdout_string)

            self.__SNPs0 = {
                "data":readPED("select/null"),
                "filename":"select/null",                
                }
            self.__SNPs0["original_iids"] = self.__SNPs0["data"]["iid"]
            self.__SNPs0["num_snps"] = self.__SNPs0["data"]["snps"].shape[1]
        else:                   
            if ext.lower() == ".bed":
                snp_set = CreateSnpSetReaderForFileNull(self.extractSim)  
                bed = Bed(root)
                snp_set_bed = snp_set.addbed(bed)
                num_snps = len(snp_set_bed)                
                self.__SNPs0={
                    "snp_set":snp_set_bed,
                    "reader":snp_set_bed.bed,
                    "filename":root,
                    #"data":readBED(root, snp_set=snp_set)
                    }
                self.__SNPs0["original_iids"] = self.__SNPs0["reader"].original_iids
                self.__SNPs0["num_snps"] = num_snps
                #import fastlmm.util.preprocess as util
                #self.__SNPs0['data']['snps'] = util.standardize(self.__SNPs0['data']['snps'])
            else:
                if self.extractSim is not None : raise Exception("extractSim not supported with filenull in ped format, please use bed format")
                self.__SNPs0 = {"data":readPED(root),
                                "filename":root}
                self.__SNPs0["original_iids"] = self.__SNPs0["data"]["iid"]
                self.__SNPs0["num_snps"] = self.__SNPs0["data"]["snps"].shape[1]

    def varcomp_test_setup(self,y,SNPs0=None, i_exclude=None): 
        #in case we exclude all SNPs (needed for score, not for lrt, not sure about lrt_up)
        nullModelTmp=self.nullModel.copy()           
        if (i_exclude is not None) and SNPs0['num_snps']==i_exclude.sum():
            excluded_all_snps = True           
            nullModelTmp['effect']='fixed'
        else:
            excluded_all_snps = False        

        if self.filenull is not None and (not excluded_all_snps): #aka two-kernel after exclusion has been done        
            varcomp_test = self.test.construct(y,self.__X, forcefullrank = self.forcefullrank,
                             SNPs0 = SNPs0, i_exclude = i_exclude, nullModel = self.nullModel, altModel = self.altModel,
                             scoring = self.scoring, greater_is_better = self.greater_is_better)
        else: #no background kernel
            varcomp_test = self.test.construct_no_backgound_kernel(y,self.__X,
                             forcefullrank = self.forcefullrank, nullModel = nullModelTmp,
                             altModel = self.altModel, scoring = self.scoring,
                             greater_is_better = self.greater_is_better)        
        return varcomp_test

    #so that it only ever gets changed once, in run_once
    @property
    def rseed(self):
        assert self._seed is not None
        return self._seed

    @property
    def mainseed(self):
        '''used to xor indexes with to get seeds for other purposes'''
        return 34343
       
    def run_once(self):
        '''
        Loading datasets (but not SNPs for alternative), etc. that we want to do only once no mater how many times we call 'run_test' and then perhaps 'reduce'.        '''         
               
        if self.calseed is not None:            
            self._seed=utilx.combineseeds(self.calseed,self.mainseed)                        
            #self._seed = self.calseed ^ self.mainseed
        else:
            self._seed = self.mainseed

        if (self._ran_once):
            return
        #self.__snapshot = self.__repr__() # must be before seeting _ran_once
        self._ran_once = True

        logging.info("Running Once")

        self._check_params_before() #must be called before test object Lrt(), Cv(), Sc() is created       
        #!! make this nicer
        if self.test=="lrt":
            self.test = Lrt()
        elif self.test=="lrt_up":
            self.test = lrt.LRT_up()
        elif self.test=="cv":
            self.test = Cv()
        elif self.test=="sc_davies":
            self.test = Sc("davies")
        elif self.test=="sc_mom":
            self.test = Sc("mom")
        else:
            raise Exception("Don't know how to construct {0}".format(self.test))        
        self._check_params_after() #must be called after test object Lrt(), Cv(), Sc() is created         

        phenhead,phentail=os.path.split(self.phenofile)
        logging.info("Reading Pheno: " +  phentail + "")        
        pheno = loadPhen(filename = self.phenofile,missing='NaN')
        pheno['vals']=pheno['vals'][:,self.mpheno-1]#use -1 so compatible with C++ version
        #pheno['header']=pheno['header'][self.mpheno-1] #header is empty 
        goodind=sp.logical_not(sp.isnan(pheno['vals']))
        pheno['vals']=pheno['vals'][goodind]
        pheno['iid']=pheno['iid'][goodind,:]        
           
        if self.genphen is not None:
            #for synthetic data generation--this is the background signal added to the phenotype
            if self.genphen["varBackNullFileGen"] is not None:
                self.__varBackNullSnpsGen=readBED(self.genphen["varBackNullFileGen"])
                import fastlmm.util.preprocess as util
                self.__varBackNullSnpsGen['snps'] = util.standardize(self.__varBackNullSnpsGen['snps'])
            else:
                self.__varBackNullSnpsGen=None

        if self.filenull is not None:
            self.setSNPs0()
            logging.info("Filter SNPs0")
            #prelim intersection so don't read in all null SNP iids if way more than in pheno; now redundant
            #filter(pheno, self.__SNPs0['data'])
        else:
            if self.extractSim is not None : raise Exception("extractSim should be specified only if filenull is specified.")
            self.__SNPs0=None
            #self.__G0=None

        if self.covarfile==None:
            covar=None
        else:
            covar=loadPhen(self.covarfile,missing='NaN')
            if self.covarimp=='standardize':
                covar['vals'],fracmissing=utilx.standardize_col(covar['vals'])
            elif self.covarimp is None:
                pass
            else:
                raise Exception("covarimp=" + self.covarimp + " not implemented")          

        covar,pheno,indarr=self.intersect_data(covar, pheno)
        N=pheno['vals'].shape[0];
        if self.covarfile==None:
            self.__X = sp.ones((N,1))
        else:
            #check for covariates which are constant, as the lmm code crashes on these            
            badind = utilx.indof_constfeatures(covar['vals'], axis=0)
            if len(badind)>0:
                raise Exception("found constant covariates with indexes: %s: please remove, or modify code here to do so" % badind)
            self.__X=sp.hstack((sp.ones((N,1)),covar['vals']))
        self.__y = pheno['vals']

        if not covar is None and sp.isnan(covar['vals'].sum()): raise Exception("found missing values in covariates file that remain after intersection")
        if sp.isnan(self.__y.sum()): raise Exception("found missing values in phenotype file that remain after intersection")
        #if hasattr(self,'__G0') and not self.__G0 is None and sp.isnan(self.__G0.sum())): raise Exception("found missing values in background SNPs that remain after intersection")

        #creating sets from set defn files. Looks at bed file to filter down the SNPs to only those present in the bed file
        self.altset_list,self.altsetlist_filtbysnps = self.create_altsetlist_filtbysnps(self.altset_list,self.alt_snpreader)
        if self.altset_list2 is not None:
                self.altset_list2,self.__altsetlist2_filtbysnps = self.create_altsetlist_filtbysnps(self.altset_list2,self.alt_snpreader)

        if self.npermabs is not None and self.nperm is not None:
            assert self.nperm is None, "cannot use both nperm and npermabs"
        if self.npermabs is not None:
            num_tests = len(self.altsetlist_filtbysnps)
            self.nperm = int(sp.ceil(self.npermabs/num_tests))
            logging.info("using npermabs=%i, found %i tests, so using %i permutations per test" % (self.npermabs, num_tests, self.nperm))            
            self.npermabs = None
        
        logging.info("------------------------------------------------")
        logging.info("Found " + str(self.__X.shape[0]) +  " individuals")
        logging.info("Found " + str(self.__X.shape[1]) +  " covariates (including bias)")
        if self.__SNPs0 is not None:
            logging.info("Found " + str(self.__SNPs0['num_snps']) +  " SNPs for null kernel")
        if (len(self.__y.shape)>1): nPhen=self.__y.shape[1]
        else: nPhen=1
        logging.info("Found " + str(nPhen) +  " phenotypes")
        if self.filenull != None:
            logging.info("Running test " +  str(self.test) + " with 1 background kernel")
        else:
            logging.info("Running test " +  str(self.test)  + " with no background kernel")
        #CL3_28_14:The following never evaluates True, as it should be #if str(self.test)=="lrt":
        #but the body is buggy, so it would crash
        #if self.test=="lrt":
        #    logging.info("nullmodel: " + "link=" + self.nullModel['link'] + ", " 
        #                          + "approx=" + self.nullModel['approx'] + ", "
        #                          + "pen="+ self.nullModel['penalty'])
        #    logging.info("altmodel: " + "link=" + self.altModel['link'] + ", " 
        #                          + "approx=" + self.altModel['approx'] + ", "
        #                          + "pen="+ self.altModel['penalty'])        
        logging.info("Found " + str(self.work_count/(self.nperm+1)) + " tests to run (" + str(self.work_count) + " work items, including " + str(self.nperm) + " permutations)")
        if self.nullfitfile is not None:
            logging.info("nullfitfile=" + self.nullfitfile)
        if self.altset_list2 is not None:
            logging.info("(running paired set tests)")
        else:
            logging.info("(running single set tests)")        
        if self.minsetsize is not None and self.maxsetsize is not None:
            logging.info("min set size=%i, max set size=%i", self.minsetsize, self.maxsetsize) 
        if self.mpheno is not None: logging.info("mpheno=%i", self.mpheno)
        if self.extractSim is not None: logging.info("extractSim=%s",self.extractSim)
        logging.info("------------------------------------------------")

        logging.info("Creating null model... ")
        #cache whatever is needed for each case (sometimes likelihood, sometimes rotated items, sometimes nothing)
        t0=time.time()
        
        #if self.filenull is not None: #aka two-kernel
        #    self.__G0 = self.__SNPs0['data']['snps']/sp.sqrt(self.__SNPs0['data']['snps'].shape[1])        
        #else:
        #    self.__G0=None
        
        #for some models, this caches important and expensive information
        #as appropriate, gets re-computed in run_test()        
        if self.genphen is None: #or self.genphen['once']: THIS CAUSES REAL DATA TO GET USED         
            #self.__varcomp_test=self.varcomp_test_setup(self.__y,self.__G0)            
            self.__varcomp_test=self.varcomp_test_setup(self.__y,self.__SNPs0)
        #elif self.genphen is not None and self.__G0 is not None:
        elif self.genphen is not None and self.__SNPs0 is not None:
            raise Exception("this will run if you comment out this line, but I think will be inefficient because it's not caching null info across each gene-set specific phenotypes even when G0 remains constant across these")
            self.__varcomp_test=None
        else:
            # single kernel synthetic, fine to re-do null model computations each time
            self.__varcomp_test=None

        t1=time.time()
        logging.info("done. %.2f seconds elapsed" % (t1-t0))

    def check_id_order(self, covar, pheno):        
        ids_pheno=pheno['iid']
        ids_SNPs=self.alt_snpreader.original_iids
        if (len(ids_pheno)!=len(ids_SNPs)) or (not sp.all(ids_pheno==ids_SNPs)):
          return False
        if self.covarfile is not None:
            ids_X=covar['iid']
            if (len(ids_pheno)!=len(ids_X)) or (not sp.all(ids_pheno==ids_X)): return False
        if self.filenull is not None:
            ids_nullSNPs=self.__SNPs0['original_iids']
            if not sp.all(ids_pheno==ids_nullSNPs): return False
        if hasattr(self,"__varBackNullSnpsGen") and self.__varBackNullSnpsGen is not None: #for synthetic data generation            
            ids_SNPgen=self.__varBackNullSnpsGen["iid"]
            if not sp.all(ids_pheno==ids_SNPgen): return False
        return True

    def intersect_data(self, covar, pheno):
        if self.check_id_order(covar,pheno):
            indarr = SP.arange(pheno['iid'].shape[0])
            logging.info(str(indarr.shape[0]) + " IDs match up across data sets")
        else:            
            logging.info("IDs do not match up, so intersecting the data over individuals")            
            if self.filenull is not None:
                nullsnpids=self.__SNPs0['original_iids']
            else:
                nullsnpids=None
            if self.covarfile is not None:
                covarids=covar['iid']
            else:
                covarids=None
            if hasattr(self,"__varBackNullSnpsGen") and self.__varBackNullSnpsGen is not None:
                varbacknullsnp=self.__varBackNullSnpsGen['iid']
            else:
                varbacknullsnp=None

            #the order of inputs here is reflected in the reordering indexes below of 0,1,2,3,4
            import warnings
            #!!warnings.warn("This intersect_ids is deprecated. Pysnptools includes newer versions of intersect_ids", DeprecationWarning)
            indarr=utilx.intersect_ids([pheno['iid'],self.alt_snpreader.original_iids,covarids,nullsnpids,varbacknullsnp])            
            assert indarr.shape[0]>0, "no individuals remain after intersection, check that ids match in files"
            #sort the indexes so that SNPs ids in their original order (and 
            #therefore we have to move things around in memory the least amount)
            #[indarr[:,3] holds the SNP order    
            sortind=sp.argsort(indarr[:,3])
            indarr=indarr[sortind]

            pheno['iid']=pheno['iid'][indarr[:,0]]
            pheno['vals']=pheno['vals'][indarr[:,0]]

            self.alt_snpreader.ind_used=indarr[:,1]

            if self.covarfile is not None:
                covar['iid']=covar['iid'][indarr[:,2]]
                covar['vals']=covar['vals'][indarr[:,2]]

            if self.filenull is not None:  
                #copy in memory to make later access faster (presumably?):
                #self.__SNPs0['iid']=sp.copy(self.__SNPs0['iid'][indarr[:,3]])
                #self.__SNPs0['snps']=sp.copy(self.__SNPs0['snps'][indarr[:,3]])
                #self.__SNPs0['data']['iid']=self.__SNPs0['data']['iid'][indarr[:,3]]
                if 'data' in self.__SNPs0:
                    self.__SNPs0['data']['iid'] = self.__SNPs0['data']['iid'][indarr[:,3]]
                    self.__SNPs0['data']['snps'] = self.__SNPs0['data']['snps'][indarr[:,3]]
                else:
                    self.__SNPs0['reader'].ind_used = indarr[:,3]
                #self.__SNPs0['data']['snps']=self.__SNPs0['data']['snps'][indarr[:,3]]

            if hasattr(self,"_varBackNullSnpsGen") and self.__varBackNullSnpsGen is not None:
                self.__varBackNullSnpsGen['iid']=self.__varBackNullSnpsGen['iid'][indarr[:,4]]
                self.__varBackNullSnpsGen['snps']=self.__varBackNullSnpsGen['snps'][indarr[:,4]]

            logging.info(str(indarr.shape[0]) + " ids left")
        
        return covar,pheno,indarr


    def create_altsetlist_filtbysnps(self, altset_list, altsetbed):
        '''
        returns: altset_list, altsetlist_filtbysnps
        'altset_list' is the raw set defintion read in from file
        'altsetlist_filtbysnps' is the result of filtering altset_list by SNPsin the specified altsetbed SNP file
        Additionally, only allows sets specified by sets are allowed (if this is not None)
        '''
        if isinstance(altset_list, str):
            altset_list = SnpAndSetNameCollection(altset_list)

        if self.sets is not None:  #filter by those sets specified on the command line, stored in self.sets
            altset_list=Subset(altset_list, self.sets)

        if self.minsetsize is not None or self.maxsetsize is not None:
            altset_list=MinMaxSetSize(altset_list,minsetsize=self.minsetsize,maxsetsize=self.maxsetsize) #would be nice if applying 1,None wouldn't wrap at all (becuase that is the identity)

        altsetlist_filtbysnps = altset_list.addbed(altsetbed)

        if len(altsetlist_filtbysnps)==0: raise Exception("Expect altset_list to contain at least one set")

        return altset_list, altsetlist_filtbysnps

    # overridden to provide a nice name on the cluster.
    def __str__(self):
        if self.outfile == None:
            return self.__class__.__name__
        else:
            return "{0} {1}".format(self.__class__.__name__, self.outfile)

    def __repr__(self):
        import io
        fp = io.StringIO() if sys.version_info >= (3,0) else io.BytesIO()
        fp.write("{0}(\n".format(self.__class__.__name__))
        varlist = []
        for f in dir(self):
            if f.startswith("_"): # remove items that start with '_'
                continue
            if type(self.__class__.__dict__.get(f,None)) is property: # remove @properties
                continue
            if callable(getattr(self, f)): # remove methods
                continue
            varlist.append(f)
        for var in varlist[:-1]: #all but last
            fp.write("\t{0}={1},\n".format(var, getattr(self, var).__repr__()))
        var = varlist[-1] # last
        fp.write("\t{0}={1})\n".format(var, getattr(self, var).__repr__()))
        result = fp.getvalue()
        fp.close()
        return result

def CreateSnpSetReaderForFileNull(snp_set):
    if (snp_set is None):
        return AllSnps()
    if isinstance(snp_set, str):        
        return SnpsFromFile(snp_set)
    return snp_set



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("last modified: %s" % time.ctime(os.path.getmtime(__file__)))
    logging.info("-------------------------------------------------")       
    
    # Parse the command line
    if len(sys.argv) == 2:
        sys.argv.append("Local()")
    if len(sys.argv) != 3:
        logging.info("Usage:python FastLmmSet.py <inputfile> [optional <runner>]")
        sys.exit(1)
    file,infile,runner_string = sys.argv
    if not os.path.exists(infile): raise Exception("Fatal Error: " + infile + " does not exist")

    #Run the job specified in infile    
    from fastlmm.association.FastLmmSet import *
    exec("runner = " + runner_string)
    d = open(infile).read()    
    exec(d)     
          
    tt0=time.time()    
    runner.run(distributable)    
    tt1=time.time()
    logging.info("Final elapsed time for all processing is %.2f seconds" % (tt1-tt0))

