from __future__ import absolute_import
from __future__ import print_function
from fastlmm.association.FastLmmSet import *
#from pysnptools.util.mapreduce1.distributable import *
from pysnptools.util.mapreduce1.runner import *
import os
import sys
import time
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("last modified: %s" % time.ctime(os.path.getmtime(__file__)))
    logging.info("-------------------------------------------------")
    
    # Parse the command line
    if len(sys.argv) == 2:
        sys.argv.append("Local()")
    if len(sys.argv) != 3:
        if len(sys.argv)!=5 and not "chrm" in sys.argv:
            logging.info("Usage:python FastLmmSetLOOC.py <inputfile> [optional <runner>]")
            sys.exit(1)
    file,infile,runner_string = sys.argv[0:3]
    if not os.path.exists(infile): raise Exception("Fatal Error: " + infile + " does not exist")

    #Run the job specified in infile
    from fastlmm.association.FastLmmSet import *
    exec("runner = " + runner_string)
    d = open(infile).read()    
    exec(d)     
    
    if "chrm" in sys.argv:
        i = sys.argv.index("chrm")
        mychrom = int(sys.argv[i+1])            
        chrmset = [mychrom]        
    else: raise Exception("please specify 'chrm' in the input file")

    for chr in chrmset:
        print("doing leave-one-out chromosome")
        print("chrm: " + str(chr))
        tt0=time.time()

        if bedfilealt.rfind("%i") > -1:
            bedfilealtnew = bedfilealt % chr  
        else: bedfilealtnew = bedfilealt
        if outfile.rfind("%i") > -1:
            outfile = outfile % chr           
        if filenull is not None and filenull.rfind("%i") > -1:
            filenullnew = filenull % chr              
        else: filenullnew = filenull       

        distributable =  FastLmmSet(                       
            phenofile = phenofile,
            outfile = outfile,
            filenull = filenullnew,
            extractSim = extractSim,
            autoselect = False,
            bedfilealt = bedfilealtnew,      
            altset_list =  altset_list,
            covarfile  = covarfile,
            mpheno=mpheno,
            mindist = mindist,   
            idist=idist,       
            nperm =nperm,            #for calibration of LRT test statistic
            test=test,     
            nullfit=nullfit,
            qmax=qmax,      
            datestamp=datestamp,
            nullModel=nullModel,
            altModel=altModel,
            genphen=None,            #for synthetic experiments
            permute=permute,         #to permute SNPs being tested: set to None on real data     
            write_lrtperm=write_lrtperm,     
            nullfitfile=nullfitfile, #if cached calibration runs to re-use (no need on real data)
            sets = sets,
            calseed=calseed,         #for different random seeds on null model permutation calibration
            minsetsize=minsetsize,
            maxsetsize=maxsetsize,
            covarimp=covarimp,       #if want to impute the covariates       
        )
                
        runner.run(distributable)    
        tt1=time.time()
        logging.info("Final elapsed time for all processing is %.2f seconds" % (tt1-tt0))


