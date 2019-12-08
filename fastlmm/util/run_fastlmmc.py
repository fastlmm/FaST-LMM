from __future__ import absolute_import
from __future__ import print_function
import subprocess 
import os 
import scipy
import scipy.io as IO
import sys
import logging

 # methods to call FaSTLMM-C++ Code


def runLIN(bfile,pheno,outFilename,fastlmm_path=None,**kwargs):
    '''    
    use linear model for filtering SNPs

    --------------------------------------------------------------------------
    Input:
    bfile       basename for PLINK's binary .bed,.fam and .bin files
    pheno       name of phenotype file
    outDir      folder in which results flie is save
    outFilename the name of the output file
    topKbyLinReg save topK snps in file
    recode      boolean. if true, recode topK snps as binary plink files
    --------------------------------------------------------------------------
    Output
    linRegFile  file containing list of topK snps
    bfileTopK   PLINK binary file containing the topK snps
    --------------------------------------------------------------------------
    '''
    run(bfile,pheno,out=outFilename,linreg=True,fastlmm_path=fastlmm_path,**kwargs)



def runFASTLMM(bfile,pheno,outDir,outFilename,methodsName=None,**kwargs):
    '''
    runs FASTLMM code
    --------------------------------------------------------------------------
    Input:
    bfile       basename for PLINK's binary .bed,.fam and .bin files
    pheno       name of phenotype file
    outDir      folder in which results flie is save
    outFilename the name of the output file
   
    for more information, we refer to the user-manual of fast-lmm
    '''
    out = '%s/%s.%s.out.txt'%(outDir,outFilename,methodsName)
    run(bfile,pheno,out=out,**kwargs)


def runLMMSELECT(bfile,pheno,outDir,outFilename,autoSelectCriterionMSE=True,**kwargs):
    '''
    runs LMM-Select code

    --------------------------------------------------------------------------
    Input:
    bfile       basename for PLINK's binary .bed,.fam and .bin files
    pheno       name of phenotype file
    outDir      folder in which results flie is save
    outFilename the name of the output file
    autoSelectCriterionMSE  directs AutoSelect ot use out-of-sample mean-squared error for the selection criterion. 
                            Otherwise out-of-sample log likelihood is used
   
    for more information, we refer to the user-manual of fast-lmm
    '''
   # 1. determine number of SNPs in the kernel
    autoSelect = '%s/%s.LMMSELECT.aoselect'%(outDir,outFilename)
    run(bfile,pheno, autoSelect=autoSelect,autoSelectCriterionMSE=autoSelectCriterionMSE,**kwargs)
    # 2. run GWAS
    out = '%s/%s.%s.out.txt'%(outDir,outFilename,'LMMSELECT')
    extractSim = '%s/%s.LMMSELECT.aoselect.snps.txt'%(outDir,outFilename)
    run(bfile,pheno,out=out,extractSim=extractSim,**kwargs)
 

def run(bfile=None,pheno=None,bfileSim=None,sim=None,linreg=None,covar=None,out=None,optLogdelta=None,extract=None,extractSim=None,autoSelect=None,autoSelectCriterionMSE=False,
        excludeByPosition=None,excludeByGeneticDistance=None,eigen=None,eigenOut=None,maxThreads=None,simOut=None,fastlmm_path=None,topKbyLinReg=None,numJobs=None,thisJob=None,REML=False):
    '''
    interface for FastLMM software

    --------------------------------------------------------------------------
    Input:
    bfile       basename for PLINK's binary .bed,.fam and .bin files
    pheno       name of phenotype file
    bfileSim    basename for PLINK's binary files for building genetic similarity
    sim         specifies that genetic similarities are to be read directly from this file
    linreg      specifies that linear regression will be performed. when this option is used, no genetic similarities should be specified (boolean)
    covar       optional file containing the covariates
    out         the name of the output file
    optLogdelta if set, delta is not optimized
    extract     FaSTLMM will only analyze the SNPs explicitly listed in the filename
    extractSIm  FastLMM will only use the SNPs explicitly listed for computing genetic similarity
    autoSelect  determines the SNPs to be included in the similarity matrix. When this option is used, GWAS is not run. SNPs are written to filename.snps.txt
                and statistics to filename.xval.txt
    autoSelectCriterionMSE  directs AutoSelect to use out-of-sample mean-squared error for the selection criterion. Otherwise out-of-sample log likelihood is used
    excludeByPosition       excludes the SNP tested and those within this distance from the genetic similarity matrix
    excludeByGeneticDistance excludes the SNP tested and those within this distance from the genetic similarity matrix
    eigen       load the spectral decomposition object from the directory name
    eigenOut    save the spectral decomposition object to the directory name
    maxThreads  suggests the level of parallelism to use
    simOut      specifies that genetic similarities are to be written to this file
    topKbyLinRegdirects AutoSelect to use only the top <int> SNPs, as determined by linear regression, while selecting SNPs
    for more information, we refer to the user-manual of fast-lmm
    '''
    os.environ["FastLmmUseAnyMklLib"] = "1"
    logging.info('Run FAST-LMM')   

    osname = sys.platform
    if (osname.find("win") >= 0):    #         was loaded, if it was loaded from a file
        fastlmmpath = os.path.join(fastlmm_path, "fastlmmc.exe")
    elif (osname.find("linux") >= 0):
        fastlmmpath = os.path.join(fastlmm_path, "fastlmmc")
        import stat
        st = os.stat(fastlmmpath)
        os.chmod(fastlmmpath, st.st_mode | stat.S_IEXEC)
    else:
        logging.info('\n\n unsupported operating system!')
    if not os.path.isfile(fastlmmpath) : raise Exception("Expect file {0}".format(fastlmmpath))
    logging.info("fastlmmC path=" + fastlmmpath)


    cmd = '\"%s\" -simLearnType Once'%(fastlmmpath) # change that! logdelta is always only optimized on the null model
    
    if bfile!=None:
        assert os.path.exists(bfile + '.bed'), 'ouch, bfile is missing: %s'%bfile
        assert os.path.exists(bfile + '.bim'), 'ouch, bfile is missing: %s'%bfile
        assert os.path.exists(bfile + '.fam'), 'ouch, bfile is missing: %s'%bfile
        cmd += ' -bfile %s'%bfile
    if pheno!=None:
        assert os.path.exists(pheno), 'ouch, pheno is missing: %s'%pheno
        cmd += ' -pheno %s'%pheno
    if sim!=None:
        assert os.path.exists(sim), 'ouch, sim is missing: %s'%sim
        cmd += ' -sim %s'%sim
    if linreg==True:
        cmd += ' -linreg'
    if covar!=None:
        assert os.path.exists(covar), 'ouch, covar is missing: %s'%covar
        cmd += ' -covar %s'%covar
    if REML:
        cmd += ' -REML'
    else:
        cmd += ' -ML'
    if out!=None:
        cmd += ' -out %s'%out
    if optLogdelta!=None:
        cmd += ' -brentMinLogVal %.4f -brentMaxLogVal %.4f -brentStarts 1 -brentMaxIter 1'%(optLogdelta-1E-3,optLogdelta+1E-3) #Why substract .001?
    else:
        cmd += ' -brentMinLogVal %.4f -brentMaxLogVal %.4f '%(-5,10)
    if extract!=None:
        assert os.path.exists(extract), 'ouch, extract is missing: %s'%extract
        cmd += ' -extract %s'%extract
    if extractSim!=None:
        assert os.path.exists(extractSim), 'ouch, extractSim is missing: %s'%extractSim
        cmd += ' -extractSim %s'%extractSim
    if autoSelect!=None:
        cmd += ' -autoSelect %s'%autoSelect
    if autoSelectCriterionMSE == True:
        cmd += ' -autoSelectCriterionMSE'
    if excludeByGeneticDistance!=None:
        cmd += ' -excludeByGeneticDistance %d'%excludeByGeneticDistance
    if excludeByPosition!=None:
        cmd += ' -excludeByPosition %d -verboseOut'%excludeByPosition
    if eigen!=None:
        cmd += ' -eigen %s'%eigen
    if eigenOut!=None:
        cmd += ' -eigenOut %s -runGwasType NORUN'%eigenOut # stop after computing the EVD
    if maxThreads!=None:
        cmd += ' -maxThreads %d'%maxThreads
    if simOut!=None:
        cmd += ' -simOut %s -runGwasType NORUN'%simOut # stop after computer the EVD
    if numJobs!=None:
        cmd += ' -numjobs %d'%numJobs
    if thisJob!=None:
        cmd += ' -thisjob %d'%thisJob

    if bfileSim!=None:
        assert os.path.exists(bfileSim + '.bed'), 'ouch, bfileSim is missing: %s'%bfileSim
        assert os.path.exists(bfileSim + '.bim'), 'ouch, bfileSim is missing: %s'%bfileSim
        assert os.path.exists(bfileSim + '.fam'), 'ouch, bfileSim is missing: %s'%bfileSim
        cmd += ' -bfileSim %s'%bfileSim
    if topKbyLinReg!=None:
        cmd += ' -topKbyLinReg %d'%topKbyLinReg
    logging.info(cmd)
    
    output = ""
    try:
        output = subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
    except Exception as e:
        print(e)
    print(output)
    #LG.info(output)
    #return output

