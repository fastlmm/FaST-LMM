from __future__ import absolute_import
from __future__ import print_function
import fastlmm.util.preprocess as util

def TESTBEFOREUSING_score_testfilesFromDir(phenofile, base0, pedfilesalt, covarfile = None, outfile = None, ipheno=0, mindist = -1.0, idist=2 ,filetype='PED'):
    '''
    given a list of basefilenames that define alternative models and a basefilename of the null model
    test all alternative models
    --------------------------------------------------------------------------
    Input:
    phenofile   : filename of the phenotype file
    base0       : basefilename of the .ped and .map files containing the
                  null-model SNPs
    pedfilesalt : [Nalt] list of basefilenames of the Nalt .ped and .map files
                  containing the alternative-model SNPs
    covarfile   : filename of the covariates file (default None, in this case
                  only a bias is used)
    outfile     : filename of the output file (default None, in this case no
                  output is written to disk)
    ipheno      : 0-based index of the phenotype to be analyzed (default 0)
    mindist     : minimum distance for SNPs to be included in null model
                  (default -1.0: no excluson in this case)
    idist       : index in pos array that the exclusion is based on.
                  (1=genetic distance, 2=basepair distance)
    filetype    : plink filetype of the input (default 'PED')
                    'PED'   : PED file format
                    'BED'   : BED file format
    --------------------------------------------------------------------------
    Output dictionary:
    'pv'        : [Nalt] array P-values,
    'lik0'      : [Nalt] array containing the model parameters and negative
                  log-likelihoods of the null models,
    'lik1'      : [Nalt] array containing the model parameters and negative
                  log-likelihoods of the alternative models,
    'nexclude'  : [Nalt] array of numbers of SNPs excluded,
    'filenames' : [Nalt] array of basefilenames
    --------------------------------------------------------------------------
    '''
    pheno = pstpheno.loadPhen(filename = phenofile, missing ='-9', pheno = None)
    if covarfile is None:
        X = SP.ones((pheno['vals'].shape[0],1))
    else:
        covar = pstpheno.loadPhen(filename = covarfile, missing ='-9', pheno = None)
        X = SP.hstack((SP.ones((pheno['vals'].shape[0],1)),covar['vals']))
    if filetype =='PED':
        SNPs0 = plink.readPED(basefilename = base0, delimiter = ' ',missing = '0',standardize = True, pheno = None)
    elif filetype =='BED':
        SNPs0 = plink.readBED(basefilename = base0)
        SNPs0['snps'] = util.standardize(SNPs0['snps'])

    y = pheno['vals'][:,ipheno]
    G0 = SNPs0['snps']/SP.sqrt(SNPs0['snps'].shape[1])

    #build the null model
    test2K = scoretest(Y=y[:,SP.newaxis],X=X,K=None,G=G0)

    squaredform = SP.zeros(len(pedfilesalt))
    expectationsqform = SP.zeros(len(pedfilesalt))
    varsqform = SP.zeros(len(pedfilesalt))
    squaredform2K = SP.zeros(len(pedfilesalt))
    expectationsqform2K = SP.zeros(len(pedfilesalt))
    varsqform2K = SP.zeros(len(pedfilesalt))
    nexclude = SP.zeros(len(pedfilesalt))
    include = SP.zeros(len(pedfilesalt))
    Pv = SP.zeros(len(pedfilesalt))
    Pv2K = SP.zeros(len(pedfilesalt))

    for i, base1 in enumerate(pedfilesalt):#iterate over all ped files
        SNPs1 = plink.readPED(basefilename = base1, delimiter = ' ',missing = '0',standardize = True, pheno = None)
        if mindist>=0:
            i_exclude =  excludeinds(SNPs0['pos'], SNPs1['pos'], mindist = mindist,idist = idist)
            nexclude[i] = i_exclude.sum()
        else:
            nexclude[i]=0
        G1 = SNPs1['snps']/SP.sqrt(SNPs1['snps'].shape[1])

        if nexclude[i]>0:
            test2Ke = scoretest(Y=y[:,SP.newaxis],X=X,K=None,G=G0[:,~i_exclude])
            squaredform2K[i], expectationsqform2K[i], varsqform2K[i] = test2Ke.score( G = G1 )
        else:
            squaredform2K[i], expectationsqform2K[i], varsqform2K[i] = test2K.score( G = G1 )
        squaredform[i], expectationsqform[i], varsqform[i] = scoreNoK( y, X = X, G = G1, sigma2=None)

        #perform moment matching
        Pv2K[i],dofchi22K,scalechi22K=pv_mom(squaredform2K[i],expectationsqform2K[i],varsqform2K[i])
        Pv[i],dofchi2,scalechi2=pv_mom(squaredform[i],expectationsqform[i],varsqform[i])

    ret = {
           'filenames': SP.array(pedfilesalt,dtype=str),
           'squaredform':squaredform,
           'expectationsqform':expectationsqform,
           'varsqform':varsqform,
           'P':Pv,
           'squaredform2K':squaredform2K,
           'expectationsqform2K':expectationsqform2K,
           'varsqform2K':varsqform2K,
           'nexclude':nexclude,
           'P2K':Pv2K
           }
    if outfile is not None:
        #TODO
        print('implement me!')
        #header = SP.array(['PV_5050','neg_log_lik_0','neg_loglik_alt','n_snps_excluded','filename_alt'])
        #data = SP.concatenate(())
    return ret


def lrt_testfilesFromDir(phenofile, base0, pedfilesalt, covarfile = None, outfile = None, ipheno=0, mindist = -1.0, idist=2 ,filetype='PED'):
    '''
    given a list of basefilenames that define alternative models and a basefilename of the null model
    test all alternative models
    --------------------------------------------------------------------------
    Input:
    phenofile   : filename of the phenotype file
    base0       : basefilename of the .ped and .map files containing the
                  null-model SNPs
    pedfilesalt : [Nalt] list of basefilenames of the Nalt .ped and .map files
                  containing the alternative-model SNPs
    covarfile   : filename of the covariates file (default None, in this case
                  only a bias is used)
    outfile     : filename of the output file (default None, in this case no
                  output is written to disk)
    ipheno      : 0-based index of the phenotype to be analyzed (default 0)
    mindist     : minimum distance for SNPs to be included in null model
                  (default -1.0: no excluson in this case)
    idist       : index in pos array that the exclusion is based on.
                  (1=genetic distance, 2=basepair distance)
    filetype    : plink filetype of the input (default 'PED')
                    'PED'   : PED file format
                    'BED'   : BED file format
    --------------------------------------------------------------------------
    Output dictionary:
    'pv'        : [Nalt] array P-values,
    'lik0'      : [Nalt] array containing the model parameters and negative
                  log-likelihoods of the null models,
    'lik1'      : [Nalt] array containing the model parameters and negative
                  log-likelihoods of the alternative models,
    'nexclude'  : [Nalt] array of numbers of SNPs excluded,
    'filenames' : [Nalt] array of basefilenames
    --------------------------------------------------------------------------
    '''
    pheno = pstpheno.loadPhen(filename = phenofile, missing ='-9', pheno = None)
    if covarfile is None:
        X = SP.ones((pheno['vals'].shape[0],1))
    else:
        covar = pstpheno.loadPhen(filename = covarfile, missing ='-9', pheno = None)
        X = SP.hstack((SP.ones((pheno['vals'].shape[0],1)),covar['vals']))
    if filetype =='PED':
        SNPs0 = plink.readPED(basefilename = base0, delimiter = ' ',missing = '0',standardize = True, pheno = None)
    elif filetype =='BED':
        SNPs0 = plink.readBED()
    y = pheno['vals'][:,ipheno]
    G0 = SNPs0['snps']/SP.sqrt(SNPs0['snps'].shape[1])

    #build the null model
    a0 = fastlmm.getLMM()
    a0.setG(G0)
    a0.setX(X)
    a0.sety(y)
    lik0_default = a0.findH2()  # The null model only has a single kernel and only needs to find h2
    lik0 = SP.zeros(len(pedfilesalt),dtype = 'object')
    lik1 = SP.zeros(len(pedfilesalt),dtype = 'object')
    lrt = SP.zeros(len(pedfilesalt))
    pv = SP.zeros(len(pedfilesalt))
    nexclude = SP.zeros(len(pedfilesalt))
    for i, base1 in enuemrate(pedfilesalt):#iterate over all ped files
        SNPs1 = plink.readPED(basefilename = base1, delimiter = ' ',missing = '0',standardize = True, pheno = None)
        i_exclude =  excludeinds(SNPs0['pos'], SNPs1['pos'], mindist = mindist,idist = idist)
        nexclude[i] = i_exclude.sum()
        G1 = SNPs1['snps']/SP.sqrt(SNPs1['snps'].shape[1])
        if nexclude[i]: #recompute the null likelihood
            G0_excluded = SNPs0['snps'][:,~i_exclude]/SP.sqrt(SNPs0['snps'][:,~i_exclude].shape[1])
            [pv[i],lik0[i],lik1[i]] = twokerneltest(G0=G0_excluded, G1=G1, y=y, covar=X, appendbias=False,lik0=None)
        else:           #use precomputed null likelihood
            [pv[i],lik0[i],lik1[i]] = twokerneltest(G0=G0, G1=G1, y=y, covar=X, appendbias=False,lik0=lik0_default)
    ret = {
           'pv':pv,
           'lik0': lik0,
           'lik1':lik1,
           'nexclude':nexclude,
           'filenames': SP.array(pedfilesalt,dtype=str)
           }
    if outfile is not None:
        #TODO
        print('implement me!')
        #header = SP.array(['PV_5050','neg_log_lik_0','neg_loglik_alt','n_snps_excluded','filename_alt'])
        #data = SP.concatenate(())
    return ret
