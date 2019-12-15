from __future__ import absolute_import
import pdb
import os
import numpy as SP
from .snpset import *
import logging
from six.moves import range

# attempt to import wrapped plink parser
WRAPPED_PLINK_PARSER_PRESENT = True
try:
    import pysnptools.snpreader.wrap_plink_parser
except Exception:
    WRAPPED_PLINK_PARSER_PRESENT = False


def readPED(basefilename, delimiter = ' ',missing = '0',standardize = True, pheno = None):
    '''
    read [basefilename].ped and [basefilename].map
    optionally standardize the SNPs and mean impute missing SNPs
    --------------------------------------------------------------------------
    Input:
    basefilename    : string of the basename of [basename].ped and [basename].map
    delimiter       : string (default ' ' space)
    missing         : string indicating a missing genotype (default '0')
    standardize     : boolean
                        True    : mean impute, zero-mean and unit variance the data
                        False   : output the data in 0,1,2 with NaN values
    --------------------------------------------------------------------------
    Output dictionary:
    'rs'     : [S] array rs-numbers,
    'pos'    : [S*3] array of positions [chromosome, genetic dist, basepair dist],
    'snps'   : [N*S] array of snps-data,
    'iid'    : [N*2] array of family IDs and individual IDs
    --------------------------------------------------------------------------
    '''
    pedfile = basefilename+".ped"
    mapfile = basefilename+".map"
    map = SP.loadtxt(mapfile,dtype = 'str',comments=None)

    rs = map[:,1]
    pos = SP.array(map[:,(0,2,3)],dtype = 'float')
    map = None

    ped = SP.loadtxt(pedfile,dtype = 'str',comments=None)
    iid = ped[:,0:2]
    snpsstr = ped[:,6::]
    inan=snpsstr==missing
    snps = SP.zeros((snpsstr.shape[0],snpsstr.shape[1]//2))
    if standardize:
        for i in range(snpsstr.shape[1]//2):
            snps[inan[:,2*i],i]=0
            vals=snpsstr[~inan[:,2*i],2*i:2*(i+1)]
            snps[~inan[:,2*i],i]+=(vals==vals[0,0]).sum(1)
            snps[~inan[:,2*i],i]-=snps[~inan[:,2*i],i].mean()
            snps[~inan[:,2*i],i]/=snps[~inan[:,2*i],i].std()
    else:
        for i in range(snpsstr.shape[1]/2):
            snps[inan[:,2*i],i]=SP.nan
            vals=snpsstr[~inan[:,2*i],2*i:2*(i+1)]
            snps[~inan[:,2*i],i]+=(vals==vals[0,0]).sum(1)
    if pheno is not None:
        #TODO: sort and filter SNPs according to pheno
        pass
    ret = {
           'rs'     :rs,
           'pos'    :pos,
           'snps'   :snps,
           'iid'    :iid
           }
    return ret

def readRAW(basefilename, delimiter = ' ',missing = '0',standardize = True, pheno = None):
    '''
    read [basefilename].raw
    optionally standardize the SNPs and mean impute missing SNPs
    --------------------------------------------------------------------------
    Input:
    basefilename    : string of the basename of [basename].ped and [basename].map
    delimiter       : string (default ' ' space)
    missing         : string indicating a missing genotype (default '0')
    standardize     : boolean
                        True    : mean impute, zero-mean and unit variance the data
                        False   : output the data in 0,1,2 with NaN values
    --------------------------------------------------------------------------
    Output dictionary:
    'rs'     : [S] array rs-numbers,
    'pos'    : [S*3] array of positions [chromosome, genetic dist, basepair dist],
    'snps'   : [N*S] array of snps-data,
    'iid'    : [N*2] array of family IDs and individual IDs
    --------------------------------------------------------------------------
    '''
    rawfile = basefilename+".raw"
    #mapfile = basefilename+".map"
    #map = SP.loadtxt(mapfile,dtype = 'str',comments=None)

    #rs = map[:,1]
    #pos = SP.array(map[:,(0,2,3)],dtype = 'float')
    #map = None
    raw = SP.loadtxt(rawfile,dtype = 'str',comments=None)
    iid = raw[:,0:2]
    snpsstr = raw[:,6::]
    inan=snpsstr==missing
    snps = SP.zeros((snpsstr.shape[0],snpsstr.shape[1]/2))
    if standardize:
        for i in range(snpsstr.shape[1]/2):
            raw[inan[:,2*i],i]=0
            vals=snpsstr[~inan[:,2*i],2*i:2*(i+1)]
            snps[~inan[:,2*i],i]+=(vals==vals[0,0]).sum(1)
            snps[~inan[:,2*i],i]-=snps[~inan[:,2*i],i].mean()
            snps[~inan[:,2*i],i]/=snps[~inan[:,2*i],i].std()
    else:
        for i in range(snpsstr.shape[1]/2):
            snps[inan[:,2*i],i]=SP.nan
            vals=snpsstr[~inan[:,2*i],2*i:2*(i+1)]
            snps[~inan[:,2*i],i]+=(vals==vals[0,0]).sum(1)
    if pheno is not None:
        #TODO: sort and filter SNPs according to pheno
        pass
    ret = {
           'rs'     :rs,
           'pos'    :pos,
           'snps'   :snps,
           'iid'    :iid
           }
    return ret


def writePhen(phen, filename, missing = '9', sep="\t"):
    '''
    must contain phen['iid'] and phen['vals']
    '''
    N = phen['iid'].shape[0]
    M = phen['vals'].shape[1]
    assert N==phen['vals'].shape[0], "number of individuals do not match up in phen['vals'] and phen['ids']"
    with open(filename, 'w') as f:
        for i in range(0,N):
            tmpstr = phen['iid'][i][0] + sep +phen['iid'][i][1]            
            for m in range(0,M):
                tmpstr += sep + str(phen['vals'][i][m])
            tmpstr += "\n"
            f.write(tmpstr)
    

def readBED(basefilename, snp_set = AllSnps(), order = 'F'):
    '''
    This is a one-shot reader for BED files that internally uses the Bed class. If you need repeated random access to a BED file,
    it is much faster to use the Bed class directly. Such use avoids the need to re-read the associated BIM file.
    '''
    from . import snpreader as sr
    bed = sr.Bed(basefilename)
    return bed.read(snp_set, order = order)

def nSnpFromBim(basefilename):
    bim = basefilename+'.bim'
    bimx = SP.loadtxt(bim,dtype = 'str',usecols = (0,1,2,3),comments=None)
    S = bimx.shape[0]
    return S

def findIndex(idsSep, bedidsSep):
    sids = set()
    for i in range(idsSep.shape[0]):
        sids.add( idsSep[i,0] + "_" + idsSep[i,1] )

    beids = dict()
    for i in range(bedidsSep.shape[0]):
        beids[ bedidsSep[i,0] + "_" + bedidsSep[i,1] ] = i

    inter = sids.intersection( set(beids.keys()) )

    return [beids[x] for x in inter]

def filter(phe, bed):
    # for pheno
    index = findIndex(bed['iid'], phe['iid'])
    iid = phe['iid']
    iid = iid[index,:]
    vals = phe['vals']
    vals = vals[index]
    phe['iid'] = iid
    phe['vals'] = vals
    # for snp
    index = findIndex(phe['iid'], bed['iid'])
    iid = bed['iid']
    iid = iid[index,:]
    snps = bed['snps']
    snps = snps[index,:]
    bed['iid'] = iid
    bed['snps'] = snps
    return

#if __name__ == "__main__":
#    #datadir = "C:\\Users\\lippert\\Projects\\ARIC" ; basefilename = os.path.join(datadir, 'whiteMale')
#    datadir = "testdata" ; basefilename = os.path.join(datadir, 'test')
#    #bed = readBED(basefilename,blocksize = 1,nSNPs = 20000,start = 650000)
#    #bed = readBED(basefilename,blocksize = 1,nSNPs = 20000,start = 0)
#    #phe = loadPhen(basefilename + ".tab")
#    filter(phe, bed)
#    bed = readBED('Gaw14/all', startpos = [1,17.5599], endpos = [1,24.6047])
