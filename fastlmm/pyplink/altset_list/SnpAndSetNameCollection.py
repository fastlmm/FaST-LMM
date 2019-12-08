from __future__ import absolute_import
import numpy as SP
import subprocess, sys, os.path
from itertools import *
from fastlmm.pyplink.snpset import *
import logging
import six

class SnpAndSetNameCollection(object):  # implements ISnpSetList
    '''
    Specifies a list of snp sets via a file that has columns 'snp' and 'group'.
    See the Bed class's 'read' method of examples of its use.
    See __init__.py for specification of interface it implements.
    '''
    def __init__(self, filename):
        self.filename = filename
        logging.info("Reading {0}".format(filename))
        import pandas as pd
        snp_and_setname_sequence = pd.read_csv(filename,delimiter = '\s',index_col=False,engine='python')

        from collections import defaultdict
        setname_to_snp_list = defaultdict(list)
        for snp,gene in snp_and_setname_sequence.itertuples(index=False):
            setname_to_snp_list[gene].append(snp)
        self.bigToSmall = sorted(six.iteritems(setname_to_snp_list), key = lambda gene_snp_list:-len(gene_snp_list[1]))

    def addbed(self, bed):
        return SnpAndSetNameCollectionPlusBed(self,bed)

    def copyinputs(self, copier):
        copier.input(self.filename)

    #would be nicer if these used generic pretty printer
    def __repr__(self):
        return "SnpAndSetNameCollection(filename={0})".format(self.filename)

    def __iter__(self):
        for gene,snp_list in self.bigToSmall:
            yield gene,snp_list


class GenomeSplitCollection(object):  # implements ISnpSetList
    '''
    Specifies a list of snp sets by splitting the genome into windows of fixed size.
    '''
    def __init__(self, bed, windowsize,idist): # implements ISnpSetListPlusBed
        '''
        Input:
        bed         : bed object
        windowsize  : size of the window
        idist       : index in pos array that the exclusion is based on.
                  (1=genetic distance, 2=basepair distance)
        '''
        self.bed = bed
        self.snp_list = []
        chrom = bed.pos[:,0]
        dist = bed.pos[:,idist]

        # divide into blocks
        chromUnique = SP.unique(chrom)
        blockId=0
        for chromId in chromUnique:
            chromIdx = chromId==chrom
            distStart = dist[chromIdx].min()
            distMax = dist[chromIdx].max()

            while distStart<distMax:
                distStop = distStart + windowsize
                idxBlock = SP.logical_and(chromIdx,SP.logical_and(distStart<=dist,dist<distStop))
                self.snp_list.append(('block%d'%blockId,bed.rs[idxBlock]))
                blockId += 1
                distStart = distStop

       

    def __len__(self):
        return len(self.snp_list)


    def __iter__(self):
        for block, snp_list in self.snp_list:
            if len(set(snp_list)) != len(snp_list) : raise Exception("Some snps are listed more than once.")
            yield SnpAndSetNamePlusBed(block,snp_list,self.bed)



class SnpAndSetNameCollectionPlusBed(object): # implements ISnpSetListPlusBed
    '''
    The SnpAndSetNameCollection with the addition of BED information.
    '''
    def __init__(self, spec, bed):
        self.spec = spec
        self.bed = bed

    def __len__(self):
        return len(self.spec.bigToSmall)

    def __iter__(self):
        for gene, snp_list in self.spec.bigToSmall:
            if len(set(snp_list)) != len(snp_list) : raise Exception("Some snps in gene {0} are listed more than once".format(gene))
            yield SnpAndSetNamePlusBed(gene,snp_list,self.bed)
