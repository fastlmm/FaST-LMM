from __future__ import absolute_import
import numpy as sp
import subprocess, sys, os.path
import fastlmm.util.util as utilx


class RandomSnpSet(object): # implements ISnpSet
    '''
     See the Bed class's 'read' method of examples of its use.
     See __init__.py for specification of interface it implements.
     '''
    def __init__(self, numsnps, randomseed):        
        self.numsnps=numsnps
        self.randomseed=randomseed

    def addbed(self, bed):
        return RandomSnpSetPlusBed(self.numsnps,self.randomseed,bed)

class RandomSnpSetPlusBed(object): # implements ISnpSetPlusBed
    '''
    One random set of snps.
    '''
    def __init__(self, numsnps, randomseed, bed):
        self.numsnps=numsnps
        self.randomseed=randomseed
        self.bed = bed
        #very inefficient extraction of a few random SNP indexes:        
        self.snpindlist = utilx.generate_permutation(sp.arange(0,bed.snp_count),randomseed)[0:numsnps]
        self.snpindlist.sort()        

    def __str__(self):
        return "RandomSnpSet(numsnps={0},randomseed={1})".format(self.numsnps,self.randomseed)

    def __len__(self):
        return self.numsnps

    def __iter__(self):       
        for i in self.snpindlist:        
            yield i

    def read(self): #!!why don't all the interface implementers have this method?
        return self.bed.read_with_specification(self)

    @property
    def pos(self):
        """
        Returns:
            pos:    position of the SNPs in the specification
        """
        return self.bed.pos[self.to_index]

    @property
    def to_index(self):
        iter = self.__iter__()
        return [i for i in iter]