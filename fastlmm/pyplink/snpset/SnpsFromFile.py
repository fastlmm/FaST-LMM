from __future__ import absolute_import
import numpy as SP
import subprocess, sys, os.path

class SnpsFromFile(object): # implements ISnpSet
    '''
     See the Bed class's 'read' method of examples of its use.
     See __init__.py for specification of interface it implements.
     '''
    def __init__(self, filename):
        self.filename = filename

    def addbed(self, bed):
        return SnpsFromFilePlusBed(self.filename, bed)

class SnpsFromFilePlusBed(object): # implements ISnpSetPlusBed
    def __init__(self, filename, bed):
        self.filename = filename
        snplist0 = ReadTokens(filename)
        bed.run_once()
        self.snplist = [val for val in snplist0 if val in bed.snp_to_index] #use the intersection        
        if len(snplist0) != len(self.snplist): raise Exception("The file '{0}' contains SNPs not in bed '{1}'".format(filename,bed.basefilename))
        self.bed = bed

    def __str__(self):
        return self.filename

    def __len__(self):
        return len(self.snplist)

    def __iter__(self):
        for snp in self.snplist:
            index = self.bed.snp_to_index[snp]
            yield index

    def read(self):
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

def ReadTokens(filename):
    token_list = open(filename).read().split() # could be written to 'yield' the tokens one at a time
    return token_list;
