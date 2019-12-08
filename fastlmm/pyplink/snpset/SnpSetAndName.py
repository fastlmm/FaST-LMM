from __future__ import absolute_import
import numpy as SP
import subprocess, sys, os.path


class SnpAndSetName(object): # implements ISnpSet
    '''
     See the Bed class's 'read' method of examples of its use.
     See __init__.py for specification of interface it implements.
     '''
    def __init__(self, setname, snplist):
        self.name = setname
        self.snplist = snplist

    def addbed(self, bed):
        return SnpAndSetNamePlusBed(self.name,self.snplist,bed)

class SnpAndSetNamePlusBed(object): # implements ISnpSetPlusBed
    '''
    A single set of snps.
    '''
    def __init__(self, setname, snplist, bed):
        self.name = setname
        self.snplist = [val for val in snplist if val in bed.snp_to_index] #use the intersection
        self.bed = bed

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.snplist)

    def __iter__(self):
        for snp in self.snplist:
            index = self.bed.snp_to_index[snp]
            yield index

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