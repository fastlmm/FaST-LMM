from __future__ import absolute_import
import numpy as SP
import subprocess, sys, os.path

class SnpIndexList(object): # implements ISnpSet
    '''
    When given to a bed reader, tells it to read the snps at the indexes given
     See the Bed class's 'read' method of examples of its use.
     See __init__.py for specification of interface it implements.
    '''

    def __init__(self, list):
        '''
        list of snp indexes to include
        '''
        self.list = list

    def addbed(self, bed):
        return SnpIndexListPlusBed(self,bed)

class SnpIndexListPlusBed(object): # implements ISnpSetPlusBed
    def __init__(self, spec, bed):
        self.spec = spec
        self.bed = bed

    def __str__(self):
        return "SnpIndex(...)"

    def __iter__(self):
        for bimindex in self.spec.list:
            yield bimindex

    def __len__(self):
        return len(self.spec.list)

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