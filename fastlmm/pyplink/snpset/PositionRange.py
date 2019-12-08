from __future__ import absolute_import
import numpy as SP
import subprocess, sys, os.path
from six.moves import range

class PositionRange(object): # implements ISnpSet
    '''
    When given to a bed reader, tells it to read 'nSNPs' starting at index position 'start'.
     See the Bed class's 'read' method of examples of its use.
     See __init__.py for specification of interface it implements.
    '''

    def __init__(self, start=0,nSnps=SP.inf):
        '''
        start           : index of the first SNP to be loaded from the .bed-file
                          (default 0)
        nSNPs           : load nSNPs from the .bed file (default SP.inf, meaning all)
        '''
        self.start = start
        self.nSNPs = nSnps

    def addbed(self, bed):
        return PositionRangePlusBed(self,bed)

class PositionRangePlusBed(object): # implements ISnpSetPlusBed
    def __init__(self, spec, bed):
        self.spec = spec
        self.bed = bed

    def __str__(self):
        return "PositionRange(start={0},nSNPs={1})".format(self.spec.start,self.spec.nSNPs)

    def __iter__(self):
        for bimindex in range(self.spec.start,self.spec.start+len(self)):  #note that 'self.spec.start+len(self)' is the 'stop', not the 'count'
            yield bimindex

    def __len__(self):
        return min(self.bed.snp_count-self.spec.start,self.spec.nSNPs)

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