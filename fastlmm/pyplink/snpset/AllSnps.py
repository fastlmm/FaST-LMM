from __future__ import absolute_import
import numpy as SP
import subprocess, sys, os.path
from six.moves import range


class AllSnps(object): # implements ISnpSet
    '''
    When given to a bed reader, tells it to read all snps. See the Bed class's 'read' method of examples of its use.
    See __init__.py for specification of interface it implements.
    '''
    def addbed(self, bed):
        return AllSnpsPlusBed(bed)

class AllSnpsPlusBed(object): # implements ISnpSetPlusBed

    def __init__(self,bed):
        self.bed = bed

    def __len__(self):
        return self.bed.snp_count

    def __iter__(self):
        for bimindex in range(self.bed.snp_count):
            yield bimindex

    def __str__(self):
        return "AllSnps"

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