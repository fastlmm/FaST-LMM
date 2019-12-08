from __future__ import absolute_import
import numpy as np
import subprocess, sys, os.path
from itertools import *
import math
from six.moves import range

class Consecutive(object):  # implements ISnpSetList
    """
    The sets should be every consecutive set of SNPs within a 2cM window of each user 
    (distance in cM is in the 3rd column of the bim file).  As for the name of the set,
    please make it <position-of-first-snp>@<position-of-middle-snp>@<position-of-last-snp>.
    For 'middle' please break a tie to the first SNP.
    """

    def __init__(self, bimFileName,cMWindow):
        self.BimFileName = bimFileName
        self.CMWindow = cMWindow

    def addbed(self, bed):
        return _ConsecutivePlusBed(self,bed)

    def copyinputs(self, copier):
        copier.input(self.BimFileName)

    #would be nicer if these used generic pretty printer
    def __repr__(self):
        return "Consecutive(bimFileName={0},bimFileName={1})".format(self.BimFileName,self.CMWindow)


        
class _ConsecutivePlusBed(object): # implements ISnpSetListPlusBed
    def __init__(self, spec, bed):
        self.spec = spec
        self.bed = bed
        import pandas as pd
        bimfields = pd.read_csv(self.spec.BimFileName,delimiter = '\t',usecols = (0,1,2,3),header=None,index_col=False)
        self.chrom = bimfields[0]
        self.sid = bimfields[1]
        self.cm = bimfields[2]


    def __iter__(self):
        startIndex=-1
        endIndex=0 #one too far
        while(True):
            startIndex+=1
            if startIndex >= len(self.sid):
                return
            while endIndex < len(self.sid) and self.chrom[startIndex] == self.chrom[endIndex] and  self.cm[endIndex] - self.cm[startIndex] <= self.spec.CMWindow:
                endIndex+=1
            lastIndex = endIndex - 1;
            midIndex = math.floor((startIndex+lastIndex)/2.0)
            name = "{0}@{1}@{2}".format(startIndex,midIndex,lastIndex)

            snpList=self.sid[list(range(startIndex,endIndex))]
            yield _SnpAndSetNamePlusBed(name,snpList,self.bed)

    def __len__(self):
        return len(self.sid)
