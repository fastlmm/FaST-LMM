from __future__ import absolute_import
import numpy as np
import subprocess, sys, os.path
from itertools import *
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
        snp_and_setname_sequence = pd.read_csv(filename,delimiter = '\t',index_col=False)

        from collections import defaultdict
        setname_to_sid = defaultdict(list)
        for snp,gene in snp_and_setname_sequence.itertuples(index=False):
            setname_to_sid[gene].append(snp)
        self.bigToSmall = sorted(six.iteritems(setname_to_sid), key = lambda gene_sid:-len(gene_sid[1]))

    def addbed(self, bed):
        return _SnpAndSetNameCollectionPlusBed(self,bed)

    def copyinputs(self, copier):
        copier.input(self.filename)

    #would be nicer if these used generic pretty printer
    def __repr__(self):
        return "SnpAndSetNameCollection(filename={0})".format(self.filename)

    def __iter__(self):
        for gene,sid in self.bigToSmall:
            yield gene,sid


class _SnpAndSetNameCollectionPlusBed(object): # implements ISnpSetListPlusBed
    '''
    The SnpAndSetNameCollection with the addition of BED information.
    '''
    def __init__(self, spec, bed):
        self.spec = spec
        self.bed = bed

    def __len__(self):
        return len(self.spec.bigToSmall)

    def __iter__(self):
        for gene, sid in self.spec.bigToSmall:
            if len(set(sid)) != len(sid) : raise Exception("Some snps in gene {0} are listed more than once".format(gene))
            sid_index = self.bed.sid_to_index(sid)
            yield gene, self.bed[:,sid_index]
