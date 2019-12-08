'''
#!! fix comments
snpset is a named set of snps.  See the Bed class's 'read' method of examples of their use.

A snpset is defined with two classes that implement these two interfaces: ISnpSet and ISnpSetPlusBed.
Note: Python doesn't enforce interfaces.

interface ISnpSet
    def addbed(self, bed):
        return # ISnpSetPlusBed

interface ISnpSetPlusBed:
    def __len__(self):
        return # number of snps in this set

    def __iter__(self):
        return # index number to position in BIM file

    def __str__(self):
        return # string of name of this set of snps
'''

from __future__ import absolute_import
from .Lrt import *
from .Cv import *
from .Sc import *
from . import LRT_up as lrt

