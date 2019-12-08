'''
a altset_list is a list of snpsets


A altset_list is defined with two classes that implement these two interfaces: ISnpSetList and ISnpSetListPlusBed.
Note: Python doesn't know enforce interfaces.

interface ISnpSetList
    def addbed(self, bed):
        return # ISnpSetListPlusBed

interface ISnpSetListPlusBed:
    def __len__(self):
        return # number of snpsets in this list

    def __iter__(self):
        return # a sequence of ISnpSetPlusBed's

'''
from __future__ import absolute_import
from fastlmm.pyplink.snpset import *
from .SnpAndSetNameCollection import *
from .Subset import *
from .MinMaxSetSize import *
from .Consecutive import *
