'''
a altset_list is a list of snpsets (defined with the stand-aline PySnpTools readers)

'''

#A altset_list is defined with two classes that implement these two interfaces: ISnpSetList and ISnpSetListPlusBed.
#Note: Python doesn't know enforce interfaces.

#interface ISnpSetList
#    def addbed(self, bed):
#        return # ISnpSetListPlusBed

#interface ISnpSetListPlusBed:
#    def __len__(self):
#        return # number of snpsets in this list

#    def __iter__(self):
#        return # a sequence of ISnpSetPlusBed's

from .snpandsetnamecollection import SnpAndSetNameCollection  # noqa: F401
from .subset import Subset # noqa: F401
from .minmaxsetsize import MinMaxSetSize # noqa: F401
from .consecutive import Consecutive # noqa: F401
#!! ISnpSetList Later replace all interfaces with abstract classes
#!!remove all these __init__.py imports?
