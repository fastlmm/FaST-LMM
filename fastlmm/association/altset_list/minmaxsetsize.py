class MinMaxSetSize(object): # implements ISnpSetList
    '''
    Returns a subset of the originally specified sets where each group as at least minsetsize and no more than maxsetsize (inclusive).
    minsetsize and maxsetsize can be None
    See the Bed class's 'read' method of examples of its use.
    See __init__.py for specification of interface it implements.
    '''
    def __init__(self, altset_list, minsetsize, maxsetsize):
        self.minsetsize = minsetsize
        self.maxsetsize = maxsetsize
        if isinstance(altset_list, str):#if given a filename, then assumes group-SNP format (default)
            from fastlmm.pyplink.altset_list import SnpAndSetNameCollection
            self.inner = SnpAndSetNameCollection(altset_list)
        else:                           #given a NucRangeList(filenam.txt), or any other reader
            self.inner = altset_list

    def addbed(self, bed):
        return _MinMaxSetSizePlusBed(self,bed)

    #would be nicer if these used generic pretty printer
    def __repr__(self):
        return "MinMaxSetSize(altset_list={0},minsetsize={1},maxsetsize={2})".format(self.inner,self.minsetsize,self.maxsetsize)


class _MinMaxSetSizePlusBed(object): # implements ISnpSetListPlusBed
    def __init__(self, spec, bed):
        self.spec = spec
        self.bed = bed
        self.inner = spec.inner.addbed(bed)
        self.__len = None


    def __len__(self):
        if self.__len is None:
            self.__len = sum(1 for i in self)
        return self.__len

    def __iter__(self):
        for name, snp_reader in self.inner:
            setsize = snp_reader.sid_count
            if (self.spec.minsetsize is None or self.spec.minsetsize <= setsize) and (self.spec.maxsetsize is None or setsize <= self.spec.maxsetsize ) :
               yield name, snp_reader
