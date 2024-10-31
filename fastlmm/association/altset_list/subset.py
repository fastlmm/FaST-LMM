class Subset(object): # implements ISnpSetList
    '''
    Returns a subset of the originally specified sets to test.
    See the Bed class's 'read' method of examples of its use.
    See __init__.py for specification of interface it implements.
    '''
    def __init__(self, altset_list, subset_list):
        self.subset_list = subset_list
        if isinstance(altset_list, str):#if given a filename, then assumes group-SNP format (default)
            from fastlmm.pyplink.altset_list import SnpAndSetNameCollection
            self.inner = SnpAndSetNameCollection(altset_list)
        else:                           #given a NucRangeList(filenam.txt), or any other reader
            self.inner = altset_list

    def addbed(self, bed):
        return _SubsetPlusBed(self,bed)

    #would be nicer if these used generic pretty printer
    def __repr__(self):
        return "Subset(altset_list={0},subset_list={1})".format(self.inner,self.subset_list)




class _SubsetPlusBed(object): # implements ISnpSetListPlusBed
    '''
    Returns a subset of the originally specified sets to test.
    '''
    def __init__(self, spec, bed):
        self.spec = spec
        self.bed = bed
        self.inner = spec.inner.addbed(bed)

    def __len__(self):
        return len(self.spec.subset_list)

    def __iter__(self):                 #after grabbing main list (self.inner_altset_list), do subsetting
        seenit=[]
        for altset_name, altset in self.inner:
            if altset_name in self.spec.subset_list:
               seenit.append(altset_name)
               yield altset_name, altset
        if not len(seenit)==len(self):
            raise Exception("some sets specified not present")
