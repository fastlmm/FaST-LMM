from .Beta import *
from .Unit import *

# import warnings
# warnings.warn("This __init__.py is deprecated. Pysnptools includes newer version", DeprecationWarning)


def factory(s):
    s = s.capitalize()
    if s == "Unit" or s == "Unit()":
        return Unit()

    if s == "Beta":
        return Beta()

    if s.startswith("Beta("):
        standardizer = eval(s)
        return standardizer


def standardize_with_lambda(snps, lambdax, blocksize=None):
    if blocksize is None:
        return lambdax(snps)

    idx_start = 0
    idx_stop = blocksize

    while idx_start < snps.shape[1]:
        # print idx_start
        lambdax(snps[:, idx_start:idx_stop])

        idx_start = idx_stop
        idx_stop += blocksize
        if idx_stop > snps.shape[1]:
            idx_stop = snps.shape[1]

    return snps
