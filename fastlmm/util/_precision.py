"""Workflow precision defaults with compatibility for published dependencies."""
import os

import bed_reader
import numpy as np
import pysnptools.util as pstutil


def array_dtype(dtype=None):
    resolver = getattr(pstutil, "array_dtype", None)
    if resolver is not None:
        return resolver(dtype)
    # Keep existing CPU/default usage working with supported older PySnpTools.
    # New workflow-wide precision requires a dependency that implements it.
    if dtype is None and "ARRAY_DTYPE" in os.environ:
        raise RuntimeError("ARRAY_DTYPE requires PySnpTools with array_dtype support")
    return np.dtype("float64" if dtype is None else dtype)


def dtype_environment(dtype):
    """Establish the new policy only when both dependencies implement it."""
    if (getattr(pstutil, "array_dtype", None) is not None
            and getattr(bed_reader, "array_dtype", None) is not None):
        return {"ARRAY_DTYPE": dtype.name}
    # Otherwise injecting ARRAY_DTYPE would turn a previously working CPU call
    # into a request for an unsupported feature in an older dependency.
    if "ARRAY_DTYPE" in os.environ:
        raise RuntimeError("ARRAY_DTYPE requires matching precision-aware PySnpTools and bed-reader")
    return {}
