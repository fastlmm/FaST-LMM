"""Workflow precision defaults, delegated to PySnpTools."""
from pysnptools.util import array_dtype


def dtype_environment(dtype):
    """Return the environment setting that scopes ``dtype`` as the default."""
    return {"ARRAY_DTYPE": dtype.name}


__all__ = ["array_dtype", "dtype_environment"]
