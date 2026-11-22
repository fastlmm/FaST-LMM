"""Report GPU precision choices without changing requested numerical behavior."""

import numpy as np


def warn_gpu_float64(dtype, xp):
    if xp is np or np.dtype(dtype) != np.dtype("float64"):
        return
    from pysnptools.util._gpu_warnings import warn_gpu_float64

    warn_gpu_float64(dtype, xp)
