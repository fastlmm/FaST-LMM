"""Report GPU precision choices without changing requested numerical behavior."""

import numpy as np


def warn_gpu_float64(dtype, xp):
    if xp is np or np.dtype(dtype) != np.dtype("float64"):
        return
    from bed_reader._gpu import _warn_gpu_float64

    _warn_gpu_float64(dtype)
