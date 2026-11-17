"""Report GPU precision choices without changing requested numerical behavior."""

import warnings

import numpy as np


def warn_gpu_float64(dtype, xp):
    if xp is np or np.dtype(dtype) != np.dtype("float64"):
        return
    import bed_reader

    warn = getattr(bed_reader, "_warn_gpu_float64", None)
    if warn is not None:
        warn(dtype)
    else:
        # Keep GPU workflows usable with older published dependencies.
        warnings.warn(
            "float64 arithmetic can be much slower than float32 on consumer GPUs. FaST-LMM's "
            "float64 GPU arrays also use twice the storage. Consider dtype='float32' if its "
            "precision is sufficient; the requested dtype is unchanged.",
            UserWarning,
            stacklevel=2,
        )
