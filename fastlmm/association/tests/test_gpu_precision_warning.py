"""Report the effective GPU precision, including cached and worker paths."""

import warnings

import bed_reader
import numpy as np
import pytest

from fastlmm.association.single_snp import _worker_array, _workflow_array_module
from fastlmm.inference.lmm_cov import LMM
from fastlmm.util._gpu_warnings import warn_gpu_float64

WarningType = bed_reader.GpuFloat64Warning


def test_cpu_and_two_kernel_coercion_do_not_warn_about_fp64():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_gpu_float64(np.float64, np)
        with _workflow_array_module("numpy", has_second_kernel=True) as xp:
            warn_gpu_float64(np.float64, xp)
    assert not caught


def test_worker_precision_warning_and_dtype_preservation():
    cupy = pytest.importorskip("cupy")
    with pytest.warns(WarningType, match="consumer GPUs"):
        values = _worker_array(np.eye(3), cupy)
    assert values.dtype == np.float64 and isinstance(values, cupy.ndarray)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _worker_array(np.eye(3, dtype=np.float32), cupy)
    assert result.dtype == np.float32 and not caught


def test_direct_gpu_model_decomposition_warns():
    cupy = pytest.importorskip("cupy")
    model = LMM(X=cupy.ones((3, 1)), Y=cupy.arange(3, dtype=cupy.float64).reshape(-1, 1),
                K=cupy.eye(3), xp=cupy)
    with pytest.warns(WarningType, match="consumer GPUs"):
        model.getSU()
    assert model.S.dtype == np.float64
