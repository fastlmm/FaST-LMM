"""Backend selection must wait for preceding workflow environment restoration."""
import importlib
import os
from unittest.mock import patch

import numpy as np
import pytest

from fastlmm.association import single_snp
from pysnptools.snpreader import SnpData


@pytest.mark.parametrize("first_backend", ["numpy", "cupy"])
@pytest.mark.parametrize("second_kernel", [False, True])
def test_concurrent_default_backend_resolves_after_lock(first_backend, second_kernel):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, RLock
    import pandas as pd

    requested = np if first_backend == "numpy" else pytest.importorskip("cupy")
    # The NumPy case tests this race even without CuPy: the second call must
    # reject its invalid original default rather than inherit the first call.
    original_backend = "invalid-default" if first_backend == "numpy" else "numpy"
    module = importlib.import_module("fastlmm.association.single_snp")
    first_entered = Event()
    second_attempted = Event()
    release_first = Event()
    lock = RLock()
    observed = []

    class ObservedLock:
        def __enter__(self):
            if first_entered.is_set():
                second_attempted.set()
            lock.acquire()

        def __exit__(self, *exception):
            lock.release()

    def internal(**kwargs):
        observed.append(os.environ["ARRAY_MODULE"])
        if not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(10)
        return pd.DataFrame({"sid_index": [0]})

    data = SnpData(iid=[["a", "a"], ["b", "b"], ["c", "c"]],
                   sid=["s"], val=np.array([[0.0], [1.0], [2.0]]), xp=np)
    with patch.dict("os.environ", {"ARRAY_MODULE": original_backend}), \
         patch.object(module, "_array_module_lock", ObservedLock()), \
         patch.object(module, "_internal_single", internal), \
         ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(single_snp, data, data, xp=requested, leave_out_one_chrom=False)
        try:
            assert first_entered.wait(10)
            second = executor.submit(single_snp, data, data, xp=None,
                                     K1=data if second_kernel else None, leave_out_one_chrom=False)
            assert second_attempted.wait(10)
        finally:
            release_first.set()
        first.result(timeout=10)
        if first_backend == "numpy":
            with pytest.raises(ValueError, match="invalid-default"):
                second.result(timeout=10)
            assert observed == ["numpy"]
        else:
            second.result(timeout=10)
            assert observed == ["cupy", "numpy"]
        assert os.environ["ARRAY_MODULE"] == original_backend
