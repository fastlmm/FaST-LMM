"""Backend tracing helpers shared by test modules.

Kept free of pytest so the canonical unittest runner can import them in
environments without test extras (CI's core and minimum-dependency jobs).
"""
import json
import os

from pysnptools.snpreader import Bed
from pysnptools.util import get_array_module


def _record(path, kind, backend, dtype=None):
    expected = os.environ["ARRAY_MODULE"]
    assert backend == expected, (kind, backend, expected)
    with open(f"{path}.{os.getpid()}.jsonl", "a") as stream:
        stream.write(json.dumps({"kind": kind, "backend": backend, "precision": os.environ.get("ARRAY_DTYPE"), "actual_dtype": None if dtype is None else str(dtype), "pid": os.getpid()}) + "\n")


class _TracingBed(Bed):
    def __init__(self, path, trace):
        super().__init__(path, count_A1=True)
        self.trace = str(trace)

    def _read_with_standardizing(self, *args, **kwargs):
        from fastlmm.inference.lmm_cov import LMM

        trace = self.trace
        if not getattr(LMM.nLLeval, "_environment_trace", False):
            original_evaluate = LMM.nLLeval

            def evaluate(model, *arguments, **keywords):
                _record(trace, "evaluate", model._xp.__name__, model.Y.dtype)
                if keywords.get("snps") is not None:
                    _record(trace, "test_snps", get_array_module(keywords["snps"]).__name__, keywords["snps"].dtype)
                return original_evaluate(model, *arguments, **keywords)

            evaluate._environment_trace = True
            LMM.nLLeval = evaluate
        backend = os.environ["ARRAY_MODULE"]
        if kwargs.get("xp") is not None:
            from pysnptools.util import array_module
            assert array_module(kwargs["xp"]).__name__ == backend
        if backend == "cupy":
            import cupy as cp
            if not getattr(cp.linalg.eigh, "_environment_trace", False):
                original_eigh = cp.linalg.eigh

                def eigh(values, *arguments, **keywords):
                    _record(trace, "eigh", get_array_module(values).__name__, values.dtype)
                    return original_eigh(values, *arguments, **keywords)

                eigh._environment_trace = True
                cp.linalg.eigh = eigh
        result = super()._read_with_standardizing(*args, **kwargs)
        data = result[0] if isinstance(result, tuple) else result
        _record(trace, "read", get_array_module(data.val).__name__, data.val.dtype)
        return result
