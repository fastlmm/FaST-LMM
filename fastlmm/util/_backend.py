"""Conservative boundaries for workflows qualified only on NumPy/float64."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from threading import RLock, get_ident
import os
import warnings

import numpy as np

from fastlmm.util._precision import array_dtype, dtype_environment

# All FaST-LMM workflow scopes share this lock. Resolve defaults after acquiring
# it so a concurrent workflow cannot inherit another call's temporary settings.
workflow_lock = RLock()
# Carry the enclosing API's CPU64 requirement into nested single_snp workers
# without restricting independent single-kernel calls or serializing a lock.
_cpu64_active = ContextVar("fastlmm_cpu64_active", default=None)


class _Cpu64Scope:
    def __init__(self):
        self.pid = os.getpid()
        self.owner = get_ident()
        self.active = True

    def is_local(self):
        return self.active and self.pid == os.getpid()

    def __reduce__(self):
        # A serialized worker must establish its own environment.
        return type(None), ()


def cpu64_required():
    scope = _cpu64_active.get()
    return scope is not None and scope.is_local()


@contextmanager
def cpu64_scope(api, *, backend=None, dtype=None, scope=None):
    if scope is None or not scope.is_local():
        scope = _cpu64_active.get()
    borrowed = scope is not None and scope.is_local() and scope.owner != get_ident()
    # A synchronous runner waits for its threads before the owning scope can
    # restore the environment. Its threads share that scope without taking
    # the owner's lock or writing to the process environment.
    with nullcontext() if borrowed else workflow_lock:
        backend = os.environ.get("ARRAY_MODULE", "numpy") if backend is None else backend
        if not isinstance(backend, str) or backend not in ("numpy", "cupy"):
            raise ValueError(f"{api}: invalid ARRAY_MODULE={backend!r}; expected numpy or cupy")
        dtype = array_dtype(dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"{api}: expected float32 or float64, got {dtype}")
        if backend != "numpy" or dtype != np.dtype("float64"):
            warnings.warn(
                f"{api} is qualified only for numpy/float64; running with numpy/float64 "
                f"instead of requested {backend}/{dtype.name}.", UserWarning, stacklevel=3,
            )
        settings = {"ARRAY_MODULE": "numpy", **dtype_environment(np.dtype("float64"))}
        if borrowed:
            if any(os.environ.get(key) != value for key, value in settings.items()):
                raise RuntimeError(f"{api}: worker changed the shared NumPy/float64 environment")
            token = _cpu64_active.set(scope)
            try:
                yield
            finally:
                _cpu64_active.reset(token)
            return

        previous = {key: os.environ.get(key) for key in settings}
        scope = _Cpu64Scope()
        token = _cpu64_active.set(scope)
        try:
            os.environ.update(settings)
            yield
        finally:
            scope.active = False
            _cpu64_active.reset(token)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _cpu_input(value, api):
    # Only inspect numerical storage, never metadata such as iid/pos. read()
    # converts in-memory PySnpTools readers without modifying the user's object.
    from pysnptools.snpreader import SnpData
    from pysnptools.kernelreader import KernelData

    if isinstance(value, (SnpData, KernelData)):
        values = value.val
        if type(values).__module__.split(".")[0] == "cupy" or values.dtype == np.float32:
            warnings.warn(f"{api}: supplied {type(values).__module__}/{values.dtype} data "
                          "will be copied to numpy/float64; existing input rounding remains.",
                          UserWarning, stacklevel=4)
            return value.read(xp=np, dtype=np.float64)
        return value
    if isinstance(value, np.ndarray) or type(value).__module__.split(".")[0] == "cupy":
        if getattr(value, "dtype", None) is None:
            return value
        device = type(value).__module__.split(".")[0] == "cupy"
        if device or value.dtype == np.float32:
            warnings.warn(f"{api}: supplied array will be copied to numpy/float64; "
                          "existing input rounding remains.", UserWarning, stacklevel=4)
            value = value.get() if device else value
            return value.astype(np.float64) if value.dtype.kind == "f" else value
        return value
    if isinstance(value, dict):
        return {key: _cpu_input(item, api) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_input(item, api) for item in value)
    if isinstance(value, list):
        return [_cpu_input(item, api) for item in value]
    return value


def cpu64(function=None, *, api=None):
    """Apply the policy at public and serialized worker computation boundaries."""
    if function is None:
        return lambda function: cpu64(function, api=api)
    parameters = signature(function)
    api = function.__qualname__ if api is None else api

    @wraps(function)
    def wrapped(*args, **kwargs):
        with cpu64_scope(api):
            # Old pickles or direct attribute assignments can contain unqualified
            # model state. Widening a cached decomposition cannot repair its error.
            if args and function.__name__ != "fit" and type(args[0]).__module__.startswith("fastlmm.inference"):
                state = dict(getattr(args[0], "__dict__", {}))
                for name in getattr(type(args[0]), "__slots__", ()):
                    state[name] = getattr(args[0], name, None)
                for name, value in state.items():
                    if isinstance(value, np.ndarray) and value.dtype == np.float32:
                        raise ValueError(f"{function.__qualname__}: cached model {name} is float32; refit with numpy/float64")
                    if type(value).__module__.split(".")[0] == "cupy":
                        raise ValueError(f"{function.__qualname__}: cached model {name} uses CuPy; refit with numpy/float64")
            args = tuple(_cpu_input(value, api) for value in args)
            kwargs = {key: _cpu_input(value, api) for key, value in kwargs.items()}
            bound = parameters.bind(*args, **kwargs)
            for name, value in bound.arguments.items():
                if name.endswith("runner"):
                    bound.arguments[name] = cpu64_runner(value, api)
            return function(*bound.args, **bound.kwargs)
    return wrapped


def cpu64_if(required, *, api=None):
    """Select a worker boundary without changing qualified single-kernel paths."""
    return lambda function: cpu64(function, api=api) if required else function


def cpu64_runner(runner, api):
    if runner is None or isinstance(runner, _Cpu64Runner):
        return runner
    return _Cpu64Runner(runner, api)


class _Cpu64Runner:
    """Keep runner overrides inside a qualified worker boundary, including reduce."""
    def __init__(self, runner, api):
        self.runner = runner
        self.api = api

    def run(self, distributable):
        return self.runner.run(_Cpu64Distributable(distributable, self.api))

    def run_list(self, distributable_list, reducer, name=None):
        work = [_Cpu64Distributable(distributable, self.api) for distributable in distributable_list]
        if hasattr(self.runner, "run_list"):
            from functools import partial
            scoped_reducer = partial(
                _cpu64_reduce, cpu64(reducer), self.api, _cpu64_active.get()
            )
            return self.runner.run_list(work, reducer=scoped_reducer, name=name)
        with cpu64_scope(self.api):
            return reducer(self.runner.run(distributable) for distributable in work)


def _cpu64_reduce(reducer, api, scope, result_sequence):
    with cpu64_scope(api, scope=scope):
        return reducer(result_sequence)


def _cpu64_work(work, api, scope):
    from pysnptools.util.mapreduce1.runner import _run_all_in_memory
    with cpu64_scope(api, scope=scope):
        return _run_all_in_memory(work)


class _Cpu64Distributable:
    def __init__(self, distributable, api, scope=None):
        self.distributable = distributable
        self.api = api
        # Capture explicitly for PySnpTools versions whose threads start with
        # an empty context. Serialized subprocesses enter their own scope.
        self.scope = _cpu64_active.get() if scope is None else scope

    @property
    def work_count(self):
        with cpu64_scope(self.api, scope=self.scope):
            return self.distributable.work_count

    @property
    def tempdirectory(self):
        with cpu64_scope(self.api, scope=self.scope):
            return self.distributable.tempdirectory

    def work_sequence(self):
        return self._sequence(self.distributable.work_sequence)

    def work_sequence_range(self, start, end):
        from itertools import islice
        if hasattr(self.distributable, "work_sequence_range"):
            return self._sequence(lambda: self.distributable.work_sequence_range(start, end))
        return self._sequence(lambda: islice(self.distributable.work_sequence(), start, end))

    def _sequence(self, make_sequence):
        from functools import partial
        # Both generator initialization and advancing can read numerical data.
        # End each scope before yielding; never hold the process lock across a
        # suspended generator that a runner might consume on another thread.
        with cpu64_scope(self.api, scope=self.scope):
            sequence = iter(make_sequence())
        while True:
            with cpu64_scope(self.api, scope=self.scope):
                try:
                    work = next(sequence)
                except StopIteration:
                    return
            if hasattr(work, "work_sequence"):
                # Preserve the nested work tree so the runner can still balance
                # and split it, rather than serializing a whole subtree as one task.
                yield _Cpu64Distributable(work, self.api, self.scope)
            else:
                yield partial(_cpu64_work, work, self.api, self.scope)

    def reduce(self, result_sequence):
        with cpu64_scope(self.api, scope=self.scope):
            return self.distributable.reduce(result_sequence)

    def copyinputs(self, copier):
        with cpu64_scope(self.api, scope=self.scope):
            return self.distributable.copyinputs(copier)

    def copyoutputs(self, copier):
        with cpu64_scope(self.api, scope=self.scope):
            return self.distributable.copyoutputs(copier)

    def __str__(self):
        return str(self.distributable)
