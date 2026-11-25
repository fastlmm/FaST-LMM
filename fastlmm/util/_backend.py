"""Conservative boundaries for workflows qualified only on NumPy/float64."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial, wraps
from inspect import signature
from threading import Condition, RLock, get_ident
import os
from types import MemberDescriptorType
import warnings
from weakref import WeakKeyDictionary

import numpy as np
from pysnptools.kernelreader import KernelData
from pysnptools.snpreader import SnpData

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
        self._condition = Condition()
        self._borrowers = 0

    def is_local(self):
        return self.active and self.pid == os.getpid()

    def __reduce__(self):
        # A serialized worker must establish its own environment.
        return type(None), ()

    @contextmanager
    def borrow(self):
        with self._condition:
            admitted = self.active
            if admitted:
                self._borrowers += 1
        try:
            yield admitted
        finally:
            if admitted:
                with self._condition:
                    self._borrowers -= 1
                    if not self._borrowers:
                        self._condition.notify_all()

    def close(self):
        with self._condition:
            self._condition.wait_for(lambda: not self._borrowers)
            self.active = False


def cpu64_required():
    scope = _cpu64_active.get()
    return scope is not None and scope.is_local()


@contextmanager
def cpu64_scope(api, *, backend=None, dtype=None, scope=None):
    if scope is None or not scope.is_local():
        scope = _cpu64_active.get()
    if (
        backend is None and dtype is None and scope is not None and scope.is_local()
        and scope.owner == get_ident() and _cpu64_settings_active()
    ):
        # Nested call on the owning thread: the owner already holds workflow_lock
        # and its environment is unchanged, so re-entering would only rewrite
        # and restore identical values. Inner-loop methods such as LMM.nLLeval
        # take this path, which keeps the policy from slowing epistasis.
        # A scope passed explicitly may not be in this context yet; nested
        # cpu64_required() checks must still see it.
        token = None if _cpu64_active.get() is scope else _cpu64_active.set(scope)
        try:
            yield
        finally:
            if token is not None:
                _cpu64_active.reset(token)
        return
    may_borrow = scope is not None and scope.is_local() and scope.owner != get_ident()
    # Workers share the owner's settings without writing to the environment.
    # If the map raises, the owner still waits for admitted threads to finish.
    with scope.borrow() if may_borrow else nullcontext(False) as borrowed:
        with _cpu64_environment(api, backend, dtype, scope, borrowed):
            yield


_CPU64_SETTINGS = {"ARRAY_MODULE": "numpy", **dtype_environment(np.dtype("float64"))}


def _cpu64_settings_active():
    return all(os.environ.get(key) == value for key, value in _CPU64_SETTINGS.items())


def _nested_on_owner_thread():
    # True inside an active scope on the thread that owns it (and so holds
    # workflow_lock), with the scope's environment still in place.
    scope = _cpu64_active.get()
    return (
        scope is not None and scope.is_local() and scope.owner == get_ident()
        and _cpu64_settings_active()
    )


@contextmanager
def _cpu64_environment(api, backend, dtype, scope, borrowed):
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
                f"instead of requested {backend}/{dtype.name}.", UserWarning, stacklevel=5,
            )
        settings = _CPU64_SETTINGS
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
            scope.close()
            _cpu64_active.reset(token)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _cpu_input(value, api):
    # Only inspect numerical storage, never metadata such as iid/pos. read()
    # converts in-memory PySnpTools readers without modifying the user's object.
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
    receiver = next(iter(parameters.parameters), None)
    check_model = (
        function.__module__.startswith("fastlmm.inference.")
        and receiver in ("self", "lmm")
        and function.__name__ != "fit"
    )

    has_runner_parameter = any(name.endswith("runner") for name in parameters.parameters)

    @wraps(function)
    def wrapped(*args, **kwargs):
        if not has_runner_parameter and (args or not check_model) and _nested_on_owner_thread():
            # Nested call on the owning thread, such as LMM.nLLeval for each
            # epistasis pair: the environment and lock are already in place, so
            # apply the same model and input checks without re-entering the
            # scope or binding the signature. The receiver is the first
            # parameter, so a positional call passes it as args[0].
            if check_model:
                _check_model_state(args[0], function.__qualname__)
            checked_args = []
            for value in args:
                checked_args.append(_cpu_input(value, api))
            checked_kwargs = {}
            for name, value in kwargs.items():
                checked_kwargs[name] = _cpu_input(value, api)
            return function(*checked_args, **checked_kwargs)
        with cpu64_scope(api):
            # Old pickles or direct attribute assignments can contain unqualified
            # model state. Widening a cached decomposition cannot repair its error.
            bound = parameters.bind(*args, **kwargs)
            if check_model:
                _check_model_state(bound.arguments[receiver], function.__qualname__)
            for name, value in bound.arguments.items():
                value = _cpu_input(value, api)
                if name.endswith("runner"):
                    value = cpu64_runner(value, api)
                bound.arguments[name] = value
            return function(*bound.args, **bound.kwargs)
    return wrapped


def _check_model_state(model, api):
    # Runs on every call to a decorated inference method (for example, per
    # epistasis pair), so the common NumPy case avoids string work.
    def check(name, value):
        if isinstance(value, np.ndarray):
            if value.dtype == np.float32:
                raise ValueError(f"{api}: cached model {name} is float32; refit with numpy/float64")
            return
        module = type(value).__module__
        if module == "cupy" or module.startswith("cupy."):
            raise ValueError(f"{api}: cached model {name} uses CuPy; refit with numpy/float64")

    for name, value in getattr(model, "__dict__", {}).items():
        check(name, value)
    for name, descriptor in _slot_descriptors(type(model)):
        try:
            value = descriptor.__get__(model)
        except AttributeError:
            continue  # The slot has not been set.
        check(name, value)


# Weak keys, so caching does not keep dynamically created classes alive.
_slot_descriptor_cache = WeakKeyDictionary()


def _slot_descriptors(model_class):
    # Walk defining classes so inherited, private, and shadowed slots are all
    # inspected directly, without invoking properties on a user subclass.
    # Slots are fixed when a class is created, so the walk is cached per class.
    descriptors = _slot_descriptor_cache.get(model_class)
    if descriptors is None:
        descriptors = tuple(
            (name, descriptor)
            for defining_class in model_class.__mro__
            if "__slots__" in vars(defining_class)
            for name, descriptor in vars(defining_class).items()
            if isinstance(descriptor, MemberDescriptorType)
        )
        _slot_descriptor_cache[model_class] = descriptors
    return descriptors


def cpu64_if(required, *, api=None):
    """Select a worker boundary without changing qualified single-kernel paths."""
    return lambda function: cpu64(function, api=api) if required else function


def cpu64_runner(runner, api):
    if runner is None or isinstance(runner, _Cpu64Runner):
        return runner
    return _Cpu64Runner(runner, api)


def cpu64_mapper(mapper, api):
    """Carry the current CPU64 scope through a synchronous map callback."""
    return partial(_cpu64_call, mapper, api, _cpu64_active.get())


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
            scoped_reducer = cpu64_mapper(cpu64(reducer), self.api)
            return self.runner.run_list(work, reducer=scoped_reducer, name=name)
        with cpu64_scope(self.api):
            return reducer(self.runner.run(distributable) for distributable in work)


def _cpu64_call(callback, api, scope, argument):
    with cpu64_scope(api, scope=scope):
        return callback(argument)


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
