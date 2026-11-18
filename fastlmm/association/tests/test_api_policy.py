"""Direct public API policy checks, also registered with the canonical suite."""
import os
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from bed_reader import to_bed, has_gpu_support
from pysnptools.snpreader import Bed, SnpData, SnpReader
from pysnptools.util import get_array_module
from pysnptools.util.filecache import LocalCache
from pysnptools.util.mapreduce1.runner import Local, LocalMultiProc

import fastlmm.association as association
from fastlmm.inference import FastLMM, LinearRegression
from fastlmm.util import compute_auto_pcs, manhattan_plot
from fastlmm.util._backend import cpu64


class _PolicyBed(Bed):
    """Assert actual reader arrays inside workers, including native kernels."""
    def __init__(self, path, trace):
        super().__init__(path, count_A1=True)
        self.trace = str(trace)

    def _note(self, values):
        import json
        backend = get_array_module(values).__name__
        assert backend == "numpy" and values.dtype == np.float64
        assert os.environ["ARRAY_MODULE"] == "numpy" and os.environ["ARRAY_DTYPE"] == "float64"
        with open(f"{self.trace}.{os.getpid()}.jsonl", "a") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "backend": backend, "dtype": str(values.dtype)}) + "\n")

    def _read_array(self, *args, **kwargs):
        values = super()._read_array(*args, **kwargs)
        self._note(values)
        return values

    def _read(self, *args, **kwargs):
        values = super()._read(*args, **kwargs)
        self._note(values)
        return values

    def _read_with_standardizing(self, *args, **kwargs):
        result = super()._read_with_standardizing(*args, **kwargs)
        self._note((result[0] if isinstance(result, tuple) else result).val)
        return result


class _FailingPolicyBed(_PolicyBed):
    def _read_array(self, *args, **kwargs):
        super()._read_array(*args, **kwargs)
        raise RuntimeError("intentional fallback worker failure")

    def _read(self, *args, **kwargs):
        super()._read(*args, **kwargs)
        raise RuntimeError("intentional fallback worker failure")


def _unqualified_worker(task_index):
    return {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}


def _unqualified_initializer():
    os.environ.update(_unqualified_worker(0))


class TestApiPolicy(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        rng = np.random.default_rng(243)
        self.path = self.folder / "tiny.bed"
        to_bed(self.path, rng.integers(0, 3, size=(40, 24), dtype=np.int8),
               properties={"chromosome": np.repeat([1, 2, 3], 8)})
        self.reader = Bed(self.path, count_A1=True)
        self.pheno = SnpData(iid=self.reader.iid, sid=["y"], val=rng.normal(size=(40, 1)), xp=np)
        self.pheno_file = self.folder / "pheno.txt"
        self.pheno_file.write_text("\n".join(f"{family} {individual} {phenotype[0]:.17g}" for (family, individual), phenotype in zip(self.reader.iid, self.pheno.val)))
        self.covar = SnpData(iid=self.reader.iid, sid=["cov"], val=rng.normal(size=(40, 1)), xp=np)
        self.set_file = self.folder / "sets.txt"
        self.set_file.write_text("snp\tgroup\n" + "\n".join(f"{sid}\tset{index // 8}" for index, sid in enumerate(self.reader.sid)))

    def _calls(self):
        reader, pheno, covar = self.reader, self.pheno, self.covar
        return {
            "single_snp (two kernels)": lambda: association.single_snp(reader, pheno, covar=covar,
                K0=reader[:, :8], K1=reader[:, 8:16], mixing=0.4, h2=0.3, leave_out_one_chrom=False),
            "snp_set": lambda: association.snp_set(str(self.path.with_suffix("")), str(self.set_file), str(self.pheno_file), test="sc_davies", nperm=0),
            "single_snp_linreg": lambda: association.single_snp_linreg(reader, pheno, covar),
            "epistasis": lambda: association.epistasis(reader[:, :3], str(self.pheno_file), G0=reader, covar={"iid": covar.iid, "vals": covar.val, "header": covar.sid}, log_delta=0),
            "single_snp_all_plus_select": lambda: association.single_snp_all_plus_select(
                reader[:, :8], pheno, G=reader, covar=covar, k_list=[1], n_folds=2, do_plot=False, seed=37),
            "single_snp_select": lambda: association.single_snp_select(
                reader[:, :8], pheno, G=reader, covar=covar, k_list=[1], n_folds=2, seed=37),
            "single_snp_scale": lambda: association.single_snp_scale(reader, pheno, covar=covar,
                cache=LocalCache(str(self.folder / ("scale-" + os.environ.get("ARRAY_MODULE", "numpy") + "-" + os.environ.get("ARRAY_DTYPE", "float64")))), force_python_only=True),
            "heritability_spatial_correction": lambda: association.heritability_spatial_correction(
                reader, np.arange(80, dtype=np.float64).reshape(40, 2), reader.iid, [10], 2, pheno,
                jackknife_count=2, permute_plus_count=0, permute_times_count=0, just_testing=True),
        }

    def test_numerical_apis_compute_on_cpu64(self):
        original_read = SnpReader.read
        for name, call in self._calls().items():
            reference = None
            for backend, dtype in [("numpy", "float64"), ("numpy", "float32"), ("cupy", "float64"), ("cupy", "float32")]:
                with self.subTest(api=name, backend=backend, dtype=dtype):
                    observed = []

                    def read(reader, *args, **kwargs):
                        result = original_read(reader, *args, **kwargs)
                        observed.append((get_array_module(result.val).__name__, result.val.dtype))
                        self.assertEqual(os.environ["ARRAY_MODULE"], "numpy")
                        self.assertEqual(os.environ["ARRAY_DTYPE"], "float64")
                        self.assertEqual(observed[-1], ("numpy", np.dtype("float64")))
                        return result

                    settings = {"ARRAY_MODULE": backend, "ARRAY_DTYPE": dtype}
                    with patch.dict(os.environ, settings), patch.object(SnpReader, "read", read), warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        actual = call()
                        self.assertEqual({key: os.environ[key] for key in settings}, settings)
                    policy = [str(item.message) for item in caught if "qualified only" in str(item.message)]
                    if backend != "numpy" or dtype != "float64":
                        self.assertEqual(len(policy), 1, policy)
                        self.assertIn(name, policy[0])
                        self.assertIn("running with numpy/float64", policy[0])
                    else:
                        self.assertEqual(policy, [])
                    if name != "snp_set":
                        self.assertTrue(observed)
                    if reference is None:
                        reference = actual
                    else:
                        import pandas as pd
                        pd.testing.assert_frame_equal(actual, reference, check_exact=True)

    def test_predictor_methods(self):
        for predictor_type in (FastLMM, LinearRegression):
            predictor = predictor_type()
            kernel = self.reader if predictor_type is FastLMM else None
            with patch.dict(os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float64"}):
                predictor.fit(X=self.covar, y=self.pheno, K0_train=kernel, **({"h2raw": 0.3} if predictor_type is FastLMM else {"h2": 0.3}))
                expected = predictor.predict(X=self.covar, K0_whole_test=kernel)
                expected_score = predictor.score(X=self.covar, y=self.pheno, K0_whole_test=kernel)
            for method in ("fit", "predict", "score"):
                with self.subTest(predictor=predictor_type.__name__, method=method), \
                     patch.dict(os.environ, {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}), \
                     warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if method == "fit":
                        predictor.fit(X=self.covar, y=self.pheno, K0_train=kernel, **({"h2raw": 0.3} if predictor_type is FastLMM else {"h2": 0.3}))
                        self.assertEqual(predictor.beta.dtype, np.float64)
                        if predictor_type is FastLMM:
                            self.assertEqual(predictor.U.dtype, np.float64)
                            self.assertIsInstance(predictor.U, np.ndarray)
                    elif method == "predict":
                        actual = predictor.predict(X=self.covar, K0_whole_test=kernel)
                        for result, reference in zip(actual, expected):
                            self.assertIsInstance(result.val, np.ndarray)
                            self.assertEqual(result.val.dtype, np.float64)
                            np.testing.assert_array_equal(result.val, reference.val)
                    else:
                        self.assertEqual(predictor.score(X=self.covar, y=self.pheno, K0_whole_test=kernel), expected_score)
                    policy = [str(item.message) for item in caught if "qualified only" in str(item.message)]
                    self.assertEqual(len(policy), 1)
                    self.assertIn(f"{predictor_type.__name__}.{method}", policy[0])
                    self.assertEqual(os.environ["ARRAY_MODULE"], "cupy")
                    self.assertEqual(os.environ["ARRAY_DTYPE"], "float32")
            predictor.beta = predictor.beta.astype(np.float32)
            with self.assertRaisesRegex(ValueError, "cached model beta is float32"):
                predictor.predict(X=self.covar, K0_whole_test=kernel)

    def test_cpu64_scope_does_not_leak_to_concurrent_single_kernel(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event, RLock
        import importlib
        import pandas as pd
        from fastlmm.util import _backend

        module = importlib.import_module("fastlmm.association.single_snp")
        entered = Event()
        attempted = Event()
        release = Event()
        lock = RLock()
        observed = []

        class ObservedLock:
            def __enter__(self):
                if entered.is_set():
                    attempted.set()
                lock.acquire()

            def __exit__(self, *exception):
                lock.release()

        observed_lock = ObservedLock()

        @cpu64
        def first():
            self.assertTrue(_backend.cpu64_required())
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float64")
            entered.set()
            self.assertTrue(release.wait(10))

        def internal(**kwargs):
            observed.append((kwargs["dtype"], os.environ["ARRAY_DTYPE"], _backend.cpu64_required()))
            return pd.DataFrame({"sid_index": np.arange(self.reader.sid_count)})

        with patch.dict(os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float32"}), \
             patch.object(_backend, "workflow_lock", observed_lock), \
             patch.object(module, "_array_module_lock", observed_lock), \
             patch.object(module, "_internal_single", internal), \
             warnings.catch_warnings(record=True) as caught, ThreadPoolExecutor(max_workers=2) as executor:
            warnings.simplefilter("always")
            first_future = executor.submit(first)
            try:
                self.assertTrue(entered.wait(10))
                second_future = executor.submit(association.single_snp, self.reader, self.pheno,
                                                leave_out_one_chrom=False)
                self.assertTrue(attempted.wait(10))
            finally:
                release.set()
            first_future.result(timeout=10)
            second_future.result(timeout=10)
            self.assertEqual(observed, [(np.dtype("float32"), "float32", False)])
            self.assertFalse(_backend.cpu64_required())
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float32")
        self.assertEqual(sum("qualified only" in str(item.message) for item in caught), 1)

    def test_failure_restores_settings_for_every_boundary(self):
        calls = list(self._calls().items()) + [
            ("snp_set", lambda: association.snp_set(None, None, None)),
            ("compute_auto_pcs", lambda: compute_auto_pcs("/does-not-exist.bed")),
        ]
        for predictor_type in (FastLMM, LinearRegression):
            predictor = predictor_type()
            for method in ("fit", "predict", "score"):
                calls.append((f"{predictor_type.__name__}.{method}", getattr(predictor, method)))
        # Force failure only after the real boundary has established its scope.
        def fail(*args, **kwargs):
            self.assertEqual(os.environ["ARRAY_MODULE"], "numpy")
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float64")
            raise RuntimeError("intentional read failure")

        for name, call in calls:
            with self.subTest(api=name), patch.dict(os.environ, {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}), \
                 patch.object(SnpReader, "read", fail), patch.object(Local, "run", fail), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaises(Exception):
                    call()
                policy = [str(item.message) for item in caught if "qualified only" in str(item.message)]
                self.assertEqual(len(policy), 1)
                self.assertIn(name, policy[0])
                self.assertEqual(os.environ["ARRAY_MODULE"], "cupy")
                self.assertEqual(os.environ["ARRAY_DTYPE"], "float32")

    def test_supplied_float32_and_gpu_data_are_copied(self):
        for backend in ("numpy",):
            if backend == "cupy":
                import cupy as xp
            else:
                xp = np
            data = self.reader.read(xp=xp, dtype=np.float32)
            pheno = self.pheno.read(xp=xp, dtype=np.float32)
            original = data.val.copy()
            reference = association.single_snp_linreg(data.read(xp=np, dtype=np.float64), pheno.read(xp=np, dtype=np.float64), self.covar)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = association.single_snp_linreg(data, pheno, self.covar)
            self.assertTrue(any("supplied" in str(item.message) for item in caught))
            self.assertEqual(data.val.dtype, np.float32)
            xp.testing.assert_array_equal(data.val, original)
            # Input rounding is retained exactly; compare against those same
            # rounded values explicitly supplied in the qualified representation.
            np.testing.assert_array_equal(result.PValue, reference.PValue)

    def test_epistasis_rejects_float32_cache(self):
        cache = self.folder / "cache.npz"
        np.savez(cache, np.eye(40, dtype=np.float32), np.ones(40, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "numpy/float64 cache"):
            association.epistasis(self.reader[:, :2], self.pheno, G0=self.reader, log_delta=0, cache_file=str(cache))

    def test_plot_host_contract_does_not_change_configuration(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        values = np.array([[1, 1, 0.1], [2, 3, 0.01]], dtype=np.float32)
        with patch.dict(os.environ, {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}), warnings.catch_warnings(record=True) as caught:
            manhattan_plot(values)
            self.assertEqual(os.environ["ARRAY_MODULE"], "cupy")
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float32")
        self.assertFalse(any("qualified only" in str(item.message) for item in caught))
        plt.close("all")

    def test_scope_restores_unset_settings_and_preserves_exceptions(self):
        @cpu64
        def fail():
            self.assertEqual(os.environ["ARRAY_MODULE"], "numpy")
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float64")
            raise LookupError("original exception")
        with patch.dict(os.environ):
            os.environ.pop("ARRAY_MODULE", None)
            os.environ.pop("ARRAY_DTYPE", None)
            with self.assertRaisesRegex(LookupError, "original exception"):
                fail()
            self.assertNotIn("ARRAY_MODULE", os.environ)
            self.assertNotIn("ARRAY_DTYPE", os.environ)


    def test_explicit_two_kernel_request_and_alias(self):
        import importlib
        module = importlib.import_module("fastlmm.association.single_snp")
        original = module._snp_tester
        observed = []

        def tester(*args, **kwargs):
            model = args[3]
            observed.append((get_array_module(model.U).__name__, model.U.dtype))
            self.assertEqual(observed[-1], ("numpy", np.dtype("float64")))
            return original(*args, **kwargs)

        settings = {"ARRAY_MODULE": "invalid-before", "ARRAY_DTYPE": "invalid-before"}
        with patch.dict(os.environ, settings), patch.object(module, "_snp_tester", tester), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            association.single_snp(self.reader, self.pheno, K1=self.reader[:, :4], xp="cupy", dtype="float32",
                                   h2=0.3, mixing=0.4, leave_out_one_chrom=False)
            self.assertEqual({key: os.environ[key] for key in settings}, settings)
        self.assertTrue(observed)
        policy = [str(item.message) for item in caught if "qualified only" in str(item.message)]
        self.assertEqual(len(policy), 1)
        self.assertIn("single_snp (two kernels)", policy[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            alias = association.single_snp_leave_out_one_chrom(self.reader, self.pheno, xp=np, dtype="float64", h2=0.3)
        reference = association.single_snp(self.reader, self.pheno, xp=np, dtype="float64", h2=0.3)
        np.testing.assert_array_equal(alias.PValue, reference.PValue)
        self.assertTrue(any(issubclass(item.category, DeprecationWarning) for item in caught))

    def test_missing_cupy_is_an_error_for_single_kernel(self):
        import builtins
        original = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == "cupy":
                raise ImportError("CuPy unavailable in this regression")
            return original(name, *args, **kwargs)

        with patch.dict(os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float64"}), patch("builtins.__import__", importing):
            with self.assertRaisesRegex(ImportError, "CuPy unavailable"):
                association.single_snp(self.reader, self.pheno, xp="cupy", dtype="float32")
            self.assertEqual(os.environ["ARRAY_MODULE"], "numpy")
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float64")

    def test_two_kernel_fallback_does_not_require_cupy(self):
        import builtins
        original_import = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == "cupy":
                raise ImportError("CuPy unavailable in this regression")
            return original_import(name, *args, **kwargs)

        arguments = dict(test_snps=self.reader, pheno=self.pheno, K1=self.reader[:, :4],
                         h2=0.3, mixing=0.4, leave_out_one_chrom=False)
        with patch.dict(os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float64"}):
            reference = association.single_snp(xp=np, dtype="float64", **arguments)
        with patch.dict(os.environ, {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}), \
             patch("builtins.__import__", importing), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = association.single_snp(**arguments)
            self.assertEqual(os.environ["ARRAY_MODULE"], "cupy")
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float32")
        self.assertTrue(any("single_snp (two kernels)" in str(item.message) and "running with numpy/float64" in str(item.message) for item in caught))
        import pandas as pd
        pd.testing.assert_frame_equal(actual, reference, check_exact=True)

    def test_unqualified_worker_overrides_use_cpu64(self):
        import json
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        before = Path.cwd()
        try:
            os.chdir(self.folder)
            runner = LocalMultiProc(2, num_threads=1, taskindex_to_environ=_unqualified_worker)
            trace = self.folder / "fallback-workers"
            reader = _PolicyBed(self.path, trace)
            calls = {
                "linreg": lambda: association.single_snp_linreg(reader, self.pheno, runner=runner),
                "two_kernel_outer": lambda: association.single_snp(reader, self.pheno, K1=reader[:, :4], h2=0.3,
                    mixing=0.4, runner=runner, xp=np, dtype="float64"),
                "two_kernel_inner": lambda: association.single_snp(reader, self.pheno, K1=reader[:, :4], h2=0.3,
                    mixing=0.4, runner=runner, map_reduce_outer=False, xp=np, dtype="float64"),
                "epistasis": lambda: association.epistasis(reader[:, :3], str(self.pheno_file), G0=reader, log_delta=0, runner=runner),
                "selection": lambda: association.single_snp_all_plus_select(reader[:, :8], self.pheno, G=reader,
                    k_list=[1], n_folds=2, do_plot=False, seed=37, runner=runner),
                "deprecated_selection": lambda: association.single_snp_select(reader[:, :8], self.pheno, G=reader,
                    k_list=[1], n_folds=2, seed=37, runner=runner),
                "scale": lambda: association.single_snp_scale(reader, self.pheno, runner=runner, min_work_count=2,
                    cache=LocalCache(str(self.folder / "worker-scale")), force_python_only=True),
            }
            for name, call in calls.items():
                with self.subTest(api=name), patch.dict(os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float64"}):
                    previous = set(self.folder.glob("fallback-workers.*.jsonl"))
                    result = call()
                    self.assertTrue(len(result))
                    current = set(self.folder.glob("fallback-workers.*.jsonl")) - previous
                    records = [json.loads(line) for path in current for line in path.read_text().splitlines()]
                    self.assertTrue(any(record["pid"] != os.getpid() for record in records), name)
                    self.assertEqual(os.environ["ARRAY_MODULE"], "numpy")
                    self.assertEqual(os.environ["ARRAY_DTYPE"], "float64")
            with ProcessPoolExecutor(max_workers=2, initializer=_unqualified_initializer, mp_context=get_context("spawn")) as executor, \
                 patch.dict(os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float64"}):
                previous = set(self.folder.glob("fallback-workers.*.jsonl"))
                association.heritability_spatial_correction(reader, np.arange(80, dtype=np.float64).reshape(40, 2),
                    reader.iid, [10], 2, self.pheno, jackknife_count=2, permute_plus_count=0, permute_times_count=0,
                    just_testing=True, always_remote=True, map_function=executor.map)
                current = set(self.folder.glob("fallback-workers.*.jsonl")) - previous
                records = [json.loads(line) for path in current for line in path.read_text().splitlines()]
                self.assertTrue(any(record["pid"] != os.getpid() for record in records))
        finally:
            os.chdir(before)

    def test_concurrent_fallback_and_single_kernel_restore_defaults(self):
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import contextmanager
        from threading import Event
        import importlib
        import pandas as pd

        module = importlib.import_module("fastlmm.association.single_snp")
        original_scope = module._workflow_array_module
        original_read = SnpReader.read
        entered = Event()
        attempted = Event()
        release = Event()
        observed = []

        def read(reader, *args, **kwargs):
            if not entered.is_set():
                entered.set()
                assert release.wait(10)
            return original_read(reader, *args, **kwargs)

        @contextmanager
        def workflow(*args, **kwargs):
            attempted.set()
            with original_scope(*args, **kwargs) as xp:
                yield xp

        def internal(**kwargs):
            observed.append((os.environ["ARRAY_MODULE"], os.environ["ARRAY_DTYPE"], kwargs["dtype"]))
            return pd.DataFrame({"sid_index": np.arange(self.reader.sid_count)})

        with patch.dict(os.environ, {"ARRAY_MODULE": "numpy", "ARRAY_DTYPE": "float32"}), \
             patch.object(SnpReader, "read", read), patch.object(module, "_workflow_array_module", workflow), \
             patch.object(module, "_internal_single", internal), ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(association.single_snp_linreg, self.reader, self.pheno)
            try:
                self.assertTrue(entered.wait(10))
                second = executor.submit(association.single_snp, self.reader, self.pheno, leave_out_one_chrom=False)
                self.assertTrue(attempted.wait(10))
                self.assertFalse(observed)
            finally:
                release.set()
            first.result(timeout=10)
            second.result(timeout=10)
            self.assertEqual(observed, [("numpy", "float32", np.dtype("float32"))])
            self.assertEqual(os.environ["ARRAY_DTYPE"], "float32")

    def test_legacy_exported_models_report_cpu64(self):
        from fastlmm.inference import (LMM, getLMM, getGLMM, LaplaceGLMM_N3K1,
                                      LaplaceGLMM_N1K3, EPGLMM_N3K1, EPGLMM_N1K3)
        values = self.reader[:, :4].read(xp=np, dtype=np.float32).val
        covariates = np.ones((40, 1), dtype=np.float32)
        labels = np.tile([-1.0, 1.0], 20).astype(np.float32)
        for model_type in (LMM, LaplaceGLMM_N3K1, LaplaceGLMM_N1K3, EPGLMM_N3K1, EPGLMM_N1K3):
            with self.subTest(model=model_type.__name__), patch.dict(os.environ, {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}), \
                 warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = model_type() if model_type is LMM else model_type("logistic" if model_type in (LaplaceGLMM_N3K1, LaplaceGLMM_N1K3) else "erf")
                model.setG(values)
                model.setX(covariates)
                model.sety(labels)
                model_values = model.G if model_type is LMM else model._G0
                self.assertIsInstance(model_values, np.ndarray)
                self.assertEqual(model_values.dtype, np.float64)
                self.assertEqual((model.X if model_type is LMM else model._X).dtype, np.float64)
                if model_type is LMM:
                    self.assertTrue(np.isfinite(model.nLLeval(h2=0.3)["nLL"]))
                else:
                    model.sig02 = 0.4
                    model.sign2 = 0.6
                    model.beta = np.zeros(1)
                    self.assertTrue(np.isfinite(model.marginal_loglikelihood()))
                self.assertTrue(any("qualified only" in str(item.message) for item in caught))
                self.assertEqual(os.environ["ARRAY_MODULE"], "cupy")
                self.assertEqual(os.environ["ARRAY_DTYPE"], "float32")
        with patch.dict(os.environ, {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsInstance(getLMM(), LMM)
            self.assertIsInstance(getGLMM("ep", "erf", labels, values, None), EPGLMM_N1K3)
            variance = association.varcomp_test(labels, covariates)
            self.assertTrue(variance.hasBias())
            self.assertEqual(variance.Y.dtype, np.float64)
            with self.assertRaises(NotImplementedError):
                variance.testG(values)
        for api in ("getLMM", "getGLMM", "varcomp_test"):
            self.assertTrue(any(api in str(item.message) for item in caught))

    def test_unqualified_worker_failure_restores_caller_settings(self):
        import json
        trace = self.folder / "failed-worker"
        reader = _FailingPolicyBed(self.path, trace)
        runner = LocalMultiProc(1, num_threads=1, taskindex_to_environ=_unqualified_worker)
        before = Path.cwd()
        settings = {"ARRAY_MODULE": "cupy", "ARRAY_DTYPE": "float32"}
        try:
            os.chdir(self.folder)
            with patch.dict(os.environ, settings), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaisesRegex(Exception, "non-zero return code"):
                    association.single_snp_linreg(reader, self.pheno, runner=runner)
                self.assertEqual({key: os.environ[key] for key in settings}, settings)
                self.assertTrue(any("single_snp_linreg" in str(item.message) and "numpy/float64" in str(item.message) for item in caught))
        finally:
            os.chdir(before)
        records = [json.loads(line) for path in self.folder.glob("failed-worker.*.jsonl") for line in path.read_text().splitlines()]
        self.assertTrue(any(record["pid"] != os.getpid() for record in records))

    def _single_kernel_matrix(self, backend):
        import json
        import importlib
        from pysnptools.util import asnumpy
        from fastlmm.inference.lmm_cov import LMM
        from fastlmm.association.tests.test_backend_environment import _TracingBed

        original_decompose = LMM.getSU
        module = importlib.import_module("fastlmm.association.single_snp")
        original_tester = module._snp_tester
        for dtype in ("float32", "float64"):
            for low_rank in (False, True):
                for loco in (False, True):
                    with self.subTest(backend=backend, dtype=dtype, low_rank=low_rank, loco=loco):
                        kernel = self.reader[:, :6] if low_rank else self.reader
                        options = dict(K0=kernel, covar=self.covar, leave_out_one_chrom=loco,
                                       force_full_rank=not low_rank, force_low_rank=low_rank, h2=0.3)
                        reference = association.single_snp(self.reader, self.pheno, xp=np, dtype="float64", **options)
                        decompositions = []

                        def decompose(model, *args, **kwargs):
                            result = original_decompose(model, *args, **kwargs)
                            for array in (model.U, model.S, model.X, model.Y):
                                self.assertEqual(get_array_module(array).__name__, backend)
                                self.assertEqual(array.dtype, np.dtype(dtype))
                            decompositions.append(model.U.shape)
                            return result

                        cache = str(self.folder / f"{backend}-{dtype}-{low_rank}-{loco}.npz")
                        models = {"fresh": [], "cached": []}
                        stage = "fresh"

                        def tester(*args, **kwargs):
                            model = args[3]
                            models[stage].append({name: None if getattr(model, name) is None else asnumpy(getattr(model, name)).copy()
                                                  for name in ("S", "U", "UY", "UUY", "X", "Y")})
                            return original_tester(*args, **kwargs)

                        with patch.object(LMM, "getSU", decompose), patch.object(module, "_snp_tester", tester):
                            first = association.single_snp(self.reader, self.pheno, xp=backend, dtype=dtype,
                                                           cache_file=cache, **options)
                            stage = "cached"
                            cached = association.single_snp(self.reader, self.pheno, xp=backend, dtype=dtype,
                                                            cache_file=cache, **options)
                        self.assertTrue(decompositions)
                        self.assertEqual(len(models["fresh"]), len(models["cached"]))
                        for fresh_model, cached_model in zip(models["fresh"], models["cached"]):
                            for name in fresh_model:
                                np.testing.assert_array_equal(fresh_model[name], cached_model[name])
                        # Serialization packs a strided eigensolver view. Verified
                        # CPU/CuPy controls give bitwise-identical model values but
                        # different BLAS reduction orders (see qualification report).
                        # Both executions must meet the existing reference bounds.
                        np.testing.assert_array_equal(first.columns, cached.columns)
                        np.testing.assert_array_equal(first.SNP, cached.SNP)
                        np.testing.assert_allclose(np.log10(cached.sort_values("sid_index").PValue),
                                                   np.log10(reference.sort_values("sid_index").PValue),
                                                   rtol=0, atol=1.5e-4 if dtype == "float32" else 2e-8)
                        reference = reference.sort_values("sid_index")
                        actual = first.sort_values("sid_index")
                        np.testing.assert_array_equal(actual.SNP, reference.SNP)
                        np.testing.assert_array_equal(actual[["Chr", "ChrPos", "sid_index"]], reference[["Chr", "ChrPos", "sid_index"]])
                        np.testing.assert_array_equal(np.isfinite(actual.PValue), np.isfinite(reference.PValue))
                        self.assertTrue(np.all((actual.PValue > 0) & (actual.PValue <= 1)))
                        # Established backend bound, with fixed h2 to isolate linear
                        # algebra from independently fitted nearly tied optima.
                        np.testing.assert_allclose(np.log10(actual.PValue), np.log10(reference.PValue),
                                                   rtol=0, atol=1.5e-4 if dtype == "float32" else 2e-8)

            # Real worker processes, observing phenotype and test-SNP arrays in
            # LMM.nLLeval rather than the host DataFrame returned by the API.
            trace = self.folder / f"trace-{backend}-{dtype}"
            reader = _TracingBed(self.path, trace)
            before = Path.cwd()
            with patch.object(LMM, "nLLeval", LMM.nLLeval):
                try:
                    os.chdir(self.folder)
                    result = association.single_snp(reader, self.pheno, xp=backend, dtype=dtype, covar=self.covar,
                        h2=0.3, force_full_rank=True, runner=LocalMultiProc(2, num_threads=1), map_reduce_outer=True)
                finally:
                    os.chdir(before)
            records = [json.loads(line) for path in self.folder.glob(trace.name + ".*.jsonl") for line in path.read_text().splitlines()]
            workers = [record for record in records if record["pid"] != os.getpid() and record["kind"] in ("evaluate", "test_snps")]
            self.assertTrue(workers)
            self.assertEqual({record["backend"] for record in workers}, {backend})
            self.assertEqual({record["actual_dtype"] for record in workers}, {dtype})
            reference = association.single_snp(self.reader, self.pheno, xp="numpy", dtype="float64", covar=self.covar, h2=0.3, force_full_rank=True)
            np.testing.assert_allclose(np.log10(result.sort_values("sid_index").PValue), np.log10(reference.sort_values("sid_index").PValue),
                                       rtol=0, atol=1.5e-4 if dtype == "float32" else 2e-8)

    def test_single_kernel_numpy_matrix(self):
        self._single_kernel_matrix("numpy")

    @unittest.skipUnless(has_gpu_support() and find_spec("cupy"), "requires CUDA bed-reader and CuPy")
    def test_single_kernel_cupy_matrix(self):
        self._single_kernel_matrix("cupy")

    @unittest.skipUnless(has_gpu_support() and find_spec("cupy"), "requires CUDA bed-reader and CuPy")
    def test_gpu_input_and_plot_contract(self):
        import cupy as cp
        original = self.reader.read(xp=cp, dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = association.single_snp_linreg(original, self.pheno)
        self.assertTrue(any("supplied" in str(item.message) for item in caught))
        self.assertIsInstance(original.val, cp.ndarray)
        self.assertEqual(original.val.dtype, np.float32)
        self.assertTrue(np.isfinite(result.PValue).all())
        reference = association.single_snp_linreg(original.read(xp=np, dtype=np.float64), self.pheno)
        import pandas as pd
        pd.testing.assert_frame_equal(result, reference, check_exact=True)
        with self.assertRaisesRegex(TypeError, "requires host data"):
            manhattan_plot(cp.ones((2, 3), dtype=cp.float32))


def getTestSuite():
    return unittest.defaultTestLoader.loadTestsFromTestCase(TestApiPolicy)
