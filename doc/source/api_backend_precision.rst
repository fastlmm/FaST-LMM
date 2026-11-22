Public API backend and precision policy
=======================================

This checklist follows :doc:`index`, including the documented predictor methods,
and reconciles the exports in ``fastlmm.association``, ``fastlmm.inference``, and
``fastlmm.util``. See :doc:`gpu` for installation and numerical limitations.
These are development-source guarantees, not acceptance of a new published alpha.

``single_snp`` with one kernel supports NumPy/CuPy and float32/float64.
An explicit ``xp`` or ``dtype`` overrides the corresponding environment variable;
otherwise ``ARRAY_MODULE`` and ``ARRAY_DTYPE`` apply, finally defaulting to
NumPy/float64. Missing CuPy or an unusable CUDA device raises an error.
Returning a pandas DataFrame does not imply CPU computation.

The numerical workflows below that are qualified only on CPU/float64 warn when
asked for another configuration. Each warning names the affected API and the
actual fallback, ``numpy/float64``. Default CPU/float64 calls have no fallback
warning. Invalid configuration names raise an error. APIs without ``xp`` or
``dtype`` parameters retain their signatures; their configuration comes from the
environment and supplied data.

.. list-table:: Documented API checklist
   :header-rows: 1
   :widths: 29 26 45

   * - API
     - Actual numerical execution
     - Contract and direct regression coverage
   * - ``single_snp`` (one kernel)
     - Requested NumPy/CuPy, float32/float64
     - Full/low rank, LOCO, covariates, cache creation/reuse, and real workers.
       ``test_api_policy`` checks model and SNP arrays inside computation;
       ``test_backend_environment`` also checks mixed CPU/GPU workers.
   * - ``single_snp`` (``K1`` or ``G1`` supplied)
     - NumPy/float64
     - Warn on an unsupported explicit or environmental request. CPU scopes
       also surround chromosome and SNP-test workers. A float32 cache is rejected.
   * - ``single_snp_scale``
     - NumPy/float64
     - Warned fallback, including stage workers and CPU file-backed arrays.
       ``test_api_policy`` and ``test_single_snp_scale`` cover fresh/reused caches.
   * - ``single_snp_all_plus_select``
     - NumPy/float64
     - Warned fallback includes selection, prediction, and final association;
       nested workers establish their own scopes. Direct policy and legacy tests.
   * - ``single_snp_linreg``
     - NumPy/float64
     - Warned fallback includes SNP-block workers. Supplied float32/device data
       is copied with a warning; the caller's arrays are preserved.
   * - ``single_snp_select``
     - NumPy/float64
     - Deprecated status and platform limitations remain. Direct policy tests
       exercise selection on a small data set; existing native limitations remain.
   * - ``epistasis``
     - NumPy/float64
     - Warned fallback includes worker initialization and pair testing. Cached
       float32 eigenvectors/eigenvalues raise an error requiring cache recreation.
       Legacy fixtures and the redundant-interaction regression remain.
   * - ``snp_set``
     - NumPy/float64
     - Warned fallback around the legacy local runner. Direct score-test coverage
       and existing likelihood-ratio/C++ fixture tests.
   * - ``FastLMM.__init__``
     - Configuration only
     - No numerical solve; existing constructor options and signature preserved.
   * - ``FastLMM.fit``, ``predict``, ``score``
     - NumPy/float64
     - Each method applies the policy independently, including calls on already
       fitted models. Float32/device model state requires refitting. Direct tests
       check model arrays, predictive mean/covariance, scores, and restoration.
   * - ``LinearRegression.__init__``
     - Configuration only
     - No numerical solve; existing standardizer option preserved.
   * - ``LinearRegression.fit``, ``predict``, ``score``
     - NumPy/float64
     - Each method applies the policy, warns on fallback, and rejects unqualified
       stored model arrays. Direct prediction/score/restoration regressions.
   * - ``heritability_spatial_correction``
     - NumPy/float64
     - Warned fallback includes calls sent to ``map_function``. Direct policy
       tests and existing numerical fixture tests.
   * - ``compute_auto_pcs``
     - NumPy/float64; native executable limitations remain
     - Deprecated. Warned fallback occurs before reading/launching work. The
       existing Intel-only/AMD limitation and unavailable-native-test warning
       remain. Direct failure/restoration checks do not qualify the executable.
   * - ``manhattan_plot``
     - Host plotting, no numerical backend selection
     - Accepts host arrays/DataFrames, including float32 data, independent of
       environment defaults. Device arrays raise an informative ``TypeError``;
       transfer with ``cupy.asnumpy``. No artificial precision override.

Other exported entry points
---------------------------

* ``single_snp_leave_out_one_chrom`` is a deprecated compatibility alias for
  ``single_snp`` and uses its identical backend/precision policy.
* ``varcomp_test`` is a legacy base class. Construction normalizes numerical
  inputs on CPU/float64 with a warning when necessary. ``hasBias`` observes the
  same scope; the base ``testG`` remains unimplemented and raises
  ``NotImplementedError``. It does not promise a GPU score test.
* ``LMM``, ``getLMM``, ``getGLMM``, ``LaplaceGLMM_N3K1``,
  ``LaplaceGLMM_N1K3``, ``EPGLMM_N3K1``, and ``EPGLMM_N1K3`` are legacy
  inference exports. Numerical setters, fitting, likelihood, and prediction
  methods use the CPU/float64 boundary; the factories do too. Constructors and
  nonnumerical properties retain their established behavior. Internal
  ``fastlmm.inference.lmm_cov.LMM`` is the separate backend-aware implementation
  used by single-kernel ``single_snp``.
* ``example_file`` downloads/locates data; it does not select a numerical backend
  or precision. Root exports ``OutputWriter`` and ``Pr`` are deprecated logging
  utilities, also independent of numerical configuration.

Inputs, caches, workers, and scoped settings
--------------------------------------------

Fallback scopes share one reentrant lock with ``single_snp``. They resolve
configuration after acquiring it and restore the selected environment variables
on success and on exceptions. Nested public calls already see CPU/float64 and
therefore do not emit another fallback warning. Unrelated code that changes
``os.environ`` is outside this coordination; do not edit process defaults while
an analysis is running.

Supplied in-memory float32 or CuPy arrays/readers are copied to host float64 with
an input-conversion warning at unqualified boundaries. Existing input rounding
cannot be recovered. Lazy readers are read under the scoped configuration.
Metadata such as sample/SNP identifiers remains on the host. A fitted predictor
or epistasis cache with unqualified numerical storage is rejected rather than
silently widened. Refit or recreate it using the qualified API.

Caches still require matching samples, phenotypes, covariates, standardizers,
and statistical settings; they have no input-content fingerprint. Clear caches
created by earlier unqualified float32 workflows, especially legacy scale
caches whose file-backed storage alone cannot establish construction precision.
Single-kernel ``single_snp`` checks all cached model-array dtypes. Serialization
can pack strided eigensolver views and change BLAS reduction order without
changing any stored values; cache reuse does not promise bitwise-identical
P-values. Both fresh and cached executions must satisfy the numerical reference
criteria; cache model values are checked exactly.

Workers establish fallback scopes inside their serialized computation functions,
including when a runner overrides ``ARRAY_MODULE`` or ``ARRAY_DTYPE``. For
single-kernel ``single_snp``, workers may deliberately select different CPU/GPU
backends; workflow precision must agree or an informative error is raised.
Parent-process settings alone do not qualify worker execution.

Numerical qualification limits
------------------------------

Float32 support describes the main model arrays and rotations. Established
float64 per-SNP statistics, CPU kernel formation, and backend-dependent internal
solver precision are deliberate and remain documented in :doc:`gpu`.

The November 18 numerical investigation retained an independent float64 fitted-h2
comparison failure: maximum synthetic absolute delta log10(P) was ``5.36e-8``
against ``2e-8``. Shared CPU64 h2 reduced it to ``2.49e-14``. Mouse float32
CuPy eigensolver differences reached about ``0.0004`` in log10(P); the concrete
example and accepted alpha precision exception are described in :doc:`gpu`.
These are historical controls, not passing strict comparisons or universal error
bounds. The investigation's original epistasis float32 failures also remain
recorded. Routing epistasis to CPU/float64 does not fix or qualify its float32
algorithm. No established accuracy tolerance or expected fixture was widened.

Development validation and remaining gates are recorded in
``specs/GPU_API_POLICY_STATUS.md`` and ``specs/todo.md``. Full notebook execution
is additional integration coverage; commented deprecated examples are not tests.
