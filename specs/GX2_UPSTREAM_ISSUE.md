# Proposed title

Inaccurate upper-tail probabilities with positive weights after auto falls back to Imhof

# Proposed body

While evaluating gx2 1.5 as a quadratic-form backend for FaST-LMM, I found
two small central chi-square cases where `cdf(..., side="upper",
method="auto")` returns substantially inaccurate probabilities with integration
warnings. Both cases use only positive weights, unit degrees of freedom,
zero noncentrality, and no additional normal term or offset.

The expected answers below follow from an elementary exponential convolution;
they do not depend on another generalized-chi-square implementation.

## Reproducer

```python
import math
import warnings
from importlib.metadata import version

import gx2

for package in ("gx2", "numpy", "scipy", "mpmath"):
    print(f"{package}=={version(package)}")

x = 20.0
for a in (1e-3, 1e-8):
    weights = [a, a, 1.0, 1.0]
    expected = (math.exp(-x / 2) - a * math.exp(-x / (2 * a))) / (1 - a)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = gx2.cdf(
            x,
            weights,
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            0.0,
            0.0,
            side="upper",
            method="auto",
        )
    print(f"a={a:g}: expected={expected:.16g}, actual={float(actual):.16g}")
    for warning in caught:
        print(f"  {warning.category.__name__}: {str(warning.message).splitlines()[0]}")
```

Environment: CPython 3.15.2, Linux x86-64/WSL2, gx2 1.5, NumPy 2.5.4,
SciPy 1.18.1, mpmath 1.4.1. The gx2 wheel was the published
`gx2-1.5-py3-none-any.whl`.

## Observed results

| `a` | Expected upper tail | gx2 auto |
| --- | ---: | ---: |
| `1e-3` | `4.544537513762248e-5` | `8.146527586513352e-4` |
| `1e-8` | `4.539993021648416e-5` | `0` |

Both calls emit `IntegrationWarning` about reaching 50 subdivisions. The
second also emits `ImhofClipWarning`.

## Derivation of the reference answer

For independent standard normals `Z1` through `Z4`, the variable is

`Q = a * (Z1**2 + Z2**2) + (Z3**2 + Z4**2)`.

Each pair of squared normals is an independent chi-square with two degrees
of freedom. Thus `Q` is the sum of independent exponential variables with
means `2*a` and `2`. For `x >= 0` and `0 < a < 1`, its exact survival function
is

`P(Q > x) = (exp(-x/2) - a*exp(-x/(2*a))) / (1-a)`.

## Investigation

From inspecting the installed 1.5 source, these inputs appear to exhaust
Ruben's default series budget, after which `auto` falls back to Imhof. Scalar
basic Imhof integration calls `scipy.integrate.quad` with its default
subdivision limit. The warning indicates that this integration has not
converged, and the clipped result can be far from the true probability.
This is a working diagnosis rather than a proposed fix.

Tightening `AbsTol`/`RelTol` alone did not resolve the problem. In separate
experiments, increasing the quadrature budget also did not reliably meet
the requested tolerance. I have not established which alternative gx2
method or configuration is best for this class of inputs.

## Suggested next steps

I suggest adding these two analytic cases as regression tests for `auto`
and its Imhof fallback, then addressing convergence reporting separately
from improving the integration:

1. Preserve the quadrature error estimate and convergence status. When the
   requested accuracy cannot be established, report that explicitly through
   a documented failure status or exception. Callers should be able to
   distinguish a genuinely tiny probability from a failed integration that
   was clipped to zero; clipping alone cannot establish accuracy.
2. Evaluate an integration strategy designed for the oscillatory integrand,
   with controlled truncation and integration error, rather than relying
   only on tighter tolerances or a larger subdivision budget. Davies-style
   error control may be a useful reference, although I have not established
   the best implementation for gx2.

If an existing method/configuration already handles both cases reliably,
documenting and testing that route in `auto` would also address the issue.
For our use case, an explicit nonconvergence result would already be a useful
improvement: it would let us handle the failure without treating the returned
number as a validated p-value.
