"""Central unit-degree quadratic-form tails used by the SNP-set score test.

The absolute accuracy target is 1e-7, matching the historical Davies adapter.
A flagged gx2 result is accepted only when a Chernoff bound independently
establishes that accuracy. gx2 warnings remain visible, including clipping
warnings for results accepted by that bound. No random sampling is used.
"""

import gx2
import numpy as np


def sf(statistic, coefficients):
    """Return the upper tail or raise ArithmeticError for unverified results.

    Preserve the historical point-mass convention: an all-zero (or empty)
    spectrum returns one at zero. Other distributions use P(Q > statistic).
    Coefficients are not filtered: negative roundoff eigenvalues are retained.
    """
    statistic = np.asarray(statistic)
    coefficients = np.asarray(coefficients)
    if (statistic.ndim != 0 or statistic.dtype.kind not in "fiu"
            or coefficients.ndim != 1 or coefficients.dtype.kind not in "fiu"
            or not np.isfinite(statistic) or not np.isfinite(coefficients).all()):
        raise ArithmeticError("Quadratic-form inputs must be a finite real scalar and a finite real vector")
    statistic = float(statistic)
    if not np.any(coefficients):
        return float(statistic <= 0)
    # Exact support checks avoid unnecessary numerical integration at edges.
    if statistic <= 0 and np.all(coefficients >= 0):
        return 1.0
    if statistic >= 0 and np.all(coefficients <= 0):
        return 0.0

    accuracy = 1e-7
    probability, diagnostic, _ = gx2.cdf(
        statistic, coefficients, np.ones(len(coefficients)),
        np.zeros(len(coefficients)), 0, 0, side="upper", method="auto",
        full_output=True, AbsTol=1e-14, RelTol=1e-12,
    )
    probability = np.asarray(probability)
    if (probability.ndim != 0 or probability.dtype.kind not in "fiu"
            or not np.isfinite(probability) or not 0 <= probability <= 1):
        raise ArithmeticError(f"gx2 returned an invalid probability: {probability}")
    probability = float(probability)
    if diagnostic is None:
        # gx2's exact chi-square dispatch has no diagnostic output.
        return probability
    diagnostic = np.asarray(diagnostic)
    if diagnostic.ndim != 0:
        raise ArithmeticError(f"gx2 returned a nonscalar diagnostic: {diagnostic}")
    if diagnostic.dtype.kind == "b":
        if not diagnostic:
            return probability
        log_bound = _log_upper_bound(statistic, coefficients)
        # If 0 <= p <= B and the true tail is in [0, B], then |p - tail|
        # is at most B. Compare in log space so a tiny bound cannot underflow
        # to a falsely exact zero. A true flag alone never establishes this.
        if (log_bound <= np.log(accuracy)
                and (probability == 0 or np.log(probability) <= log_bound)):
            return probability
        raise ArithmeticError(f"gx2 flagged an unreliable tail; diagnostic={diagnostic}, log upper bound={log_bound}")
    # Ruben's diagnostic is an absolute error bound, not a boolean flag.
    if (diagnostic.dtype.kind not in "fiu" or not np.isfinite(diagnostic)
            or not 0 <= diagnostic <= accuracy):
        raise ArithmeticError(f"gx2 failed the absolute accuracy requirement; diagnostic={diagnostic}")
    return probability


def _log_upper_bound(statistic, coefficients):
    """Bound a central unit-degree upper tail with its moment-generating function."""
    # Normalize by the largest absolute coefficient, preserving signs. The
    # fixed t=0.45 is strictly below every positive MGF pole because each
    # normalized coefficient is <= 1. Negative coefficients are valid too.
    scale = np.max(np.abs(coefficients.astype(float)))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return float(-0.45 * (statistic / scale)
                         - 0.5 * np.log1p(-0.9 * (coefficients / scale)).sum())
    except FloatingPointError as error:
        raise ArithmeticError("Could not evaluate the quadratic-form tail bound") from error
