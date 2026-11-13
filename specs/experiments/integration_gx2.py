"""Evaluate gx2 on real score-test calls without changing production source."""
import json
import os
import sys
import time
import unittest
import warnings
from pathlib import Path

import gx2
import numpy as np
from fastlmmclib.quadform import qf


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from fastlmm.association.tests.Sc import Sc
    from fastlmm.association.tests.test_snp_set import TestSnpSet

    baseline = os.environ.get("GX2_BASELINE") == "1"
    output = Path(os.environ.get("GX2_EVALUATION_OUTPUT", "/tmp/fastlmm-maintenance-review"))
    output.mkdir(parents=True, exist_ok=True)
    filename = "qfc-baseline-inputs.jsonl" if baseline else "gx2-real-inputs.jsonl"
    with (output / filename).open("w") as captured:
        def evaluated(cutoff, weights):
            start = time.perf_counter()
            old, fault, _ = qf(cutoff, weights, acc=1e-7)
            old_elapsed = time.perf_counter() - start
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                probability, error, _ = gx2.cdf(
                    cutoff, weights, np.ones(len(weights)), np.zeros(len(weights)),
                    0, 0, side="upper", full_output=True,
                )
            row = dict(
                cutoff=float(cutoff), weights=weights.tolist(), qfc=float(old),
                fault=int(fault), gx2=float(probability),
                gx2_error=None if error is None else np.asarray(error).tolist(),
                qfc_seconds=old_elapsed, gx2_seconds=time.perf_counter()-start,
                warnings=[str(w.message) for w in caught],
            )
            captured.write(json.dumps(row) + "\n")
            captured.flush()
            return float(old if baseline else probability)

        Sc.pv_davies_eig = staticmethod(evaluated)
        suite = unittest.TestSuite([TestSnpSet("test_three"), TestSnpSet("test_four")])
        os.chdir(root / "tests")
        sys.path.insert(0, str(root / "tests"))
        from test import WidgetTestCase

        suite.addTests(WidgetTestCase(p.name) for p in sorted((root / "tests/inputs").glob("*davies*.py")))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(main())
