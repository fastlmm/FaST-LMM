# pytest collects only test_*.py files, so this module is pytest's single entry point for the
# feature-selection tests in test.py (which the legacy tests/test.py runs directly).
from fastlmm.feature_selection.test import TestFeatureSelection  # noqa: F401
