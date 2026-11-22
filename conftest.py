"""Keep standalone scripts and legacy helpers out of pytest collection."""

collect_ignore = [
    # Manual performance experiments with CPU and GPU backends.
    "fastlmm/association/tests/test_gpu_perf.py",
    # Executable configs, optional Azure helpers, and old packaging scripts.
    "fastlmm/feature_selection/examples/ScanISP.Toydata.config.py",
    "fastlmm/feature_selection/examples/ScanLMM.Toydata.config.py",
    "fastlmm/feature_selection/examples/ScanOSP.Toydata.config.py",
    "fastlmm/ludicrous2/azurehelper.py",
    "fastlmm/ludicrous2/blobxfer.py",
    "fastlmm/pyplink/setup.py",
    "fastlmm/pyplink/vcfpy.py",
]
