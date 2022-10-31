import platform
import os
import shutil
from setuptools import setup
import numpy

# Work around https://github.com/pypa/pip/issues/7953
import site
import sys

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Version number
version = "0.6.4"


def readme():
    with open("README.md") as f:
        return f.read()


# python setup.py sdist bdist_wininst upload
setup(
    name="fastlmm",
    version=version,
    description="Fast GWAS",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords="gwas bioinformatics LMMs MLMs linear mixed models genomics genetics python",
    url="https://fastlmm.github.io/",
    author="FaST-LMM Team",
    author_email="fastlmm-dev@python.org",
    project_urls={
        "Bug Tracker": "https://github.com/fastlmm/FaST-LMM/issues",
        "Documentation": "http://fastlmm.github.io/FaST-LMM",
        "Source Code": "https://github.com/fastlmm/FaST-LMM",
    },
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python",
    ],
    packages=[  # basically, everything with a __init__.py
        "fastlmm",
        "fastlmm/association",
        "fastlmm/association/altset_list",
        "fastlmm/association/tests",
        "fastlmm/external",
        "fastlmm/external/util",
        "fastlmm/feature_selection",
        "fastlmm/inference",
        "fastlmm/inference/tests",
        "fastlmm/pyplink",  # old snpreader
        "fastlmm/pyplink/altset_list",  # old snpreader
        "fastlmm/pyplink/snpreader",  # old snpreader
        "fastlmm/pyplink/snpset",  # old snpreader
        "fastlmm/util",
        "fastlmm/util/matrix",
        "fastlmm/util/standardizer",
        "fastlmm/util/stats",
    ],
    package_data={
        "fastlmm/association": [
            "Fastlmm_autoselect/FastLmmC.exe",
            "Fastlmm_autoselect/libiomp5md.dll",
            "Fastlmm_autoselect/fastlmmc",
            "Fastlmm_autoselect/FastLmmC.Manual.pdf",
        ],
        "fastlmm/feature_selection": [
            "examples/bronze.txt",
            "examples/ScanISP.Toydata.config.py",
            "examples/ScanLMM.Toydata.config.py",
            "examples/ScanOSP.Toydata.config.py",
            "examples/toydata.5chrom.bed",
            "examples/toydata.5chrom.bim",
            "examples/toydata.5chrom.fam",
            "examples/toydata.cov",
            "examples/toydata.map",
            "examples/toydata.phe",
            "examples/toydata.shufflePlus.phe",
            "examples/toydata.sim",
            "examples/toydataTest.phe",
            "examples/toydataTrain.phe",
        ],
        "fastlmm": ["util/fastlmm.hashdown.json"],
    },
    install_requires=[
        "pandas>=1.1.1",
        "matplotlib>=1.5.1",
        "scikit-learn>=0.19.1",
        "pysnptools>=0.5.7",
        "cloudpickle>=2.2.0",
        "statsmodels>=0.10.1",
        "psutil>=5.6.7",
        "fastlmmclib>=0.0.2",
    ],
)
