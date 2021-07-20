"""
file to set up python package, see http://docs.python.org/2/distutils/setupscript.html for details.
"""


import platform
import os
import sys
import shutil

from distutils.core import setup
from distutils.command.clean import clean as Clean

try:
    import numpy
except Exception:
    print("numpy needed for installation, please install numpy first")
    sys.exit()


def readme():
    with open('README.md') as f:
       return f.read()


class CleanCommand(Clean):
    description = "Remove build directories, and compiled files (including .pyc)"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')


setup(
    name='fastlmm',
    version='0.1',
    description='Fast GWAS',
    long_description=readme(),
    keywords='gwas bioinformatics LMMs MLMs',
    url='',
    author='FaST-LMM Team',
    author_email='...',
    packages=[
        "fastlmm/association/tests",
        "fastlmm/association",
        "fastlmm/external/sklearn/externals",
        "fastlmm/external/sklearn/metrics",
        "fastlmm/external/sklearn",
        "fastlmm/external/util",
        "fastlmm/external",
        "fastlmm/feature_selection",
        "fastlmm/inference/bingpc",
        "fastlmm/inference",
        "fastlmm/pyplink/altset_list",
        "fastlmm/pyplink/snpreader",
        "fastlmm/pyplink/snpset",
        "fastlmm/pyplink",
        "fastlmm/util/stats",
        "fastlmm/util",
        "fastlmm"
    ],
    install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib'],
    cmdclass = {'clean': CleanCommand},
    include_dirs = [numpy.get_include()],
  )

