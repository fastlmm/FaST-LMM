"""
file to set up python package, see http://docs.python.org/2/distutils/setupscript.html for details.
"""


from __future__ import absolute_import
from __future__ import print_function
import platform
import os
import sys
import shutil

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.clean import clean as Clean

try:
    from Cython.Distutils import build_ext
except Exception:
    print("cython needed for installation, please install cython first")
    sys.exit()

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
        for dirpath, dirnames, filenames in os.walk('fastlmm'):
            for filename in filenames:
                if (filename.endswith('.so') or filename.endswith('.pyd')
                             #or filename.endswith('.dll')
                             #or filename.endswith('.pyc')
                             ):
                    os.unlink(os.path.join(dirpath, filename))

# set up macro
if platform.system() == "Darwin":
    macros = [("__APPLE__", "1")]
elif platform.system() == "Windows":
    macros = [("_WIN32", "1")]

ext = [Extension("fastlmm.util.stats.quadform.qfc_src.wrap_qfc", ["fastlmm/util/stats/quadform/qfc_src/wrap_qfc.pyx", "fastlmm/util/stats/quadform/qfc_src/QFC.cpp"], language="c++", define_macros=macros)]

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
        "fastlmm/util/stats/quadform",
        "fastlmm/util/stats",
        "fastlmm/util",
        "fastlmm"
    ],
    install_requires=['cython', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib'],
    #zip_safe=False,
    # extensions
    cmdclass = {'build_ext': build_ext, 'clean': CleanCommand},
    ext_modules = ext,
    include_dirs = [numpy.get_include()],
  )

