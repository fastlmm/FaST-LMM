from __future__ import absolute_import
from __future__ import print_function
import platform
import os
import sys
import shutil
from setuptools import setup, Extension 
from distutils.command.clean import clean as Clean
import numpy

# Version number
version = '0.4.8'


def readme():
    with open('README.md') as f:
       return f.read()

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

class CleanCommand(Clean):
    description = "Remove build directories, and compiled files (including .pyc)"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if (   (filename.endswith('.so') and not filename.startswith('libmkl_core.'))#!!!cmk
                    or filename.endswith('.pyd')
                    or (use_cython and filename.find("wrap_qfc.cpp") != -1) # remove automatically generated source file
                    or (use_cython and filename.find("cample.cpp") != -1) # remove automatically generated source file
                    or (use_cython and filename.find("mmultfilex.cpp") != -1) # remove automatically generated source file
                    or filename.endswith('.pyc')
                                ):
                    tmp_fn = os.path.join(dirpath, filename)
                    print("removing", tmp_fn)
                    os.unlink(tmp_fn)

# set up macros
if platform.system() == "Darwin":
    macros = [("__APPLE__", "1")]
    cmkintel_root = os.path.join(os.path.dirname(__file__),"external/intel/linux")
    cmkmp5lib = 'iomp5'
    cmkmkl_core = 'mkl_core'
    extra_compile_args0 = []
    cmkextra_compile_args1 = ['-DMKL_ILP64','-fpermissive']
    cmkextra_compile_args2 = ['-fopenmp', '-DMKL_LP64','-fpermissive']
elif "win" in platform.system().lower():
    macros = [("_WIN32", "1")]
    blas_root = os.path.join(os.path.dirname(__file__),"external/openblas/windows")
    mp5lib = 'libiomp5md' #!!!cmk
    blas_core = 'openblas_core_dll' #!!!cmk
    extra_compile_args0 = ['/EHsc']
    extra_compile_args1 = ['/DLAPACK_ILP64', '/DHAVE_LAPACK_CONFIG_H'] #cmk LAPACK_ILP64 seems to be 'long' not 'long long int'
    extra_compile_args2 = ['/EHsc', '/openmp', '/DLAPACK_LP64', '/DHAVE_LAPACK_CONFIG_H']
else:
    macros = [("_UNIX", "1")]
    cmkintel_root = os.path.join(os.path.dirname(__file__),"external/intel/linux")
    mp5lib = 'iomp5'
    cmkmkl_core = 'mkl_core'
    extra_compile_args0 = []
    cmkextra_compile_args1 = ['-DMKL_ILP64','-fpermissive']
    cmkextra_compile_args2 = ['-fopenmp', '-DMKL_LP64','-fpermissive']

blas_library_list = [blas_root+"/lib"]
print(blas_library_list)
blas_include_list = [blas_root+"/include"]
runtime_library_dirs = None if "win" in platform.system().lower() else blas_library_list

#see http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
print("use_cython? {0}".format(use_cython))
if use_cython:
    ext_modules = [Extension(name="fastlmm.util.stats.quadform.qfc_src.wrap_qfc",
                             language="c++",
                             sources=["fastlmm/util/stats/quadform/qfc_src/wrap_qfc.pyx", "fastlmm/util/stats/quadform/qfc_src/QFC.cpp"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args = extra_compile_args0,
                             define_macros=macros),
                   Extension(name="fastlmm.util.matrix.cample",
                            language="c++",
                            sources=["fastlmm/util/matrix/cample.pyx"],
                            libraries = ['openblas'],
                            library_dirs = blas_library_list,
                            runtime_library_dirs = runtime_library_dirs,
                            include_dirs = blas_include_list+[numpy.get_include()],
                            extra_compile_args = extra_compile_args1,
                            define_macros=macros),
                    Extension(name="fastlmm.util.matrix.mmultfilex",
                            language="c++",
                            sources=["fastlmm/util/matrix/mmultfilex.pyx","fastlmm/util/matrix/mmultfile.cpp"],
                            libraries = ['openblas'],
                            library_dirs = blas_library_list,
                            runtime_library_dirs = runtime_library_dirs,
                            include_dirs = blas_include_list+[numpy.get_include()],
                            extra_compile_args = extra_compile_args2,
                            define_macros=macros)
                     ]
    cmdclass = {'build_ext': build_ext, 'clean': CleanCommand}
else:
    assert false, "!!!cmk Need to update"
    ext_modules = [Extension(name="fastlmm.util.stats.quadform.qfc_src.wrap_qfc",
                             language="c++",
                             sources=["fastlmm/util/stats/quadform/qfc_src/wrap_qfc.cpp", "fastlmm/util/stats/quadform/qfc_src/QFC.cpp"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args = extra_compile_args0,
                             define_macros=macros),
                   Extension(name="fastlmm.util.matrix.cample",
                            language="c++",
                            sources=["fastlmm/util/matrix/cample.cpp"],
                            libraries = ['mkl_intel_ilp64', mkl_core, 'mkl_intel_thread', mp5lib],
                            library_dirs = mkl_library_list,
                            include_dirs = mkl_include_list+[numpy.get_include()],
                            extra_compile_args = extra_compile_args1,
                            define_macros=macros),
                    Extension(name="fastlmm.util.matrix.mmultfilex",
                            language="c++",
                            sources=["fastlmm/util/matrix/mmultfilex.cpp","fastlmm/util/matrix/mmultfile.cpp"],
                            libraries = ['mkl_intel_lp64', mkl_core, 'mkl_intel_thread', mp5lib],
                            library_dirs = mkl_library_list,
                            include_dirs = mkl_include_list+[numpy.get_include()],
                            extra_compile_args = extra_compile_args2,
                            define_macros=macros) 
                    ]
    cmdclass = {}

setup(
    name='fastlmm',
    version=version,
    description='Fast GWAS',
    long_description=readme(),
    long_description_content_type = 'text/markdown',
    keywords='gwas bioinformatics LMMs MLMs linear mixed models',
    url="https://fastlmm.github.io/",
    author='FaST-LMM Team',
    author_email='fastlmm-dev@python.org',
    license='Apache 2.0',
    packages=[ #basically, everything with a __init__.py
        "fastlmm",
        "fastlmm/association",
        "fastlmm/association/altset_list",
        "fastlmm/association/tests",
        "fastlmm/external",
        "fastlmm/external/util",
        "fastlmm/feature_selection",
        "fastlmm/inference",
        "fastlmm/inference/tests",
        "fastlmm/pyplink", #old snpreader
        "fastlmm/pyplink/altset_list", #old snpreader
        "fastlmm/pyplink/snpreader", #old snpreader
        "fastlmm/pyplink/snpset", #old snpreader
        "fastlmm/util",
        "fastlmm/util/matrix",
        "fastlmm/util/standardizer",
        "fastlmm/util/stats",
        "fastlmm/util/stats/quadform",
        "fastlmm/util/stats/quadform/qfc_src"
    ],
    package_data={"fastlmm/association" : [
                       "Fastlmm_autoselect/FastLmmC.exe",
                       "Fastlmm_autoselect/libiomp5md.dll",
                       "Fastlmm_autoselect/fastlmmc",
                       "Fastlmm_autoselect/FastLmmC.Manual.pdf"],
                  "fastlmm/feature_selection" : [
                       "examples/bronze.txt",
                       "examples/ScanISP.Toydata.config.py",
                       "examples/ScanLMM.Toydata.config.py",
                       "examples/ScanOSP.Toydata.config.py",
                       "examples/toydata.5chrom.bed",
                       "examples/toydata.5chrom.bim",
                       "examples/toydata.5chrom.fam",
                       "examples/toydata.bed",
                       "examples/toydata.bim",
                       "examples/toydata.cov",
                       "examples/toydata.dat",
                       "examples/toydata.fam",
                       "examples/toydata.map",
                       "examples/toydata.phe",
                       "examples/toydata.shufflePlus.phe",
                       "examples/toydata.sim",
                       "examples/toydataTest.phe",
                       "examples/toydataTrain.phe"
                       ]
                 },
    install_requires = ['pandas>=0.19.0','matplotlib>=1.5.1',
                       'scikit-learn>=0.19.1', 'pysnptools>=0.4.10', 'dill>=0.2.9',
                       'statsmodels>=0.10.1', 'psutil>=5.6.7'],
    cmdclass = cmdclass,
    ext_modules = ext_modules,
  )
