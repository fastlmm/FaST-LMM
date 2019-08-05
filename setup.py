import platform
import os
import sys
import shutil
from setuptools import setup, Extension 
from distutils.command.clean import clean as Clean
import numpy

# Version number
version = '0.2.33' #!!!cmk update


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
                if (   filename.endswith('.so')
                    or filename.endswith('.pyd')
                    or (use_cython and filename.find("wrap_qfc.cpp") != -1) # remove automatically generated source file
                    or (use_cython and filename.find("cample.cpp") != -1) # remove automatically generated source file
                    or (use_cython and filename.find("mmultfilex.cpp") != -1) # remove automatically generated source file
                    or filename.endswith('.pyc')
                                ):
                    tmp_fn = os.path.join(dirpath, filename)
                    print "removing", tmp_fn
                    os.unlink(tmp_fn)

# set up macro
if platform.system() == "Darwin":
    macros = [("__APPLE__", "1")]
    intel_root = os.path.join(os.path.dirname(__file__),"external/intel/linux")
    mp5lib = 'iomp5'
    extra_compile_args1 = ['-DMKL_ILP64','-fpermissive']
    extra_compile_args2 = ['-fopenmp', '-DMKL_LP64','-fpermissive']
elif "win" in platform.system().lower():
    macros = [("_WIN32", "1")]
    intel_root = os.path.join(os.path.dirname(__file__),"external/intel/windows")
    mp5lib = 'libiomp5md'
    extra_compile_args0 = ['/EHsc']
    extra_compile_args1 = ['/DMKL_ILP64']
    extra_compile_args2 = ['/EHsc', '/openmp', '/DMKL_LP64']
else:
    macros = [("_UNIX", "1")]
    intel_root = os.path.join(os.path.dirname(__file__),"external/intel/linux")
    mp5lib = 'iomp5'
    extra_compile_args0 = []
    extra_compile_args1 = ['-DMKL_ILP64','-fpermissive']
    extra_compile_args2 = ['-fopenmp', '-DMKL_LP64','-fpermissive']

mkl_library_list = [intel_root+"/mkl/lib/intel64",intel_root+"/compiler/lib/intel64"]
mkl_include_list = [intel_root+"/mkl/include"]

#see http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
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
                            libraries = ['mkl_intel_ilp64', 'mkl_core', 'mkl_intel_thread', mp5lib],
                            library_dirs = mkl_library_list,
                            include_dirs = mkl_include_list+[numpy.get_include()],
                            extra_compile_args = extra_compile_args1,
                            define_macros=macros),
                    Extension(name="fastlmm.util.matrix.mmultfilex",
                            language="c++",
                            sources=["fastlmm/util/matrix/mmultfilex.pyx","fastlmm/util/matrix/mmultfile.cpp"],
                            libraries = ['mkl_intel_lp64', 'mkl_core', 'mkl_intel_thread', mp5lib],
                            library_dirs = mkl_library_list,
                            include_dirs = mkl_include_list+[numpy.get_include()],
                            extra_compile_args = extra_compile_args2,
                            define_macros=macros)
                     ]
    cmdclass = {'build_ext': build_ext, 'clean': CleanCommand}
else:
    ext_modules = [Extension(name="fastlmm.util.stats.quadform.qfc_src.wrap_qfc",
                             language="c++",
                             sources=["fastlmm/util/stats/quadform/qfc_src/wrap_qfc.cpp", "fastlmm/util/stats/quadform/qfc_src/QFC.cpp"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args = extra_compile_args0,
                             define_macros=macros),
                   Extension(name="fastlmm.util.matrix.cample",
                            language="c++",
                            sources=["fastlmm/util/matrix/cample.cpp"],
                            libraries = ['mkl_intel_ilp64', 'mkl_core', 'mkl_intel_thread', mp5lib],
                            library_dirs = mkl_library_list,
                            include_dirs = mkl_include_list+[numpy.get_include()],
                            extra_compile_args = extra_compile_args1,
                            define_macros=macros),
                    Extension(name="fastlmm.util.matrix.mmultfilex",
                            language="c++",
                            sources=["fastlmm/util/matrix/mmultfilex.cpp","fastlmm/util/matrix/mmultfile.cpp"],
                            libraries = ['mkl_intel_lp64', 'mkl_core', 'mkl_intel_thread', mp5lib],
                            library_dirs = mkl_library_list,
                            include_dirs = mkl_include_list+[numpy.get_include()],
                            extra_compile_args = extra_compile_args2,
                            define_macros=macros) 
                    ]
    cmdclass = {}

#python setup.py sdist bdist_wininst upload
setup(
    name='fastlmm',
    version=version,
    description='Fast GWAS',
    long_description=readme(),
    keywords='gwas bioinformatics LMMs MLMs',
    url="http://research.microsoft.com/en-us/um/redmond/projects/mscompbio/fastlmm/",
    author='MSR',
    author_email='fastlmm@microsoft.com',
    license='Apache 2.0',
    packages=[
        "fastlmm/association/tests", #!!!cmk update
        "fastlmm/association",
        "fastlmm/external/util",
        "fastlmm/external",
        "fastlmm/feature_selection",
        "fastlmm/inference",
        "fastlmm/pyplink/altset_list", #old snpreader
        "fastlmm/pyplink/snpreader", #old snpreader
        "fastlmm/pyplink/snpset", #old snpreader
        "fastlmm/pyplink", #old snpreader
        "fastlmm/util/runner",
        "fastlmm/util/stats/quadform",
        "fastlmm/util/stats/quadform/qfc_src",
        "fastlmm/util/standardizer",
        "fastlmm/util/stats",
        "fastlmm/util",
        "fastlmm",
    ],
    package_data={"fastlmm/association" : [ #!!!cmk update
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
                       "examples/toydata.iidmajor.hdf5",
                       "examples/toydata.map",
                       "examples/toydata.phe",
                       "examples/toydata.shufflePlus.phe",
                       "examples/toydata.sim",
                       "examples/toydata.snpmajor.hdf5",
                       "examples/toydataTest.phe",
                       "examples/toydataTrain.phe"
                       ]
                 },
    install_requires = ['scipy>=0.15.1', 'numpy>=1.11.3', 'pandas>=0.19.0','matplotlib>=1.5.1', 'scikit-learn>=0.19.1', 'pysnptools>=0.3.14', 'dill>=0.2.9'],
    cmdclass = cmdclass,
    ext_modules = ext_modules,
  )
