#!!!1 Need to merge this with fastlmm's main one.
import platform
import os
import sys
import shutil
from setuptools import setup, Extension
from distutils.command.clean import clean as Clean
import numpy

# Version number
version = '0.0.1'

def readme():
    with open('README.md') as f:
        return f.read()

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

#use_cython=False

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
                    #or filename.find("wrap_plink_parser.cpp") != -1 # remove automatically generated source file
                    #or filename.find("wrap_matrix_subset.cpp") != -1 # remove automatically generated source file
                    or filename.endswith('.pyc')
                                ):
                    tmp_fn = os.path.join(dirpath, filename)
                    print "removing", tmp_fn
                    os.unlink(tmp_fn)

# set up macro
if platform.system() == "Darwin":
    macros = [("__APPLE__", "1")]
elif "win" in platform.system().lower():
    macros = [("_WIN32", "1")]
else:
    macros = [("_UNIX", "1")]


#see http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
if use_cython:
    ext_modules = [Extension(name="ludicrous.cample",
                            language="c++",
                            sources=["ludicrous/cample.pyx"],
                            libraries = ['mkl_intel_ilp64', 'mkl_core', 'mkl_intel_thread', 'libiomp5md'],
                            library_dirs = ['ludicrous/Externals/Intel/MKL/Lib/intel64'],
                            include_dirs = [numpy.get_include(),r"ludicrous/Externals/Intel/MKL/Inc"],
                            extra_compile_args = ['/DMKL_ILP64'],
                            define_macros=macros),
                    Extension(name="ludicrous.mmultfilex",
                            language="c++",
                            sources=["ludicrous/mmultfilex.pyx","ludicrous/mmultfile.cpp"],
                            libraries = ['mkl_intel_lp64', 'mkl_core', 'mkl_intel_thread', 'libiomp5md'],
                            library_dirs = ['ludicrous/Externals/Intel/MKL/Lib/intel64'],
                            include_dirs=[numpy.get_include(),"ludicrous/Externals/Intel/MKL/Inc"],
                            extra_compile_args = ['/EHsc','/openmp', '/DMKL_LP64'],
                            define_macros=macros)]
    cmdclass = {'build_ext': build_ext, 'clean': CleanCommand}
else:
    ext_modules = [Extension(name="ludicrous.cample",
                            language="c++",
                            sources=["ludicrous/cample.cpp"],
                            libraries = ['mkl_intel_ilp64', 'mkl_core', 'mkl_intel_thread', 'libiomp5md'],
                            library_dirs = ['Externals/Intel/MKL/Lib/intel64'],
                            include_dirs = [numpy.get_include(),"Externals/Intel/MKL/Inc"],
                            extra_compile_args = ['/DMKL_ILP64'],
                            define_macros=macros),
                    Extension(name="ludicrous.mmultfilex",
                            language="c++",
                            sources=["ludicrous/mmultfilex.cpp","ludicrous/mmultfile.cpp"],
                            libraries = ['mkl_intel_lp64', 'mkl_core', 'mkl_intel_thread', 'libiomp5md'],
                            library_dirs = ['Externals/Intel/MKL/Lib/intel64'],
                            include_dirs=[numpy.get_include(),"Externals/Intel/MKL/Inc"],
                            extra_compile_args = ['/EHsc','/openmp', '/DMKL_LP64'],
                            define_macros=macros)]

    cmdclass = {}



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
                    or filename.find("cample.cpp") != -1 # remove automatically generated source file
                    or filename.endswith('.pyc')
                                ):
                    tmp_fn = os.path.join(dirpath, filename)
                    print "removing", tmp_fn
                    os.unlink(tmp_fn)

#python setup.py sdist bdist_wininst upload
setup(
    name='cample',
    version=version,
    description='cample',
    long_description=readme(),
    keywords='gwas bioinformatics', #!!!1 update
    url="http://research.microsoft.com/en-us/um/redmond/projects/mscompbio/", #!!!best?
    author='MSR', #!!!update
    author_email='fastlmm@microsoft.com', #!!!update
    license='Apache 2.0', #!!!update
    packages=[
        "ludicrous",
    ],
    install_requires = ['scipy>=0.15.1', 'numpy>=1.9.2', 'pandas>=0.16.2'],

    # extensions
    cmdclass = cmdclass,
    ext_modules = ext_modules
  )

