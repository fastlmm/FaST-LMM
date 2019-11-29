FaST-LMM
=================================

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing both single-SNP and SNP-set
genome-wide association studies (GWAS) on extremely large data sets.  This release contains the improvements described in Widmer *et al.*, *Scientific Reports* 2014,
and tests for epistasis.

Our documentation (including live examples) is also available as IPython notebooks:

* Main Functions: https://github.com/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb
* Heritability with Spatial Correction (described in Heckerman *et al.*, *PNAS* 2016): https://github.com/fastlmm/FaST-LMM/blob/master/doc/ipynb/heritability_si.ipynb

Additionally, API documentation is available at http://fastlmm.github.io/FaST-LMM/.

A C++ version, which is generally less functional, is available. See http://fastlmm.github.io/.

Also http://fastlmm.github.io/ for the project's home and bibliography.

Quick install:
=================================

If you have pip installed, installation is as easy as:

    pip install fastlmm

Detailed Package Install Instructions:
==================================================================

fastlmm has the following dependencies:

python 2.7

Packages:

* scipy
* numpy
* pandas
* matplotlib
* scikit.learn (sklearn)
* dill
* pysnptools
* cython
* optional: [statsmodels -- install only required for logistic-based tests, not the standard linear LRT]


(1) Installation of dependent packages
-------------------------------------------

We highly recommend using a python distribution such as 
Anaconda (https://store.continuum.io/cshop/anaconda/) 
or Enthought (https://www.enthought.com/products/epd/free/).
Both these distributions can be used on Linux and Windows, are free 
for non-commercial use, and optionally include an MKL-compiled distribution
for optimal speed. This is the easiest way to get all the required package
dependencies.


(2) Installing from source
-------------------------------------------

Go to the directory where you copied the source code for fastlmm.

On Linux:

At the shell, type: 

    sudo python setup.py install


On Windows:

At the OS command prompt, type 

    python setup.py install



For developers (and also to run regression tests)
=====================================================

When working on the developer version, first add the src directory of the package to your PYTHONPATH 
environment variable.

For building C-extensions, first make sure all of the above dependencies are installed (including cython)

To build extension (from .\src dir), type the following at the OS prompt:

    python setup.py build_ext --inplace


Note, if this fails with a gcc permission denied error, then specifying the correct compiler will
likely fix the problem, e.g.

    python setup.py build_ext --inplace --compiler=msvc


Don't forget to set your PYTHONPATH to point to the directory above the one named fastlmm in
the fastlmm source code. For e.g. if fastlmm is in the [somedir] directory, then
in the unix shell use:

    export PYTHONPATH=$PYTHONPATH:[somedir]

Or in the Windows DOS terminal,
one can use: 

    set PYTHONPATH=%PYTHONPATH%;[somedir]

(or use the Windows GUI for env variables).

Note for Windows: You must have Visual Studio installed. If you have VisualStudio2008 installed 
(which was used to build python2.7) you need to nothing more. Otherwise, follow these instructions:

If you have Visual Studio 2010 installed, execute:

    SET VS90COMNTOOLS=%VS100COMNTOOLS%

or with Visual Studio 2012 installed:

    SET VS90COMNTOOLS=%VS110COMNTOOLS%

or with Visual Studio 2013 installed:

    SET VS90COMNTOOLS=%VS120COMNTOOLS%

or with Visual Studio 2015 installed:

    SET VS90COMNTOOLS=%VS130COMNTOOLS%

or with Visual Studio 2017 installed:

    SET VS90COMNTOOLS=%VS140COMNTOOLS%


Running regression tests
--------------------------------------

From the directory tests at the top level, run:

    python test.py

This will run a
series of regression tests, reporting "." for each one that passes, "F" for each
one that does not match up, and "E" for any which produce a run-time error. After
they have all run, you should see the string "............" indicating that they 
all passed, or if they did not, something such as "....F...E......", after which
you can see the specific errors.

Note that you must use "python setup.py build_ext --inplace" to run the 
regression tests, and not "python setup.py install".

