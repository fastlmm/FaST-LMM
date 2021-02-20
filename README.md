FaST-LMM
=================================

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing 
genome-wide association studies (GWAS) on datasets of all sizes, up to one millions samples.

This release contains the following features, each illustrated with an IPython notebook.

* Core FaST-LMM ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)

Improvements:

* Ludicrous-Speed GWAS ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/SingleSnpScale.ipynb)) -- [Kadie and Heckerman, *bioRxiv* 2018](https://www.biorxiv.org/content/10.1101/154682v2)
* Heritability with Spatial Correction ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/heritability_si.ipynb)), [Heckerman *et al.*, *PNAS* 2016](http://www.pnas.org/content/113/27/7377.abstract)
* Two Kernels ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Widmer *et al.*, *Scientific Reports* 2014](http://www.nature.com/srep/2014/141112/srep06874/full/srep06874.html)
* Set Analysis ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Bioinformatics* 2014](http://bioinformatics.oxfordjournals.org/content/early/2014/09/07/bioinformatics.btu504)
* Epistasis ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Scientific Reports,* 2013](http://www.nature.com/srep/2013/130122/srep01099/full/srep01099.html)
* Prediction ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)

*A C++ version, which is generally less functional, is available. See http://fastlmm.github.io/.*


Documentation
=================================

* IPython Notebooks:
	* [Core, Epistasis, Set Analysis, Two Kernels](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)
	* [Heritability with Spatial Correction](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/heritability_si.ipynb)
	* [Ludicrous-Speed GWAS](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/SingleSnpScale.ipynb)
* [API Documentation](http://fastlmm.github.io/FaST-LMM/)
* [Project Home and Full Annotated Bibliography](https://fastlmm.github.io/)


Code
=================================
* [PyPi](https://pypi.org/project/fastlmm/)
* [GitHub](https://github.com/fastlmm/FaST-LMM)

Contacts
=================================

* Email the developers at fastlmm-dev@python.org.
* [Join](mailto:fastlmm-user-join@python.org?subject=Subscribe) the user discussion and announcement list (or use [web sign up](https://mail.python.org/mailman3/lists/fastlmm-user.python.org)).
* [Open an issue](https://github.com/fastlmm/FaST-LMM/issues) on GitHub.

Quick install:
=================================

If you have Miniconda or Anaconda installed, installation is as easy as:

    conda install "mkl==2019.4" "scipy" "numpy"
    pip install --no-build-isolation fastlmm

(1) Installation of dependent packages
-------------------------------------------

You must have the "mkl" (and related) packages installed. It is not available via pip,
but the conda command above will install it.

We recommend using a Python distribution such as 
[Anaconda](https://www.anaconda.com/distribution/).
This distribution can be used on Linux, Windows, and Mac and is free.
It is the easiest way to get all the required package
dependencies, especially the those related to the
MKL library.


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


Don't forget to set your PYTHONPATH to point to the directory above the one named fastlmm in
the fastlmm source code. For e.g. if fastlmm is in the [somedir] directory, then
in the unix shell use:

    export PYTHONPATH=$PYTHONPATH:[somedir]

Or in the Windows DOS terminal,
one can use: 

    set PYTHONPATH=%PYTHONPATH%;[somedir]

(or use the Windows GUI for env variables).

Note for Windows: You must have Visual Studio installed.

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

