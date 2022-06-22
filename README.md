FaST-LMM
=================================

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing 
genome-wide association studies (GWAS) on datasets of all sizes, up to one millions samples.

This release contains the following features, each illustrated with an IPython notebook.

* Core FaST-LMM ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)

Improvements:

* New features for single_snp (including effect size and multiple phenotype support) and epistasis (including reporting beta and using pre-computed eigenvalue decompositions) ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/fastlmm2021.ipynb))  -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)
* Ludicrous-Speed GWAS ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/SingleSnpScale.ipynb)) -- [Kadie and Heckerman, *bioRxiv* 2018](https://www.biorxiv.org/content/10.1101/154682v2)
* Heritability with Spatial Correction ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/heritability_si.ipynb)), [Heckerman *et al.*, *PNAS* 2016](http://www.pnas.org/content/113/27/7377.abstract)
* Two Kernels ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Widmer *et al.*, *Scientific Reports* 2014](http://www.nature.com/srep/2014/141112/srep06874/full/srep06874.html)
* Set Analysis ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Bioinformatics* 2014](http://bioinformatics.oxfordjournals.org/content/early/2014/09/07/bioinformatics.btu504)
* Epistasis ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Scientific Reports,* 2013](http://www.nature.com/srep/2013/130122/srep01099/full/srep01099.html)
* Prediction ([notebook](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)

*A C++ version, which is generally less functional, is available. See http://fastlmm.github.io/.*

Quick install:
=================================

`pip install fastlmm`

For best performance, be sure your Python distribution includes a fast version of NumPy. We use Anaconda's [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Documentation
=================================

* IPython Notebooks:
	* [Core, Epistasis, Set Analysis, Two Kernels](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb)
    * [Multiple-Phenotype GWAS and related features](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/fastlmm2021.ipynb)
	* [Heritability with Spatial Correction](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/heritability_si.ipynb)
	* [Ludicrous-Speed GWAS](https://nbviewer.jupyter.org/github/fastlmm/FaST-LMM/blob/master/doc/ipynb/SingleSnpScale.ipynb)
* [Main Documentation](http://fastlmm.github.io/FaST-LMM/)
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
