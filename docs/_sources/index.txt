################################
:mod:`fastlmm` API Documentation
################################

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing both
single-SNP and SNP-set genome-wide association studies (GWAS) on extremely large data sets.
This release contains the improvements described in Widmer *et al.*, *Scientific Reports* 2014, and tests for epistasis.

The FaST-LMM GitHub site:  
https://github.com/MicrosoftGenomics/FaST-LMM/

Our main documentation (including live examples) is also available as ipython notebook:
https://github.com/MicrosoftGenomics/FaST-LMM/blob/master/doc/ipynb/FaST-LMM.ipynb


**************************************************
:mod:`single_snp`
**************************************************

.. autofunction:: fastlmm.association.single_snp

**************************************************
:mod:`single_snp_all_plus_select`
**************************************************

.. autofunction:: fastlmm.association.single_snp_all_plus_select


**************************************************
:mod:`single_snp_linreg`
**************************************************

.. autofunction:: fastlmm.association.single_snp_linreg


**************************************************
:mod:`single_snp_select`
**************************************************

.. autofunction:: fastlmm.association.single_snp_select


**************************************************
:mod:`epistasis`
**************************************************
.. autofunction:: fastlmm.association.epistasis


**************************************************
:mod:`snp_set`
**************************************************
.. autofunction:: fastlmm.association.snp_set



**************************************************
:class:`FastLMM`
**************************************************
.. autoclass:: fastlmm.inference.FastLMM
    :members:
    :undoc-members:
	:show-inheritance:
	:special-members:

**************************************************
:class:`LinearRegression`
**************************************************
.. autoclass:: fastlmm.inference.LinearRegression
    :members:
    :undoc-members:
	:show-inheritance:
	:special-members:

**************************************************
:mod:`heritability_spatial_correction`
**************************************************
.. autofunction:: fastlmm.association.heritability_spatial_correction


**************************************************
:mod:`compute_auto_pcs`
**************************************************
.. autofunction:: fastlmm.util.compute_auto_pcs
 
.. only:: html 

***********************
Indices and Tables
***********************

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
