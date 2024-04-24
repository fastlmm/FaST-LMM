################################
:mod:`fastlmm` Documentation
################################

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing both
single-SNP and SNP-set genome-wide association studies (GWAS) on extremely large data sets.

See `FaST-LMM's README.md <https://github.com/fastlmm/FaST-LMM/blob/master/README.md>`_ for installation instructions, documentation, code, and a bibliography.

:new:

:func:`.single_snp` now supports effect size, multiple phenotypes, and related features (`notebook demonstrating new features
<https://github.com/fastlmm/FaST-LMM/blob/master/doc/ipynb/fastlmm2021.ipynb>`_).

:Code:

* `PyPi <https://pypi.org/project/fastlmm/>`_
* `GitHub <https://github.com/fastlmm/FaST-LMM>`_

:Contacts:

* Email the developers at `fastlmm-dev@python.org <mailto:fastlmm-dev@python.org>`_.
* `Join <mailto:fastlmm-user-join@python.org?subject=Subscribe>`_ the user discussion and announcements
  list (or use `web sign up <https://mail.python.org/mailman3/lists/fastlmm-user.python.org>`_).
* `Open an issue <https://github.com/fastlmm/PySnpTools/issues>`_ on GitHub.
* `Project Home <https://fastlmm.github.io/>`_ (including bibliography).


**************************************************
:mod:`single_snp`
**************************************************

.. autofunction:: fastlmm.association.single_snp

**************************************************
:mod:`single_snp_scale`
**************************************************

.. autofunction:: fastlmm.association.single_snp_scale

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


***********************
:mod:`util` Module
***********************


:mod:`util.compute_auto_pcs`
++++++++++++++++++++++++++++++++++++++++++++++++++
.. autofunction:: fastlmm.util.compute_auto_pcs

:mod:`util.manhattan_plot`
++++++++++++++++++++++++++++++++++++++++++++++++++
.. autofunction:: fastlmm.util.manhattan_plot

 
.. only:: html 

***********************
Indices and Tables
***********************

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
