.. Comment: Keep these badges in sync with the README.md file manually.

.. image:: https://badge.fury.io/py/fastlmm.svg
    :target: https://badge.fury.io/py/fastlmm
.. image:: https://github.com/fastlmm/FaST-LMM/actions/workflows/ci.yml/badge.svg?branch=main
    :target: https://github.com/fastlmm/FaST-LMM/actions/workflows/ci.yml
.. image:: https://img.shields.io/pypi/pyversions/fastlmm

################################
:mod:`fastlmm` Documentation
################################

.. py:module:: fastlmm

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing both
single-SNP and SNP-set genome-wide association studies (GWAS) on extremely large data sets.

See `FaST-LMM's README.md <https://github.com/fastlmm/FaST-LMM/blob/main/README.md>`_ for installation instructions, documentation, code, and a bibliography.


NumPy is the default backend. Single-kernel :func:`.single_snp` can also run on an
NVIDIA GPU, over 10 times as fast with float32. See the :doc:`GPU guide <gpu>`
and the :doc:`benchmark <benchmark>`.

HDF5 and BGEN support are optional and available on Python 3.11 through 3.14.
Install ``fastlmm[hdf5]``, ``fastlmm[bgen]``, or ``fastlmm[hdf5,bgen]`` as needed.

Effect size and multiple phenotypes are illustrated in the `multiple-phenotype notebook
<https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/fastlmm2021.ipynb>`_.

.. toctree::
   :hidden:

   gpu
   benchmark
   api_backend_precision

:Code:

* `PyPi <https://pypi.org/project/fastlmm/>`_
* `GitHub <https://github.com/fastlmm/FaST-LMM>`_

:Contacts:

* Email the developers at `fastlmm-dev@python.org <mailto:fastlmm-dev@python.org>`_.
* `Join <mailto:fastlmm-user-join@python.org?subject=Subscribe>`_ the user discussion and announcements
  list (or use `web sign up <https://mail.python.org/mailman3/lists/fastlmm-user.python.org>`_).
* `Open an issue <https://github.com/fastlmm/FaST-LMM/issues>`_ on GitHub.
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
