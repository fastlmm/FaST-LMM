FaST-LMM
=================================

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing 
genome-wide association studies (GWAS) on datasets of all sizes, up to one millions samples.

This release contains the following features, each illustrated with an IPython notebook.

* Core FaST-LMM ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)

Improvements:

* New features for single_snp (including effect size and multiple phenotype support) and epistasis (including reporting beta and using pre-computed eigenvalue decompositions) ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/fastlmm2021.ipynb))  -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)
* Ludicrous-Speed GWAS ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/SingleSnpScale.ipynb)) -- [Kadie and Heckerman, *bioRxiv* 2018](https://www.biorxiv.org/content/10.1101/154682v2)
* Heritability with Spatial Correction ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/heritability_si.ipynb)), [Heckerman *et al.*, *PNAS* 2016](http://www.pnas.org/content/113/27/7377.abstract)
* Two Kernels ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/FaST-LMM.ipynb)) -- [Widmer *et al.*, *Scientific Reports* 2014](http://www.nature.com/srep/2014/141112/srep06874/full/srep06874.html)
* Set Analysis ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Bioinformatics* 2014](http://bioinformatics.oxfordjournals.org/content/early/2014/09/07/bioinformatics.btu504)
* Epistasis ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Scientific Reports,* 2013](http://www.nature.com/srep/2013/130122/srep01099/full/srep01099.html)
* Prediction ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)

*A C++ version, which is generally less functional, is available. See http://fastlmm.github.io/.*

Quick install:
=================================

`pip install fastlmm`

FaST-LMM supports Python 3.11.5 through 3.14. Python 3.15 qualification is in progress.

HDF5 support is optional. To read or write HDF5 files, install:

    pip install fastlmm[hdf5]

Existing HDF5 users upgrading from an earlier release should install this extra.
Install `fastlmm[hdf5,bgen]` to enable both optional formats.

*If you need support for BGEN files, instead do:*

    pip install fastlmm[bgen]

For best performance, be sure your Python distribution includes a fast version of NumPy. We use Anaconda's [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

GPU and precision features
=================================

`single_snp` supports NumPy/CuPy backends and float32/float64 linear algebra.
Explicit `xp`/`dtype` override the `ARRAY_MODULE`/`ARRAY_DTYPE` environment defaults.
For the GPU workflow in this checkout, activate a fresh virtual environment
and run `uv pip install ".[cupy]"` from the repository root. PySnpTools
1.1.0b2 or later transfers CPU-decoded genotypes to CuPy
and follows CuPy's NVIDIA CUDA prerequisites. CPU remains the default.
See the [GPU guide](https://github.com/fastlmm/FaST-LMM/blob/main/doc/source/gpu.rst)
for platform limits and complete examples.

Documentation
=================================

* IPython Notebooks:
	* [Core, Epistasis, Set Analysis, Two Kernels](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/FaST-LMM.ipynb)
    * [Multiple-Phenotype GWAS and related features](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/fastlmm2021.ipynb)
	* [Heritability with Spatial Correction](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/heritability_si.ipynb)
	* [Ludicrous-Speed GWAS](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/SingleSnpScale.ipynb)
* [Main Documentation](http://fastlmm.github.io/FaST-LMM/)
* [Project Home and Full Annotated Bibliography](https://fastlmm.github.io/)


Code
=================================

* [PyPi](https://pypi.org/project/fastlmm/)
* [GitHub](https://github.com/fastlmm/FaST-LMM)

Development
=================================

The project uses [uv](https://docs.astral.sh/uv/) for reproducible development
environments, dependency locking, testing, and builds:

```console
uv sync --frozen --extra dev --extra hdf5 --extra bgen
cd tests
uv run --frozen --no-sync python test.py
cd ..
uv build --no-sources
```

Testing has two supported runners. `pytest` collects Python tests and module,
reStructuredText, and Markdown doctests under `fastlmm/`; it is useful for
focused tests and editor integration. The canonical `tests/test.py` runner also
executes the input-based expected-output cases in `tests/inputs` and assembles
the project's legacy unittest suites. The runners overlap but cover different
workflows, so pytest collection does not replace the canonical runner.
CI and release qualification also run the focused backend and dtype pytest
modules. Cases that need a usable CUDA device require a separate GPU run.

See [RELEASING.md](RELEASING.md) for lower-bound, artifact, notebook, and
release qualification.

Policy on AI-assisted development and contributions
=================================

AI tools may be used as productivity aids for drafting, exploration, and
refactoring. Every contributed code or documentation change must be reviewed,
edited, and validated by a human. AI does not replace design judgment,
testing, or human responsibility for correctness.

The repository's published instructions and constraints for AI tools are in
[AGENTS.md](AGENTS.md).

Contacts
=================================

* Email the developers at fastlmm-dev@python.org.
* [Join](mailto:fastlmm-user-join@python.org?subject=Subscribe) the user discussion and announcement list (or use [web sign up](https://mail.python.org/mailman3/lists/fastlmm-user.python.org)).
* [Open an issue](https://github.com/fastlmm/FaST-LMM/issues) on GitHub.
