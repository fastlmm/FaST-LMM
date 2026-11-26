FaST-LMM
=================================

FaST-LMM, which stands for Factored Spectrally Transformed Linear Mixed Models, is a program for performing 
genome-wide association studies (GWAS) on datasets of all sizes, up to one million individuals.

This release contains the following features, each illustrated with an IPython notebook.

* Core FaST-LMM ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/FaST-LMM.ipynb)) -- [Lippert *et al.*, *Nature Methods* 2011](http://www.nature.com/nmeth/journal/v8/n10/abs/nmeth.1681.html)

Improvements:

* **New: GPU and float32 for `single_snp`** ([notebook](https://github.com/fastlmm/FaST-LMM/blob/main/doc/ipynb/FaST-LMM.ipynb), [GPU guide](https://github.com/fastlmm/FaST-LMM/blob/main/doc/source/gpu.rst)) -- single-kernel `single_snp` runs over 10 times as fast on an NVIDIA GPU with float32 as on the CPU with the default float64. See [GPU and float32](#gpu-and-float32) below.
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

FaST-LMM supports Python 3.11.5 through 3.15. HDF5 and BGEN file support are optional and available on Python 3.11 through 3.14: install `fastlmm[hdf5]`, `fastlmm[bgen]`, or `fastlmm[hdf5,bgen]`.

For best CPU performance, limit BLAS threads to your physical core count (for example, `export OMP_NUM_THREADS=16` on a 16-core, 32-thread CPU); in our tests, using every logical core was slower. For large problems, an NVIDIA GPU is much faster; see [GPU and float32](#gpu-and-float32).

GPU and float32
=================================

FaST-LMM's single-kernel `single_snp` can run on an NVIDIA GPU, in float32 (strongly recommended) or float64, thanks to [CuPy](https://cupy.dev/), the NumPy-compatible GPU array library.
On a synthetic dataset with 14,925 individuals and 100,000 SNPs, float32 on an RTX 3060
ran over 10 times as fast as the default CPU float64, and about 5 times as fast as
CPU float32. On the CPU alone, float32 ran about 2.5 times as fast as float64.
See the [benchmark](https://fastlmm.github.io/FaST-LMM/benchmark.html) for details and how to reproduce it.

Install the `cupy` extra:

    pip install "fastlmm[cupy]==1.1.0b1"

Then ask for the GPU and float32:

```python
from fastlmm.association import single_snp

results = single_snp(bed_fn, pheno_fn, covar=cov_fn, count_A1=False, xp="cupy", dtype="float32")
```

Float32 changes p-values slightly, so compare against float64 on representative data
when precision matters. Other FaST-LMM functions, including two-kernel `single_snp` and
epistasis, warn and run on the CPU in float64. GPU support requires NVIDIA CUDA on
x86-64 Windows or Linux, including WSL2; Python 3.15 is currently CPU-only.
A 12 GB card handles about 20,000 individuals with a full-rank kernel, or about 50,000
individuals with an 8,000-SNP low-rank kernel; the guide gives the general rule.
See the [GPU guide](https://github.com/fastlmm/FaST-LMM/blob/main/doc/source/gpu.rst)
for setup, defaults, caching, and limitations.

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
