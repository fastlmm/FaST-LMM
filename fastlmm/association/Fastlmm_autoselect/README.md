# FaST-LMM C++ (FastLmmC)

## Preface to version 2.08 (2026)

Version 2.08 is a small update of the 2014 FaST-LMM C++ release, version 2.07. It changes
nothing about what the program computes. Its purpose is to make the code compile and run with
current tools: g++ 11 or later on Linux, Visual Studio 2022 on Windows, and Intel oneMKL
2026.2.1. The original 2.07 README follows below, unchanged. Where its build instructions
differ from this preface, follow this preface.

**Why an update was needed.** The 2.07 binaries were built with MKL 11.0 (2013). On current
CPUs (tested on an AMD Ryzen 9 7950X), the 2.07 Linux binary crashes with "Illegal instruction"
in the eigendecomposition. If you must run the old binary, set `MKL_CBWR=COMPATIBLE` first.

**What changed in 2.08:**

- Relicensed under the Apache License, Version 2.0 (previously the Microsoft Research License
  Agreement, non-commercial only). See [License](#license).
- Builds against oneMKL 2026.2.1 instead of MKL 11.0.
- Fixes for current compilers: g++ 11 (`nullptr`, `abs`) and Visual Studio 2022 (removed C
  runtime functions, a non-MFC resource header).
- Compiles cleanly with `-Wall`. Fixing those warnings also fixed AutoSelect's memory-partitioning
  calculation, which never took effect (results were not affected), and added a missing
  individual-count check when reading `.big` files.
- Creates the `-eigenout` folder directly instead of running a shell `mkdir` command, which
  failed on Windows for paths with forward slashes and everywhere for paths with spaces.
- The Linux build uses a `Makefile` (replacing `DoMkl_linux.sh`). On Windows, `Cpp/FastLmmC.sln`
  now has two configurations, `MKL-Release|x64` and `MKL-Debug|x64`. The 32-bit configurations
  (current oneMKL has no 32-bit libraries) and the AMD ACML ones (ACML was discontinued) were
  removed, along with `DoAcml_linux.sh`. The `USE_ACML` code paths remain in the source.
- A regression suite, `tests/regression`, compares outputs with those of the 2.07 binary. Across
  its 42 commands, log-likelihoods agree to about 1e-14 and p-values to about 1e-6 (relative),
  the same size as the differences between the old MKL's own code paths.

### Building on Linux

Building on Linux is easier than it used to be: oneMKL installs from PyPI without root, and
one `make` command builds a self-contained static binary.

```bash
# 1. Get oneMKL (once). Needs uv (or use pip with --target).
uv pip install --target ~/.cache/fastlmmcpp/mkl-2026.2.1 mkl-devel==2026.2.1 mkl-static==2026.2.1

# 2. Build Cpp/fastlmmc.
make -C Cpp -j MKLROOT=~/.cache/fastlmmcpp/mkl-2026.2.1

# 3. Optional: run the regression suite (a few minutes).
tests/regression/run.sh Cpp/fastlmmc
```

With a standard Intel oneAPI install, `MKLROOT` can be left out (it defaults to
`/opt/intel/oneapi/mkl/latest`).

### Building on Windows

Windows takes about as much setup as before. You need:

1. **Visual Studio 2022**, or its free Build Tools, with the "Desktop development with C++"
   workload.
2. **oneMKL.** Install Intel oneAPI MKL, or get it from PyPI with
   `pip install --target C:\mkl mkl-devel==2026.2.1 mkl-static==2026.2.1`; in that case the MKL
   folder is `C:\mkl\Library`.
3. **zlib**, built as a static library with the static C runtime. From the zlib source folder,
   in a "x64 Native Tools Command Prompt for VS 2022":
   `nmake -f win32\Makefile.msc CFLAGS="-nologo -MT -W3 -O2 -Oy- -Z7" zlib.lib`.
   Put `zlib.h` and `zconf.h` in an `include` folder, and `zlib.lib`, renamed to
   `zlibstat.lib`, in a `lib` folder. For the Debug configuration, also run
   `nmake -f win32\Makefile.msc clean`, rebuild with `-MTd -Od` in place of `-MT -O2 -Oy-`, and
   put that `zlibstat.lib` in `lib\Debug`.

Then, from the same command prompt:

```bat
set MKLROOT=C:\mkl\Library
set ZLIB_ROOT=C:\path\to\zlib
msbuild Cpp\FastLmmC.sln -p:Configuration=MKL-Release -p:Platform=x64
```

The program is `Cpp\x64\MKL-Release\FastLmmC.exe`; keep `libiomp5md.dll` (copied there by the
build) beside it. Use `-p:Configuration=MKL-Debug` for a debug build in `Cpp\x64\MKL-Debug`.

### Newer software

For new work, consider the Python version of FaST-LMM (<https://github.com/fastlmm/FaST-LMM>),
which is maintained and has more features.

### License

Starting with 2.08, FaST-LMM C++ is licensed under the Apache License, Version 2.0
(`LICENSE.TXT`), like the Python version of FaST-LMM. Versions through 2.07 were released under
the Microsoft Research License Agreement, which permitted only non-commercial use; the 2.07
README below and the 2.07 user manual (`Doc/user-manual.pdf`) still describe those old terms,
which no longer apply. Third-party components are listed in `NOTICE.TXT`.

## Original 2.07 README

```text
FaST-LMM, (Factored Spectrally Transformed Linear Mixed Models) is 
a program for performing genome-wide association studies (GWAS) on 
large data sets. It runs on both Windows and Linux systems, and has 
been tested on data sets with over 120,000 individuals.

This file is located in the root directory for the FaSTLMM source
tree.  To get started quickly with a precompiled version, see the 
user's manual in the Doc directory.

All the sources for the C++ versions are included in this tree.
However, there are additional external dependencies that you will 
need to download and install to build the code.

This document describes the directories within the tree and their 
usage/function.  The directories and their functions are:

  Cpp -
      The C++ project is contained within the Cpp tree.  All the 
      files for editing and starting a build with Visual Studio 
      are present.

  Data -
      The Data directory holds files for running simple tests
      to confirm proper installation of FaSTLMM.  The data may
      also be used as a sanity test after re-building from 
      sources or downloading copy of the program

  Doc - 
      All documentation files for the project are located under the 
      Doc directory

  Externals -
      All external dependencies used under their own license
      are located under the Externals directory
	  
  Test
      Unit Tests that drive the different FaSTLMM executables 
      and validate their behavior are located under this directory.
      
	  
---------- Building the code ----------
In addition to the code in this tree, the following external dependencies
that must be installed to build FaSTLMM.

Building for Windows.
  The C++ versions require Visual Studio 2010 (VS)- 
      Any version of Visual Studio (Express through Universal) is 
      capable of building FaSTLMM.  If you do not already have a copy
      of Visual Studio, the Visual Studio 2010 Express edition can be 
      freely downloaded from 
        http://www.microsoft.com/express/downloads/

  For the C++ version 
    FastLmmC uses a 3rd party math library for advanced math funtions
    and performance.  FastLmmC can use either Intel's MKL or AMD's 
    ACML math libraries.  Once you have installed the appropriate 
    library, use the Visual Studio IDE to open the solution file
    Cpp\FastLmmC.sln and select the configuration matching your
    library from the solution and build.  ACML requires an 
    additional step to tell Visual Studio where it is located.  
    You must set the environment variable ACML_ROOT to point to 
    your install location or libraries - for example
	  C>set ACML_ROOT=C:\AMD\acml4.4.0
	You can find more about the math libraries at their respective
	web sites:
    http://software.intel.com/en-us/articles/intel-mkl/
    http://developer.amd.com/libraries/acml/pages/default.aspx

  Building the C++ version for Linux
    FastLmmC uses a 3rd party math library for advanced math functions 
    and performance.  The program has been run on Ubuntu Linux and can 
    use either Intel's MKL or AMD's ACML math libraries for Linux.  Once 
    you have selected and installed the appropriate library, you can 
    then build using the appropriate script file located in the Cpp 
    directory.  Review of the two files, DoMKL_linux and DoAcml_linux, 
    will show very simple scripts to compile the program using g++ and 
    then link the .o files with the appropriate math library.  The *.o 
    files are written to version specific directories, so it is necessary 
    to create the appropriate directory prior to running the script.  
    For more details, see the script.

----------
KNOWN ISSUES
----------
```
