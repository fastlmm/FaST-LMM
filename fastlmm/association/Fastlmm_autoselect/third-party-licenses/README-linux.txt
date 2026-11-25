fastlmmc is a static executable. Besides FaST-LMM itself (Apache License 2.0, LICENSE.TXT),
it contains:

- Intel oneMKL 2026.2.1 (Intel Simplified Software License; Intel-oneMKL-LICENSE.txt and
  Intel-oneMKL-third-party-programs.txt)
- Intel OpenMP runtime 2026.2.0 (Intel End User License Agreement for Developer Tools;
  Intel-OpenMP-LICENSE.txt)
- The GNU C library, libstdc++ and libgcc from Ubuntu 22.04 (GCC 11.4). glibc is under the
  GNU LGPL 2.1; its source is available from https://www.gnu.org/software/libc/ and the
  Ubuntu archive. The complete FaST-LMM source needed to relink is at
  https://github.com/fastlmm/fastlmmc (tag v2.08).
- zlib (zlib license), from Ubuntu 22.04.
