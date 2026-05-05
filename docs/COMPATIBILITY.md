# Compatibility notes

## Python and compiler support

- Python: 3.11+
- C++: C++20 compiler
- CMake: 3.20+

## libhmm version

Current binding baseline is `libhmm` tag `v3.4.0`.

## SIMD and wheel portability

`libhmm` build defaults target the build machine CPU (`-march=native` on GCC/Clang or CPU-selected `/arch` on MSVC). This is ideal for local development and benchmarks, but binary wheels require additional controls to avoid producing architecture-specific artifacts that fail on older CPUs.

Current recommendation:

1. Validate source builds first on all target OSes.
2. Enable wheel publishing only after finalizing a portable SIMD policy for dependency builds.

## Local source preference

`pylibhmm` CMake uses `../libhmm` when available to support side-by-side development with the local C++ project.
