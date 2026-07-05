# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project purpose

`pylibhmm` provides Python bindings for `libhmm` using `nanobind` and `scikit-build-core`.

Primary goals:

- expose core `libhmm` modeling/training APIs to Python
- provide NumPy-friendly interfaces
- keep Python stubs and docs in sync with native bindings

## Key paths

- `CMakeLists.txt` — native build wiring and libhmm dependency strategy
- `src/pylibhmm/_core.cpp` — binding definitions
- `src/pylibhmm/_common.h` — NumPy conversion helpers
- `src/pylibhmm/__init__.py` — Python wrappers/validation
- `src/pylibhmm/__init__.pyi`, `src/pylibhmm/_core.pyi` — type stubs
- `tests/` — pytest coverage
- `docs/` — API and contributor docs

## Session Start Baseline Workflow (Required)

**Requires Python ≥ 3.11.** At the start of every session, do these steps in order:

1. Verify machine architecture (OS + CPU) and Python architecture.
2. Select the platform-specific build path for this host.
3. Build/install/test only after architecture is confirmed.

Architecture checks:

```bash
# macOS/Linux shells
uname -m
uname -s
python -c "import platform, struct; print(platform.system(), platform.machine(), struct.calcsize('P')*8)"
```

```powershell
# Windows PowerShell
[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
[System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
python -c "import platform, struct; print(platform.system(), platform.machine(), struct.calcsize('P')*8)"
```

## Platform-specific build requirements

### macOS (non-Catalina)

- `pylibhmm` prefers local `../libhmm` when present; otherwise it fetches `libhmm` `v4.2.4` via FetchContent.
- Ensure the active Python and `libhmm` build target the same architecture.

```bash
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -v --tb=short
```

### macOS Catalina (10.15)

- When `pylibhmm` builds local/fetched `libhmm`, apply the macOS Catalina notes from `../libhmm/AGENTS.md`:
  - avoid Homebrew LLVM/libc++ on Catalina unless explicit troubleshooting is required; use the system AppleClang toolchain.
- If you must override the guard for troubleshooting only, pass `-Ccmake.define.LIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX=ON`. This flag bypasses the guard that blocks Homebrew libc++ on Catalina due to ABI incompatibility with the 10.15 deployment target; use only when debugging.

### Linux

- Requires GCC ≥ 12 or Clang ≥ 14 for C++20 support.
- If `libhmm` is not found locally, CMake fetches it automatically at v4.2.4.

```bash
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -v --tb=short
```

### Windows (MSVC)

> **Windows tool paths vary** by installation method (direct installer, `winget`, `chocolatey`, Microsoft Store, etc.). See libhmm `AGENTS.md` Windows section for the full toolchain activation snippet and path alternatives.

- Visual Studio 2022 (Build Tools or full IDE) is required as the C++ compiler. Install from https://aka.ms/vs/17/release/vs_buildtools.exe, `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`.
- Use the VS 2022 x64 generator for reproducible MSVC builds (`-Ccmake.define.CMAKE_GENERATOR="Visual Studio 17 2022"`).
- `libhmm` SIMD selection and resulting binaries are architecture-dependent; keep the architecture check mandatory.

```powershell
python -m pip install -e ".[test]" `
  -Ccmake.define.CMAKE_GENERATOR="Visual Studio 17 2022" `
  -Ccmake.define.CMAKE_GENERATOR_PLATFORM=x64 `
  -Ccmake.build-type=Release
python -m pytest tests -v --tb=short
```

## Canonical commands

```bash
# macOS/Linux
python -m pip install -e ".[test]"   # installs package (editable) + test dependencies
python -m pytest tests -v --tb=short
```

```powershell
# Windows
python -m pip install -e ".[test]"
python -m pytest tests -v --tb=short
```

## Editing rules

1. Keep wrapper signatures (`__init__.py`) aligned with `_core.cpp`.
2. Update stubs whenever API signatures change.
3. Add tests for each new bound class/method.
4. Keep documentation concise and accurate.

## Release checklist

1. `pytest` passes on local platform.
2. CI matrix passes on Linux/macOS/Windows.
3. API docs and stubs updated.
4. Wheel portability considerations reviewed against `docs/COMPATIBILITY.md`.
