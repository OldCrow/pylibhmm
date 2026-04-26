# WARP.md

This file provides guidance to Warp when working in this repository.

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

## Canonical commands

```powershell
pip install -e ".[test]"
pytest tests -v --tb=short
pip install . -v
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
