# Development guide

## Setup

```bash
pip install -e ".[test]"
```

## Build

```bash
pip install . -v
```

## Test

```bash
pytest tests -v --tb=short
```

## Repository layout

- `CMakeLists.txt` — native build entrypoint
- `src/pylibhmm/_core.cpp` — nanobind module
- `src/pylibhmm/_common.h` — NumPy/libhmm conversion helpers
- `src/pylibhmm/__init__.py` — Python wrappers/validation
- `tests/` — pytest suite
- `examples/` — sample usage

## Rules for API changes

When changing `_core.cpp` bindings:

1. Update `__init__.py` if wrapper signatures change
2. Update `__init__.pyi` and `_core.pyi`
3. Add or update tests for new API behavior
4. Keep docs in `README.md` and `docs/API.md` aligned
