# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project Overview

`pylibhmm` provides Python bindings for `libhmm` using `nanobind` and `scikit-build-core`.

Primary goals:

- expose core `libhmm` modeling/training APIs to Python
- provide NumPy-friendly interfaces
- keep Python stubs and docs in sync with native bindings

Key paths:

- `CMakeLists.txt` — native build wiring and libhmm dependency strategy
- `src/pylibhmm/_core.cpp` — binding definitions
- `src/pylibhmm/_common.h` — NumPy conversion helpers
- `src/pylibhmm/__init__.py` — Python wrappers/validation
- `src/pylibhmm/__init__.pyi`, `src/pylibhmm/_core.pyi` — type stubs
- `tests/` — pytest coverage
- `docs/` — API and contributor docs
- `docs/PARITY.md` — libhmm ↔ pylibhmm parity ledger (verified behavior,
  intentional divergences, open items); check it before and update it
  after any binding-parity review

## Architecture

`_core` is a single nanobind extension module built via scikit-build-core.
`CMakeLists.txt` prefers a local `../libhmm` checkout over the pinned
`FetchContent` tag when present (dev-loop speed); the pin itself lives
only in `CMakeLists.txt`'s `GIT_TAG`, never restated in docs — see PLAN.md
Cross-Repo Dependencies. `_common.h` conversions are copy-based by
default; the one exception, `obs_matrix_views()`, builds non-owning views
directly into the caller's NumPy buffer for the MV distributions'
`fit()`/`fit_weighted()` bindings, safe only because the views are
consumed synchronously within that same call. **Do not extend this
zero-copy pattern to any binding that would retain the views beyond the
call's lifetime** — that's a silent use-after-free, not a test failure.

Full build-model, dependency-resolution, `__init__.py`-validation, and
type-stub detail: `docs/DEVELOPMENT.md`.

## Session Start

**Requires Python ≥ 3.11.** Follow the standard architecture-check ritual:
[SESSION-START.md](https://github.com/OldCrow/standards/blob/main/SESSION-START.md).

## Build Commands

Run from the repository root:

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

### CMake standard

Full rules: [CMake House Style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md)
in the fleet standards repo; this section is self-sufficient for this repo. pylibhmm is built via
scikit-build-core (the `pip install -e` path above is primary and
authoritative); `CMakePresets.json` (schema 6, min CMake 3.25) exists only
for direct-CMake dev/debugging (e.g. exercising the extension module build
outside pip): `release` → `build/`, `debug` → `build-debug/`. No project
extras, no `generator` field. Deviation: prefers a local `../libhmm`
sibling checkout over the pinned FetchContent tag when present —
dev-loop speed; the FetchContent pin is what's exercised on machines
without the sibling (e.g. CI).

## Platform-Specific Notes

### macOS (non-Catalina)

- `pylibhmm` prefers local `../libhmm` when present; otherwise it fetches `libhmm` at the pinned `GIT_TAG` (see `CMakeLists.txt`) via FetchContent.
- Ensure the active Python and `libhmm` build target the same architecture.

```bash
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -v --tb=short
```

### macOS Catalina (10.15)

- When `pylibhmm` builds local/fetched `libhmm`, avoid Homebrew LLVM/libc++ on Catalina unless explicit troubleshooting is required; use the system AppleClang toolchain. Homebrew sets `CC`/`CXX`/`LDFLAGS` to Homebrew LLVM's libc++, which is ABI-incompatible with the 10.15 deployment target.
- If you must override the guard for troubleshooting only, pass `-Ccmake.define.LIBHMM_ALLOW_UNSUPPORTED_CATALINA_HOMEBREW_LIBCXX=ON`. This flag bypasses the guard that blocks Homebrew libc++ on Catalina; use only when debugging.

### Linux

- Requires GCC ≥ 12 or Clang ≥ 14 for C++20 support.
- If `libhmm` is not found locally, CMake fetches it automatically at the pinned `GIT_TAG` (see `CMakeLists.txt`).

```bash
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -v --tb=short
```

### Windows (MSVC)

- **Minimum toolchain**: Visual Studio 2022 (17.x) or later with the C++
  desktop workload — any MSVC toolset with full C++20 support. Verified
  through VS 2026 (v18, MSVC 14.5x). Build Tools or a full IDE edition both
  work.
- Don't pin a generator locally — see
  [WINDOWS-TOOLCHAIN.md §3](https://github.com/OldCrow/standards/blob/main/WINDOWS-TOOLCHAIN.md)
  and [CMAKE-HOUSE-STYLE.md](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md).
- `libhmm` SIMD selection and resulting binaries are architecture-dependent; keep the architecture check mandatory.

```powershell
python -m pip install -e ".[test]" -Ccmake.build-type=Release
python -m pytest tests -v --tb=short
```

#### Windows toolchain setup

pylibhmm needs no per-session `vcvars` activation — the Visual Studio
CMake generator locates its own toolchain (VS-generator case only;
non-VS generators and direct `cl.exe` use still need it). See
[WINDOWS-TOOLCHAIN.md](https://github.com/OldCrow/standards/blob/main/WINDOWS-TOOLCHAIN.md)
for one-time setup, the Smart App Control note, and the CMake version
requirement.

## Coding Conventions

1. Keep wrapper signatures (`__init__.py`) aligned with `_core.cpp`.
2. Update stubs whenever API signatures change (hand-edit — see Architecture).
3. Add tests for each new bound class/method.
4. Keep documentation concise and accurate.

### Linting

**Python** (`src/pylibhmm/__init__.py`, `tests/`, `examples/`): ruff, config
in `pyproject.toml`. Rules: `B`/`E`/`F`/`I`/`UP` — `B` (flake8-bugbear) was
adopted at 0.12.0 after confirming every previously-blind
`pytest.raises(Exception)` site actually raises `ValueError` (nanobind's
translation of `std::invalid_argument`); tests now assert that exact type.
`.pyi` stubs are exempt from the line-length rule (compact single-line
signatures are idiomatic there).
```bash
ruff check src/pylibhmm tests examples
ruff format src/pylibhmm tests examples   # applied repo-wide at v0.11.1+ (issue #13)
```
pyright runs via the editor/agent language server only, not CI (`[tool.pyright]`
points it at `.venv` so `numpy` and the editable install resolve); baseline
is 0 errors. mypy is not adopted — `__init__.py` is only partially annotated,
so enabling it needs a real annotation pass first (see PLAN.md Known Gaps).

**C++ binding layer** (`_core.cpp`, `_common.h`): its own cppcheck
invocation, `scripts/lint-cpp.sh`, not a copy of libhmm's — it needs
`--language=c++` explicit (cppcheck misparses `_common.h` as C otherwise)
and suppresses findings from libhmm's own headers by path (libhmm's
concern, not this repo's). Suppression list otherwise matches libhmm's.
```bash
bash scripts/lint-cpp.sh
```

## CI / Validation

Fleet-wide workflow rules (runner budget, bounded parallelism, ISA hazards on
hosted runners, action pinning, wheel builds):
[CI House Style](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md).

Wheel builds follow the fleet wheel contract
([CI House Style §9](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md#9-wheel-builds-pylibhmm-pylibstats)),
settled here at v0.10.0: `CIBW_BUILD` is an allowlist *defined* as the
interpreters `ci.yml` tests; `requires-python` moves in the same change as
the built set (0.9.2/0.9.3 declared `>= 3.11` while shipping no cp311
wheel — the incident behind the rule); the cp312 wheel is Stable ABI, and
`wheel.py-api` in `pyproject.toml` plus `SKBUILD_SABI_COMPONENT` in CMake
are one mechanism in two files — set both, or the result is an
abi3-tagged, version-locked wheel that cibuildwheel is structurally
unable to catch; cibuildwheel is pinned. `musllinux` stays a `CIBW_SKIP`
entry (an ABI axis orthogonal to the interpreter set, applied after
BUILD).

Release checklist:

1. `pytest` passes on local platform.
2. CI matrix passes on Linux/macOS/Windows.
3. API docs and stubs updated.
4. Wheel portability considerations reviewed against `docs/COMPATIBILITY.md`.
5. `ruff check`, `ruff format --check`, and `scripts/lint-cpp.sh` pass (all enforced by ci.yml's `lint` job; ruff is version-pinned there — bump it deliberately).

## Reading map — load on demand, not preemptively
- Full architecture detail (build model, dependency resolution,
  `_common.h`/`__init__.py` design, type-stub policy) → `docs/DEVELOPMENT.md`.
- Public class/method surface → `docs/API.md`.
- Wheel/platform compatibility considerations → `docs/COMPATIBILITY.md`.
- libhmm ↔ pylibhmm parity work (verified behavior, intentional
  divergences, open items) → `docs/PARITY.md`; check before and update
  after any binding-parity review.
- Session bootstrap ritual → [SESSION-START.md](https://github.com/OldCrow/standards/blob/main/SESSION-START.md).
- CMake conventions in depth → [CMAKE-HOUSE-STYLE.md](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md).
- Windows toolchain setup in depth → [WINDOWS-TOOLCHAIN.md](https://github.com/OldCrow/standards/blob/main/WINDOWS-TOOLCHAIN.md).
- CI/workflow rules fleet-wide → [CI-HOUSE-STYLE.md](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md).
- Session state, decisions, open items → `PLAN.md`.

## Open Items
See PLAN.md for current status, in-progress work, and open questions.
