# Changelog

## v0.11.1 (2026-08-26)

Patch, on the 0.11.0 precedent: the pinned libhmm moves a patch release and
wheel users observe its fixes.

### Changed
- **libhmm pin moved to v4.4.1** (correctness patch, libhmm PR #102).
  Brings libhmm's ten defensive-review fixes into the wheels: out-span
  guard on `getBatchLogProbabilities` (was an OOB heap write, #86); bounded
  JSON number parse (was an OOB read on non-NUL-terminated views, #87);
  von Mises `fit()` κ = NaN for near-degenerate angular data (#84); SSE2
  `log_pd` subnormal prescale (#85); `sin_pd(−0)` sign of zero (#81);
  AVX-512/AVX2 CPUID feature-mask completion (#83); StudentT μ validation
  (#90); JSON `pi`/`trans` value validation (#91); count-distribution
  double→int cast bounds — x86/AArch64 parity (#88); legacy `States:`
  bound and exception contract (#89). No binding-surface change.

Minor, on the 0.10.0 precedent: the pinned libhmm moves a minor release and
wheel users observe its fixes.

### Changed
- **libhmm pin moved to v4.4.0.** Brings libhmm's v4.4.0 fixes into the
  wheels: clean-room SIMD cos/sin at every tier with per-tier ULP gates
  (libhmm #74), ISA dispatch extended to the transcendental kernels (#58),
  Bessel log-I0 seam and circular-variance fixes (#72/#73/#76),
  weighted-Gaussian NaN (#80), rejection of all-zero `pi`/`trans` (#78).
  No binding-surface change. libhmm v4.4.0's new API (topology constraints,
  `fit_best_of_n()`, `sample()`, `clone()`) is not exposed by this bump;
  that is follow-up feature work.

## v0.10.0 (2026-08-16)

Minor rather than patch: `requires-python` narrows to `>=3.12`, the wheel set
changes shape, and the pinned libhmm moves to a release with breaking build
changes of its own.

### Fixed
- **0.9.2 and 0.9.3 declared `requires-python >= 3.11` but shipped no cp311
  wheel.** 0.9.0/0.9.1 did; the `CIBW_SKIP` line gained `cp311-*` without
  `requires-python` moving in the same change. A 3.11 user therefore passed the
  metadata gate, found no wheel, and pip fell back to the sdist — compiling
  libhmm on their machine. `requires-python` is now `>=3.12`, matching what is
  actually built and tested.
- **`STABLE_ABI` had never done anything.** `nanobind_add_module` has carried
  the flag since this repo began, but nanobind gates it on
  `TARGET Python::SABIModule`, which only exists when `find_package(Python)` is
  given `${SKBUILD_SABI_COMPONENT}` — and `wheel.py-api` was never set, so the
  component was never requested. Both halves are now in place.

### Build
- **One abi3 wheel per platform instead of one per interpreter.** `cp312-abi3`
  covers 3.12 and every later version; free-threaded builds still get their own
  wheel, as the limited API does not apply there. 21 files per release becomes
  roughly 15.
- **Interpreter set is an allowlist (`CIBW_BUILD`), not a denylist.** The old
  `CIBW_SKIP` naming cp39/cp310/cp311 could not express the invariant it existed
  for — anything upstream adds ships untested until someone notices. It failed
  open twice in one run for pylibstats: `cp314-*` does not match the
  free-threaded `cp314t-` prefix, and nothing named cp315 because 3.15 did not
  exist when the line was written. `musllinux` stays a `CIBW_SKIP` entry, being
  an ABI axis orthogonal to the interpreter set.
- **cibuildwheel pinned to 4.2.0.** Unpinned, the set of interpreters it knows
  about grows on upstream's schedule rather than ours.
- **CI matrix gains free-threaded 3.14t**, so the allowlist's claim to cover
  only what CI tests is true rather than decorative.
- **libhmm pin moves `v4.2.5` → `v4.3.0`**, and the seven forced unprefixed
  option sets in the FetchContent branch are removed, as their own comment
  instructed. v4.3.0 retired those spellings outright and defaults its component
  toggles off `PROJECT_IS_TOP_LEVEL`, which is false under FetchContent.

---

## v0.9.3 (2026-07-19)

### Build
- Pinned libhmm FetchContent fallback to `v4.2.5`, a license-hygiene release
  that reimplements libhmm's incomplete gamma/beta and inverse-erf special
  functions from public-domain references (Abramowitz & Stegun, NIST DLMF,
  Lentz, Winitzki) instead of Numerical Recipes. No API or behavior change;
  numerical results are identical.

---

## v0.9.2 (2026-07-04)

### Changed
- **CI Python matrix updated to 3.12–3.14**: drops 3.11 (past SPEC 0 42-month
  window as of April 2026; security-only since October 2025) and adds 3.14
  (released October 2025). Wheel builds also drop cp311 via `CIBW_SKIP`.
  `requires-python = ">=3.11"` is retained for one more cycle.

### Added
- **ASan CI job** (Finding 2, pylibhmm half): builds the extension with
  `-fsanitize=address` and runs pytest under ASan via `LD_PRELOAD` on Linux.
  `detect_leaks=0` suppresses CPython false positives. Would have caught the
  calculator UAF (Finding 1) at the extension layer.

### Build
- **`LIBHMM_PORTABLE=ON` in `wheels.yml`** (Finding 9, pylibhmm half): sets
  `CIBW_CONFIG_SETTINGS: cmake.define.LIBHMM_PORTABLE=ON` so cibuildwheel
  passes the portable baseline ISA flag to libhmm's SIMD TUs. Tier-2
  runtime-dispatched kernels are unaffected.
- Pinned libhmm FetchContent fallback to `v4.2.4`, which adds `LIBHMM_PORTABLE`,
  the ASan CI job, E-step deduplication, and per-state observation copy
  elimination.

---

## v0.9.1 (2026-07-04)

### Fixed
- **Use-after-free in all four calculator bindings** (Finding 1): `ForwardBackwardCalculator`,
  `ViterbiCalculator`, `MVForwardBackwardCalculator`, and `MVViterbiCalculator` stored their
  observation sequence by reference to a temporary that died when the `__init__` lambda
  returned. Any subsequent `compute()` / `decode()` call re-read freed memory, producing
  silently-corrupted results rather than a crash. Extended the established Holder pattern
  (already used for all trainer classes) to all four calculator classes. No API changes.
  Regression tests in `tests/test_calculator_uaf_regression.py`.
- Removes the `ViterbiCalculator.decode()` workaround that returned the cached
  `getStateSequence()` instead of re-running Viterbi; `decode()` now safely re-runs.

### Build
- Pinned libhmm FetchContent fallback to `v4.2.3`, which adds compile-time guards
  (deleted rvalue overloads on derived calculator classes) that flag the old binding
  patterns as compile errors, and fixes denormal guards, `DiscreteDistribution` weighted
  fit under-normalization, and `decodePosterior()` silent failure.

---

## v0.9.0 (2026-07-04)

### Added
- **`MVViterbiCalculator`** — binds `BasicViterbiCalculator<ObservationVectorView>`. Accepts
  `HmmMV` and a 2-D NumPy array; exposes `decode()` (1-D int64 MAP path) and `log_probability`.
  Closes pylibhmm#7.
- **`MVMapBaumWelchTrainer`** — binds `BasicMapBaumWelchTrainer<ObservationVectorView>`. Adds
  Dirichlet priors on A and π for sparse MV data. Exposes `train()`, `last_log_probability`,
  `pseudo_count` (read/write), and `compute_log_prior()`. `pseudo_count=0` recovers standard
  `MVBaumWelchTrainer` exactly. Closes pylibhmm#8.

### Build
- Pinned libhmm FetchContent fallback to `v4.2.2` (adds `getLastLogProbability()` to
  `BasicMapBaumWelchTrainer`, enabling the `last_log_probability` property below).

---

## v0.8.0 (2026-07-04)

### Added
- **`MVSegmentalKMeansTrainer`** — multivariate segmental k-means trainer binding
  (`BasicSegmentalKMeansTrainer<ObservationVectorView>`). Accepts `HmmMV` and a list of
  2-D NumPy sequences; exposes `train()` and `is_terminated`. Recommended workflow:
  `kmeans_init` → `MVSegmentalKMeansTrainer` → `MVBaumWelchTrainer`.
- **`max_iterations` parameter** on `SegmentalKMeansTrainer` (default 100). Previously
  the scalar trainer had no iteration cap; the cap now matches the C++ API.

### Build
- Pinned libhmm FetchContent fallback to `v4.2.0`, which adds
  `BasicSegmentalKMeansTrainer<Obs>` and lifts the discrete-only restriction.

---

## v0.7.4 (2026-07-02)

### Build
- Pinned libhmm FetchContent fallback to `v4.1.4`, which resolves MSVC C4244 and
  C4267 warnings in the AVX-512 SIMD math helpers and example support headers.

## v0.7.3 (2026-07-02)

### Build
- Pinned libhmm FetchContent fallback to `v4.1.3`, which removes the remaining
  invalid `vreinterpretq_u64_f64(vceqq_f64(...))` wrappers in the StudentT and
  VonMises NEON batch kernels. Resolves linux-aarch64 wheel build failures.

## v0.7.2 (2026-07-02)

### Build
- Pinned libhmm FetchContent fallback to `v4.1.2`, which completes the
  linux-aarch64 NEON compile fixes in the upstream SIMD batch kernels (Beta,
  StudentT, and VonMises `vreinterpretq_u64_f64` wrappers).

## v0.7.1 (2026-07-02)

### Build
- Pinned libhmm FetchContent fallback to `v4.1.1`, which fixes the
  linux-aarch64 NEON compile failure caused by a spurious
  `vreinterpretq_u64_f64` wrapper around `vceqq_f64` in
  `detail/simd_math_helpers.h`.

## v0.7.0 (2026-07-02)

### Added
- **`BaumWelchTrainer.last_log_probability`** — exposes the total finite
  E-step log-probability computed during `train()`. Returns `-inf` before
  training and after all-invalid-sequence training. Binds
  `BasicBaumWelchTrainer::getLastLogProbability()` introduced in libhmm
  v4.1.0; available on both `BaumWelchTrainer` and `MVBaumWelchTrainer`.

### Build
- Pinned libhmm FetchContent fallback to `v4.1.0` (was `v4.0.4`), picking
  up the tier-2 SIMD distribution backend expansion and the runtime
  `DoubleVecOps` CPU-dispatch table.

## v0.6.3 (2026-06-18)

### Bug fixes
- **P-1 Thread safety** — `g_rng` (the module-level RNG used by no-arg `sample()`
  and `sample_mv()` calls) is now `thread_local`. Concurrent calls from multiple
  Python threads no longer share a single RNG state without locking, removing a
  data race under free-threaded (PEP 703) and multi-threaded workloads.
- **P-3 MV decode_posterior** — `MVForwardBackwardCalculator.decode_posterior()`
  was missing despite the scalar `ForwardBackwardCalculator` having it. Added,
  returning a 1-D `int64` NumPy array of shape `(T,)` matching the scalar
  binding's behaviour.
- **P-4 JSON deserializers return Python subclass** — `from_json()`, `load_json()`,
  `from_json_mv()`, and `load_json_mv()` previously returned the raw C++
  extension type (`_core.Hmm` / `_core.HmmMV`). They now return the Python
  wrapper (`pylibhmm.Hmm` / `pylibhmm.HmmMV`) so `isinstance` checks and
  validated setters (`set_pi`, `set_trans`) work as expected on deserialized
  models.
- **P-5 Exception-safe MV property buffers** — `DiagonalGaussian.means`,
  `DiagonalGaussian.variances`, `FullCovGaussian.mean`, and both `sample_mv()`
  methods allocated with bare `new double[]` before constructing the owning
  `nb::capsule`. All five lambdas now use `std::make_unique<double[]>` so the
  buffer is not leaked if an exception is raised between allocation and capsule
  construction.

### Infrastructure
- Pinned FetchContent fallback for libhmm to `v4.0.4` (was `v4.0.2`).
- Updated `AGENTS.md` platform notes with correct libhmm FetchContent version.

### Deferred (tracked)
- **P-2 BaumWelchTrainer convergence telemetry** — deferred; `BasicBaumWelchTrainer`
  does not expose per-iteration log-probability natively. Tracked in pylibhmm#4
  (binding gap) and libhmm#29 (native exposure). Targeted for v0.6.4 / libhmm
  v4.0.5.

## v0.6.2 (2026-05-12)

Initial public MV release changelog entry. See git history for earlier changes.
