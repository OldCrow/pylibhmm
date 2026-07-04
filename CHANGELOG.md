# Changelog

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
