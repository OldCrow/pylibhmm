# Changelog

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
