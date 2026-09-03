# Parity Ledger: libhmm ↔ pylibhmm

_Last updated: 2026-09-02 (created; first entries cover the v4.4.0
model-level bindings shipped in pylibhmm 0.12.0, verified against
libhmm v4.4.1 headers and empirically via the 0.12.0 test suite)._

Scope note: surfaces bound before 0.12.0 (distributions, calculators,
trainers, I/O) have working tests but have not been audited in ledger
terms; add entries as they are reviewed rather than back-filling claims.

## Verified Parity

| API surface | Checked | Notes |
|---|---|---|
| `clone_hmm` / `clone_hmm_mv` | 2026-09-02 | Deep-copy semantics match core `clone()`: independent pi/trans, cloned emissions, MV null slots stay null (tested). Wrapper rebuilds the Python-layer type via the `from_json` pattern; the per-slot `except RuntimeError` in `clone_hmm_mv` is sound because, within a valid index, `getDistribution`'s only `runtime_error` is the null slot (basic_hmm.h:210). |
| `sample` / `sample_mv` — error path | 2026-09-02 | Core `runtime_error` on a zero pi/transition row → Python `RuntimeError` with the core message intact (tested: "probability row sums to zero"). |
| `sample` / `sample_mv` — edge cases | 2026-09-02 | T=0 returns empty arrays both paths (tested). Scalar shape `(0,)`; MV shape `(0, 0)` — the D dimension is lost because core returns a default-constructed `ObservationMatrix()`; binding mirrors core faithfully. Seeded draws reproducible (tested); dtypes float64/int64, no narrowing. |
| `fit_best_of_n` / `fit_best_of_n_mv` | 2026-09-02 | `invalid_argument` (empty obsLists, n_restarts=0) → `ValueError` (tested). Restart-0 ≥ single-run guarantee holds through the binding (tested). Mutates in place across the boundary via C++ move-assign; Python object identity and wrapper class preserved (tested). Seed determinism per platform (tested). |
| `HmmTopology` + `initialize_topology[_mv]` / `enforce_topology[_mv]` | 2026-09-02 | `invalid_argument` (max_skip < 1 where used) → `ValueError` (tested). Mask/renormalise/degenerate-row-uniform-reset semantics match topology.h exactly (tested against hand-computed matrices). Ergodic enforce is a no-op, matching the core early-return (tested). |
| `validateInitialized()` surfacing (libhmm #78) | 2026-09-02 | Core `runtime_error` message ("pi … all zero … call setPi()") reaches Python intact through calculators and trainers (tested). Fires at scoring/training entry, not at trainer construction — binding matches core timing. |
| Docstring accuracy for the above | 2026-09-02 | Wrapper docstrings written against v4.4.1 headers this session: restart-0 guarantee, "only the transition matrix is managed — pi is the caller's", max_iters=500 default, exception types. Each claim matches a tested behavior. |

## Intentional Divergences

| API surface | Core behavior | Binding behavior | Justification |
|---|---|---|---|
| Function naming, scalar vs MV | One overloaded name (`clone_hmm`, `sample`, `fit_best_of_n`, topology fns take either type) | Separate `_mv`-suffixed names | Repo-wide binding convention (`count_free_parameters_mv`, `to_json_mv` precedent); keeps stubs and docs unambiguous. |
| RNG parameter | Caller passes a `std::mt19937_64&` engine | Integer `seed` (constructs a fresh engine per call); seedless `sample` uses a thread-local RNG | Established binding precedent (distribution `sample(seed)`, `kmeans_init(seed=42)`); no engine object is exposed anywhere in the binding surface. Consequence: consecutive unseeded/seed-reusing calls cannot share one engine stream the way C++ callers can. |
| `fit_best_of_n` seed default | No default — engine is a required parameter | `seed=42` default | Matches the `kmeans_init(seed=42)` precedent; deterministic-by-default for training-adjacent randomness. |
| `HmmTopology` member casing | `Ergodic`, `LeftToRight`, … (C++ enum-class) | Same PascalCase names (not Python UPPER_CASE) | Traceability to upstream docs/issues outweighs PEP 8 enum casing; nanobind enum, not `enum.Enum`. |
| Negative `T` / `n_restarts` | Unrepresentable (`std::size_t` parameters) | Wrapper raises `ValueError` before the boundary | Python callers can pass negatives; raw nanobind would raise `TypeError` on the implicit conversion — `ValueError` with a clear message is the better Python idiom for a value-range error. |

## Open / Unresolved

| API surface | Issue | Severity |
|---|---|---|
| `sample` / `sample_mv` / `fit_best_of_n[_mv]` `seed` | A negative seed raises nanobind's `TypeError` (uint64 conversion), while NumPy's own seed handling raises `ValueError` — inconsistent with the wrappers' own `ValueError` idiom for range errors. Candidate one-line wrapper check. | Low |
| `fit_best_of_n[_mv]` all-restarts-failed path | Core rethrows the last restart's exception (or `runtime_error` if none recorded); binding pass-through is untested — hard to trigger deliberately. Mapping is generic nanobind translation, so risk is low, but no test pins it. | Low |
