# pylibhmm API overview

`pylibhmm` exposes libhmm functionality through a native `_core` module and a thin Python wrapper layer in `src/pylibhmm/__init__.py`.

## Core classes

- `Hmm(num_states)`
  - `set_pi(pi: 1D float64)`
  - `set_trans(trans: 2D float64)`
  - `set_distribution(state: int, distribution: EmissionDistribution)`
  - `get_distribution(state)`
  - `validate()`

- `ForwardBackwardCalculator(hmm, observations)`
  - `compute(observations=None)`
  - `log_probability`
  - `probability`
  - `get_log_forward_variables() -> ndarray[T, N]`
  - `get_log_backward_variables() -> ndarray[T, N]`

- `ViterbiCalculator(hmm, observations)`
  - `decode() -> ndarray[int64]`
  - `log_probability`
  - `get_state_sequence()`

## Trainers

- `BaumWelchTrainer(hmm, sequences)`
- `ViterbiTrainer(hmm, sequences, config=TrainingConfig())`
- `SegmentalKMeansTrainer(hmm, sequences)`

`TrainingConfig` fields:

- `convergence_tolerance`
- `max_iterations`
- `convergence_window`
- `enable_progress_reporting`

Presets:

- `training_preset_fast()`
- `training_preset_balanced()`
- `training_preset_precise()`

## Distributions

Bound classes:

- `Discrete`, `Binomial`, `NegativeBinomial`, `Poisson`
- `Gaussian`, `Exponential`, `Gamma`, `LogNormal`, `Pareto`, `Beta`, `Uniform`, `Weibull`, `Rayleigh`, `StudentT`, `ChiSquared`

Common methods:

- `pdf(x: float)`
- `log_pdf(x: float | ndarray[float64])`
- `fit(data: ndarray[float64])`
- `fit_weighted(data, weights)`
- `reset()`
- `is_discrete`

Most distributions also expose:

- `cdf(x: float)`
- `mean`, `variance`, `std`

## XML I/O

- `load_hmm(filepath)`
- `save_hmm(hmm, filepath)`

## Model-level operations (v0.12.0, libhmm v4.4.0)

Scalar and multivariate variants share semantics; the MV names carry an
`_mv` suffix and take an `HmmMV`.

- `clone_hmm(hmm) -> Hmm` / `clone_hmm_mv(hmm) -> HmmMV`
  — explicit deep copy (libhmm deletes the HMM copy constructor to keep
  the cost visible). MV emission slots that are unset stay unset.

- `sample(hmm, T, seed=None) -> (observations, states)` /
  `sample_mv(hmm, T, seed=None)`
  — draw one sequence of length `T`: 1-D `float64` observations (scalar)
  or `(T, D)` (MV), plus an `int64` state path. `seed` gives a
  reproducible draw; `None` uses a non-deterministic module-level RNG.
  Raises `RuntimeError` if pi or a visited transition row sums to zero.

- `fit_best_of_n(hmm, sequences, n_restarts, seed=42, max_iters=500) -> float` /
  `fit_best_of_n_mv(...)`
  — multi-restart Baum-Welch; keeps the best model by total
  forward-backward log-likelihood and copies it into `hmm` in place.
  Restart 0 trains from the current parameters unrandomised, so the
  result is at least as good as a single run. Scalar restarts refit
  emissions to small random subsamples; MV restarts re-seed via
  k-means++. Returns the best total log-likelihood.

- `HmmTopology` — enum: `Ergodic`, `LeftToRight`, `LeftToRightSkip`,
  `Banded`.

- `initialize_topology(hmm, topology, max_skip=1)` /
  `initialize_topology_mv(...)`
  — overwrite the transition matrix: uniform over the topology's valid
  transitions, exactly 0 elsewhere. Only the transition matrix is
  managed — pi stays the caller's responsibility (a left-to-right model
  conventionally starts with a point mass on state 0).

- `enforce_topology(hmm, topology, max_skip=1)` /
  `enforce_topology_mv(...)`
  — re-impose the mask after an M-step: zero invalid entries,
  renormalise rows, reset a row with no remaining valid mass to uniform
  over its valid entries. Cheap; call after each `train()` iteration to
  make the constraint unconditional.

---

## Multivariate API (v0.6.0)

Observations are 2-D `float64` NumPy arrays of shape `(T, D)`. Sequences for training are Python lists of such arrays.

### Multivariate distributions

All MV distributions accept `ObservationVectorView` (a D-element row of a 2-D array).

- `DiagonalGaussian(mean: ndarray[D], variance: ndarray[D])`
  - `mean`, `variance` — properties returning 1-D arrays
  - `log_pdf(x: ndarray[D]) -> float`
  - `fit(X: ndarray[T,D])`, `fit_weighted(X, weights)`
  - `set_parameters(mean, variance)`, `set_means(mean)`, `set_variances(variance)`

- `FullCovGaussian(mean: ndarray[D], covariance: ndarray[D,D])`
  - `mean`, `covariance` — properties returning arrays
  - `log_pdf(x: ndarray[D]) -> float`
  - `fit(X: ndarray[T,D])`, `fit_weighted(X, weights)`
  - `set_mean(mean)`, `set_covariance(cov)`, `set_parameters(mean, cov)`

- `IndependentComponents(components: list[EmissionDistribution])`
  - `get_component(d: int) -> EmissionDistribution`
  - `set_component(d: int, dist: EmissionDistribution)`
  - `log_pdf(x: ndarray[D]) -> float`
  - `fit(X: ndarray[T,D])`, `fit_weighted(X, weights)`

### Multivariate HMM

- `HmmMV(num_states)`
  - Same `set_pi`, `set_trans`, `set_distribution`, `get_distribution`, `validate` interface as `Hmm`.

### Multivariate calculators

- `MVForwardBackwardCalculator(hmm: HmmMV, observations: ndarray[T,D])`
  - `log_probability`, `probability`
  - `decode_posterior() -> ndarray[int64]`
  - `compute(observations: ndarray[T,D])`

- `MVViterbiCalculator(hmm: HmmMV, observations: ndarray[T,D])`
  - `decode() -> ndarray[int64]`
  - `log_probability`

### Multivariate trainers

- `MVBaumWelchTrainer(hmm: HmmMV, sequences: list[ndarray[T_i, D]])`
  - `train()`
  - `has_converged() -> bool`
  - `get_log_likelihood() -> float`

- `kmeans_init(hmm: HmmMV, sequences: list[ndarray[T_i, D]]) -> None`
  — k-means++ seeded Lloyd's initialisation; call before `MVBaumWelchTrainer`.

### Multivariate JSON I/O

- `save_json_mv(hmm: HmmMV, filepath: str)`
- `load_json_mv(filepath: str) -> HmmMV`
- `to_json_mv(hmm: HmmMV) -> str`
- `from_json_mv(json_str: str) -> HmmMV`
- `count_free_parameters_mv(hmm: HmmMV) -> int`
