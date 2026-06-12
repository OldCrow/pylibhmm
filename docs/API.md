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
