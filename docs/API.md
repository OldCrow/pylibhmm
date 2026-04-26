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
