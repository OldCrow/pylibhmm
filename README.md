# pylibhmm

Python bindings for [libhmm](https://github.com/OldCrow/libhmm) v4 — a modern C++20 Hidden Markov Model library with 16 scalar and 3 multivariate emission distributions, canonical log-space inference, and training algorithms.

## Features

**Scalar HMMs**
- All 16 emission distributions: `Gaussian`, `StudentT`, `Gamma`, `LogNormal`, `Beta`, `Weibull`, `Exponential`, `Rayleigh`, `Pareto`, `Uniform`, `ChiSquared`, `VonMises`, `Discrete`, `Poisson`, `Binomial`, `NegativeBinomial`
- `Hmm`, `ForwardBackwardCalculator`, `ViterbiCalculator`, `BaumWelchTrainer`, `MapBaumWelchTrainer`, `ViterbiTrainer`, `SegmentalKMeansTrainer`
- Posterior decoding (`decode_posterior`) and model selection (AIC / BIC / AICc)
- JSON model I/O (`save_json` / `load_json` / `to_json` / `from_json`) — recommended
- Legacy XML model I/O (`save_hmm` / `load_hmm`) — retained for existing files

**Multivariate HMMs** (v0.6.0 / libhmm v4)
- `DiagonalGaussian`, `FullCovGaussian`, `IndependentComponents` distributions
- `HmmMV`, `MVForwardBackwardCalculator`, `MVBaumWelchTrainer`
- `kmeans_init` for k-means++ seeded initialisation
- MV JSON I/O (`save_json_mv` / `load_json_mv` / `to_json_mv` / `from_json_mv`)
- NumPy-friendly: observations as 2-D `float64` arrays, sequences as lists of 2-D arrays

**General**
- Python 3.11+ packaging via `scikit-build-core` + `nanobind`
- Real-data examples: DAX regime detection, elk movement, earthquake counts, S&P 500, wind direction

## Quick start

**Scalar HMM:**
```python
import numpy as np
import pylibhmm

hmm = pylibhmm.Hmm(2)
hmm.set_pi(np.array([0.75, 0.25], dtype=np.float64))
hmm.set_trans(np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float64))

fair = pylibhmm.Discrete(6)
for i in range(6):
    fair.set_probability(i, 1.0 / 6.0)

loaded = pylibhmm.Discrete(6)
for i in range(5):
    loaded.set_probability(i, 0.125)
loaded.set_probability(5, 0.375)

hmm.set_distribution(0, fair)
hmm.set_distribution(1, loaded)

obs = np.array([0, 1, 5, 4, 2], dtype=np.float64)
fb = pylibhmm.ForwardBackwardCalculator(hmm, obs)
print(fb.log_probability)

pylibhmm.save_json(hmm, "model.json")
hmm2 = pylibhmm.load_json("model.json")
```

**Multivariate HMM:**
```python
import numpy as np
import pylibhmm

# 3-state, 2-feature diagonal-Gaussian HMM
hmm = pylibhmm.HmmMV(3)
hmm.set_pi(np.array([0.5, 0.3, 0.2]))
hmm.set_trans(np.array([[0.8, 0.1, 0.1],
                         [0.1, 0.8, 0.1],
                         [0.1, 0.1, 0.8]]))

for i, (mean, var) in enumerate([
    ([0.0, 0.0], [1.0, 1.0]),
    ([3.0, 3.0], [1.5, 1.5]),
    ([6.0, 6.0], [2.0, 2.0]),
]):
    hmm.set_distribution(i, pylibhmm.DiagonalGaussian(
        np.array(mean), np.array(var)))

# observations: 2-D float64 array, shape (T, D)
obs = np.random.randn(200, 2).astype(np.float64)

# Initialise with k-means++ then train
pylibhmm.kmeans_init(hmm, [obs])
trainer = pylibhmm.MVBaumWelchTrainer(hmm, [obs])
trainer.train()

fb = pylibhmm.MVForwardBackwardCalculator(hmm, obs)
print(fb.log_probability)

pylibhmm.save_json_mv(hmm, "model_mv.json")
hmm2 = pylibhmm.load_json_mv("model_mv.json")
```

## Build and install

Prerequisites:

- Python 3.11+
- CMake 3.25+
- C++20 compiler

Install locally:

```bash
pip install .
```

Run tests:

```bash
pip install ".[test]"
pytest
```

## Dependency strategy

`pylibhmm` prefers a local sibling `../libhmm` source tree if present. If not found, CMake falls back to `FetchContent` for `libhmm` tag `v4.2.5`.

## Notes on wheel portability

`libhmm` defaults to machine-tuned SIMD flags (`-march=native` or CPU-selected MSVC `/arch`). That is ideal for local builds but requires extra care for portable binary wheels. See `docs/COMPATIBILITY.md`.

## Documentation

- `docs/API.md` — bound API surface
- `docs/DEVELOPMENT.md` — contributor workflow
- `docs/COMPATIBILITY.md` — platform/version/SIMD notes

## License

MIT
