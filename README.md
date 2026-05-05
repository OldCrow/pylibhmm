# pylibhmm

Python bindings for [libhmm](https://github.com/OldCrow/libhmm), a modern C++20 Hidden Markov Model library with 15 emission distributions, canonical log-space inference, and training algorithms.

## Features

- Native bindings for `Hmm`, `ForwardBackwardCalculator`, `ViterbiCalculator`, and the trainer classes
- Bindings for all 15 emission distributions in libhmm
- NumPy integration for vectors/matrices/observation sequences
- JSON model I/O (save_json / load_json / to_json / from_json) - recommended
- Legacy XML model I/O (save_hmm / load_hmm) - retained for existing files
- Python 3.11+ packaging via `scikit-build-core` + `nanobind`

## Quick start

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

# Save and reload the model (JSON - recommended)
pylibhmm.save_json(hmm, "model.json")
hmm2 = pylibhmm.load_json("model.json")

# Or round-trip through a string
json_str = pylibhmm.to_json(hmm)
hmm3 = pylibhmm.from_json(json_str)
```

## Build and install

Prerequisites:

- Python 3.11+
- CMake 3.20+
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

`pylibhmm` prefers a local sibling `../libhmm` source tree if present. If not found, CMake falls back to `FetchContent` for `libhmm` tag `v3.4.0`.

## Notes on wheel portability

`libhmm` defaults to machine-tuned SIMD flags (`-march=native` or CPU-selected MSVC `/arch`). That is ideal for local builds but requires extra care for portable binary wheels. See `docs/COMPATIBILITY.md`.

## Documentation

- `docs/API.md` — bound API surface
- `docs/DEVELOPMENT.md` — contributor workflow
- `docs/COMPATIBILITY.md` — platform/version/SIMD notes
- `WARP.md` — Warp agent guide for this repository

## License

MIT
