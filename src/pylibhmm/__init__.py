"""pylibhmm — Python bindings for libhmm."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from pylibhmm import _core


def _as_f64_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr)


def _as_f64_2d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr)


def _as_sequence_list(sequences: Iterable[np.ndarray]) -> list[np.ndarray]:
    converted: list[np.ndarray] = []
    for i, seq in enumerate(sequences):
        converted.append(_as_f64_1d(seq, f"sequences[{i}]"))
    if not converted:
        raise ValueError("sequences must contain at least one observation sequence")
    return converted


class Hmm(_core.Hmm):
    """Python convenience wrapper for libhmm::Hmm."""

    def __init__(self, num_states: int):
        if num_states <= 0:
            raise ValueError("num_states must be greater than 0")
        super().__init__(num_states)

    def set_pi(self, pi) -> None:
        arr = _as_f64_1d(pi, "pi")
        if arr.shape[0] != self.num_states:
            raise ValueError("pi length must match num_states")
        super().set_pi(arr)

    def set_trans(self, trans) -> None:
        arr = _as_f64_2d(trans, "trans")
        expected = (self.num_states, self.num_states)
        if arr.shape != expected:
            raise ValueError(f"trans must have shape {expected}")
        super().set_trans(arr)


class ForwardBackwardCalculator(_core.ForwardBackwardCalculator):
    def __init__(self, hmm: Hmm, observations):
        super().__init__(hmm, _as_f64_1d(observations, "observations"))

    def compute(self, observations=None):
        if observations is None:
            return super().compute()
        return super().compute(_as_f64_1d(observations, "observations"))


class ViterbiCalculator(_core.ViterbiCalculator):
    def __init__(self, hmm: Hmm, observations):
        super().__init__(hmm, _as_f64_1d(observations, "observations"))


class BaumWelchTrainer(_core.BaumWelchTrainer):
    def __init__(self, hmm: Hmm, sequences):
        super().__init__(hmm, _as_sequence_list(sequences))


class ViterbiTrainer(_core.ViterbiTrainer):
    def __init__(self, hmm: Hmm, sequences, config=None):
        if config is None:
            config = TrainingConfig()
        super().__init__(hmm, _as_sequence_list(sequences), config)


class SegmentalKMeansTrainer(_core.SegmentalKMeansTrainer):
    def __init__(self, hmm: Hmm, sequences):
        super().__init__(hmm, _as_sequence_list(sequences))


EmissionDistribution = _core.EmissionDistribution

Discrete = _core.Discrete
Binomial = _core.Binomial
NegativeBinomial = _core.NegativeBinomial
Poisson = _core.Poisson
Gaussian = _core.Gaussian
Exponential = _core.Exponential
Gamma = _core.Gamma
LogNormal = _core.LogNormal
Pareto = _core.Pareto
Beta = _core.Beta
Uniform = _core.Uniform
Weibull = _core.Weibull
Rayleigh = _core.Rayleigh
StudentT = _core.StudentT
ChiSquared = _core.ChiSquared

TrainingConfig = _core.TrainingConfig
training_preset_fast = _core.training_preset_fast
training_preset_balanced = _core.training_preset_balanced
training_preset_precise = _core.training_preset_precise


def load_hmm(filepath: str | Path):
    return _core.load_hmm(str(filepath))


def save_hmm(hmm: Hmm, filepath: str | Path) -> None:
    _core.save_hmm(hmm, str(filepath))


__all__ = [
    "BaumWelchTrainer",
    "Beta",
    "Binomial",
    "ChiSquared",
    "Discrete",
    "EmissionDistribution",
    "Exponential",
    "ForwardBackwardCalculator",
    "Gamma",
    "Gaussian",
    "Hmm",
    "LogNormal",
    "NegativeBinomial",
    "Pareto",
    "Poisson",
    "Rayleigh",
    "SegmentalKMeansTrainer",
    "StudentT",
    "TrainingConfig",
    "Uniform",
    "ViterbiCalculator",
    "ViterbiTrainer",
    "Weibull",
    "load_hmm",
    "save_hmm",
    "training_preset_balanced",
    "training_preset_fast",
    "training_preset_precise",
]
