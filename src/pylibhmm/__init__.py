"""pylibhmm — Python bindings for libhmm.

Exposes libhmm's HMM model, emission distributions, calculators, and trainers
through a NumPy-friendly Python API.  The C++ extension lives in
:mod:`pylibhmm._core`; this module wraps it with input validation and array
coercion so callers can pass plain lists, arrays, or any array-like without
worrying about dtype or memory layout.

Distributions are re-exported as top-level names (e.g. ``pylibhmm.Gaussian``).
Model I/O is available via :func:`load_hmm` and :func:`save_hmm`.

Example::

    import numpy as np
    import pylibhmm

    hmm = pylibhmm.Hmm(2)
    hmm.set_pi(np.array([0.6, 0.4]))
    hmm.set_trans(np.array([[0.7, 0.3], [0.4, 0.6]]))
    hmm.set_distribution(0, pylibhmm.Gaussian(mu=0.0, sigma=1.0))
    hmm.set_distribution(1, pylibhmm.Gaussian(mu=5.0, sigma=1.0))

    obs = np.array([0.1, 4.9, 5.2, 0.3, -0.1], dtype=np.float64)
    calc = pylibhmm.ForwardBackwardCalculator(hmm, obs)
    print(f"log P(obs|model) = {calc.log_probability:.4f}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from . import _core


def _as_f64_1d(values, name: str) -> np.ndarray:
    """Coerce *values* to a contiguous 1-D float64 array.

    Args:
        values: Any array-like input.
        name: Parameter name used in error messages.

    Raises:
        ValueError: If the result is not 1-D or contains non-finite values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr)


def _as_f64_2d(values, name: str) -> np.ndarray:
    """Coerce *values* to a contiguous 2-D float64 array.

    Args:
        values: Any array-like input.
        name: Parameter name used in error messages.

    Raises:
        ValueError: If the result is not 2-D or contains non-finite values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr)


def _as_sequence_list(sequences: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Validate and coerce each element of *sequences* to a 1-D float64 array.

    Args:
        sequences: Iterable of array-like observation sequences.

    Returns:
        List of contiguous float64 arrays.

    Raises:
        ValueError: If *sequences* is empty or any element fails validation.
    """
    converted: list[np.ndarray] = []
    for i, seq in enumerate(sequences):
        converted.append(_as_f64_1d(seq, f"sequences[{i}]"))
    if not converted:
        raise ValueError("sequences must contain at least one observation sequence")
    return converted


class Hmm(_core.Hmm):
    """Hidden Markov Model with Python-level input validation.

    Wraps :class:`pylibhmm._core.Hmm` and validates inputs to
    :meth:`set_pi` and :meth:`set_trans` before forwarding to C++.
    Emission distributions are assigned via :meth:`set_distribution`
    (inherited directly from the extension).

    Args:
        num_states: Number of hidden states.  Must be > 0.
    """

    def __init__(self, num_states: int):
        if num_states <= 0:
            raise ValueError("num_states must be greater than 0")
        super().__init__(num_states)

    def set_pi(self, pi) -> None:
        """Set the initial state distribution π.

        Args:
            pi: 1-D array of length ``num_states``.  Values are passed
                through to libhmm without renormalization; callers are
                responsible for ensuring a valid probability distribution.

        Raises:
            ValueError: If *pi* is not 1-D, has the wrong length, or
                contains non-finite values.
        """
        arr = _as_f64_1d(pi, "pi")
        if arr.shape[0] != self.num_states:
            raise ValueError("pi length must match num_states")
        super().set_pi(arr)

    def set_trans(self, trans) -> None:
        """Set the state transition probability matrix.

        Args:
            trans: 2-D array of shape ``(num_states, num_states)``.
                Row *i* gives the transition probabilities from state *i*.
                Rows are passed through without renormalization.

        Raises:
            ValueError: If *trans* is not 2-D, has the wrong shape, or
                contains non-finite values.
        """
        arr = _as_f64_2d(trans, "trans")
        expected = (self.num_states, self.num_states)
        if arr.shape != expected:
            raise ValueError(f"trans must have shape {expected}")
        super().set_trans(arr)


class ForwardBackwardCalculator(_core.ForwardBackwardCalculator):
    """Forward-backward calculator with automatic array coercion.

    Runs the forward-backward algorithm in log-space, computing
    log P(observations | model) and the full α/β variable matrices.

    Args:
        hmm: A validated :class:`Hmm` instance.
        observations: 1-D array-like observation sequence.

    Note:
        The HMM must outlive this object (nanobind keep-alive enforced
        at the C++ layer).
    """

    def __init__(self, hmm: Hmm, observations):
        super().__init__(hmm, _as_f64_1d(observations, "observations"))

    def compute(self, observations=None):
        """Re-run the algorithm, optionally on a new observation sequence.

        Args:
            observations: New 1-D observation sequence.  If ``None``,
                re-uses the sequence supplied at construction.
        """
        if observations is None:
            return super().compute()
        return super().compute(_as_f64_1d(observations, "observations"))


class ViterbiCalculator(_core.ViterbiCalculator):
    """Viterbi decoder with automatic array coercion.

    Finds the most probable hidden state sequence via the Viterbi
    algorithm in log-space.  Call :meth:`decode` to run and retrieve
    the sequence as a 1-D int64 NumPy array.

    Args:
        hmm: A validated :class:`Hmm` instance.
        observations: 1-D array-like observation sequence.

    Note:
        The HMM must outlive this object (nanobind keep-alive enforced
        at the C++ layer).
    """

    def __init__(self, hmm: Hmm, observations):
        super().__init__(hmm, _as_f64_1d(observations, "observations"))


class BaumWelchTrainer(_core.BaumWelchTrainer):
    """Baum-Welch (EM) trainer with automatic sequence coercion.

    Each :meth:`train` call executes one full EM iteration: an E-step
    that accumulates γ and ξ statistics across all sequences, followed
    by an M-step that updates π, the transition matrix, and all emission
    distributions in place.

    Args:
        hmm: The :class:`Hmm` to train.  Mutated in place by
            :meth:`train`.
        sequences: Iterable of 1-D array-like observation sequences.
            At least one sequence is required.

    Note:
        The HMM must outlive this object (nanobind keep-alive enforced
        at the C++ layer).
    """

    def __init__(self, hmm: Hmm, sequences):
        super().__init__(hmm, _as_sequence_list(sequences))


class ViterbiTrainer(_core.ViterbiTrainer):
    """Viterbi trainer with automatic sequence coercion.

    Each :meth:`train` call iterates Viterbi decoding + hard-assignment
    M-step until convergence or ``config.max_iterations`` is reached.
    Convergence is declared when the absolute change in total
    log-probability stays below ``config.convergence_tolerance`` for
    ``config.convergence_window`` consecutive iterations.

    Args:
        hmm: The :class:`Hmm` to train.  Mutated in place.
        sequences: Iterable of 1-D array-like observation sequences.
        config: Training parameters.  Defaults to :class:`TrainingConfig`
            with standard settings.  Use :func:`training_preset_fast`,
            :func:`training_preset_balanced`, or
            :func:`training_preset_precise` for named presets.

    Note:
        The HMM must outlive this object (nanobind keep-alive enforced
        at the C++ layer).
    """

    def __init__(self, hmm: Hmm, sequences, config=None):
        if config is None:
            config = TrainingConfig()
        super().__init__(hmm, _as_sequence_list(sequences), config)


class SegmentalKMeansTrainer(_core.SegmentalKMeansTrainer):
    """Segmental K-means trainer with automatic sequence coercion.

    A faster, deterministic alternative to Baum-Welch.  Uses Viterbi-
    based hard segmentation with K-means-style distribution
    re-estimation.  Terminates when the Viterbi segmentation no longer
    changes between iterations.

    Args:
        hmm: The :class:`Hmm` to train.  Mutated in place.
        sequences: Iterable of 1-D array-like observation sequences.

    Note:
        The HMM must outlive this object (nanobind keep-alive enforced
        at the C++ layer).
    """

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
    """Load an HMM from an XML file written by :func:`save_hmm`.

    Args:
        filepath: Path to the XML model file.

    Returns:
        Reconstructed :class:`Hmm` instance.

    Raises:
        RuntimeError: If the file cannot be read or parsed.
    """
    return _core.load_hmm(str(filepath))


def save_hmm(hmm: Hmm, filepath: str | Path) -> None:
    """Save an HMM to an XML file.

    The format is compatible with libhmm's XMLFileReader/XMLFileWriter.
    Distribution types and all parameters are preserved.

    Args:
        hmm: The model to serialize.
        filepath: Destination path.  Parent directory must exist.

    Raises:
        RuntimeError: If the file cannot be written.
    """
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
