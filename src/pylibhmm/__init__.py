"""pylibhmm — Python bindings for libhmm.

Exposes libhmm's HMM model, emission distributions, calculators, and trainers
through a NumPy-friendly Python API.  The C++ extension lives in
:mod:`pylibhmm._core`; this module wraps it with input validation and array
coercion so callers can pass plain lists, arrays, or any array-like without
worrying about dtype or memory layout.

Distributions are re-exported as top-level names (e.g. ``pylibhmm.Gaussian``).
Model I/O is available via :func:`save_json` / :func:`load_json` (JSON, recommended)
or :func:`save_hmm` / :func:`load_hmm` (XML, legacy).

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
from typing import Iterable, NamedTuple

import numpy as np

from . import _core  # pylint: disable=import-self


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


class MapBaumWelchTrainer(_core.MapBaumWelchTrainer):
    """MAP-EM Baum-Welch trainer with symmetric Dirichlet priors.

    Each :meth:`train` call executes one MAP-EM iteration. The MAP objective
    ``log P(O|λ) + log P(λ|c)`` is guaranteed monotone; the likelihood alone
    is not when ``pseudo_count > 0``. Use :meth:`compute_log_prior` to form
    the correct convergence criterion.

    Args:
        hmm: The :class:`Hmm` to train. Mutated in place.
        sequences: Iterable of 1-D array-like observation sequences.
        pseudo_count: Dirichlet pseudo-count c ≥ 0. c=0 recovers standard MLE.

    Note:
        The HMM must outlive this object (nanobind keep-alive enforced
        at the C++ layer).
    """

    def __init__(self, hmm: Hmm, sequences, pseudo_count: float = 1.0):
        super().__init__(hmm, _as_sequence_list(sequences), pseudo_count)


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


class ModelCriteria(NamedTuple):
    """AIC, BIC, and AICc for a fitted HMM.

    All criteria follow the **lower is better** convention.
    Returned by :func:`evaluate_model`.
    """

    aic: float
    bic: float
    aicc: float


# repr(d) returns the distribution's text representation.
# Use named properties (d.mu, d.sigma, …) for programmatic parameter access
# rather than parsing the repr string.
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
VonMises = _core.VonMises

TrainingConfig = _core.TrainingConfig
training_preset_fast = _core.training_preset_fast
training_preset_balanced = _core.training_preset_balanced
training_preset_precise = _core.training_preset_precise


def count_free_parameters(hmm: Hmm) -> int:
    """Count the free parameters of a fitted HMM.

    Free parameters = N(N−1) transitions + (N−1) initial + Σ emission params.

    Args:
        hmm: A fitted :class:`Hmm` instance.

    Returns:
        Total free parameter count.
    """
    return _core.count_free_parameters(hmm)


def compute_aic(log_likelihood: float, k: int) -> float:
    """AIC = 2k − 2 logL (lower is better)."""
    return _core.compute_aic(log_likelihood, k)


def compute_bic(log_likelihood: float, k: int, n: int) -> float:
    """BIC = k ln(n) − 2 logL (lower is better)."""
    return _core.compute_bic(log_likelihood, k, n)


def compute_aicc(log_likelihood: float, k: int, n: int) -> float:
    """AICc = AIC + 2k(k+1)/(n−k−1) (lower is better)."""
    return _core.compute_aicc(log_likelihood, k, n)


def evaluate_model(hmm: Hmm, log_likelihood: float, sequence_length: int) -> ModelCriteria:
    """Compute AIC, BIC, and AICc for a fitted HMM.

    Convenience wrapper: derives k from :func:`count_free_parameters`, then
    calls the three individual criterion functions.

    Args:
        hmm: The fitted :class:`Hmm`.
        log_likelihood: log P(O | λ) from :class:`ForwardBackwardCalculator`.
        sequence_length: Number of observations T.

    Returns:
        :class:`ModelCriteria` with ``aic``, ``bic``, and ``aicc`` fields.
    """
    aic, bic, aicc = _core.evaluate_model(hmm, log_likelihood, sequence_length)
    return ModelCriteria(aic=aic, bic=bic, aicc=aicc)


def to_json(hmm: Hmm) -> str:
    """Serialize an HMM to a compact JSON string.

    Args:
        hmm: The model to serialize.

    Returns:
        JSON string that round-trips exactly through :func:`from_json`.
    """
    return _core.to_json(hmm)


def from_json(src: str) -> Hmm:
    """Deserialize an HMM from a JSON string produced by :func:`to_json`.

    Args:
        src: JSON string.

    Returns:
        Reconstructed :class:`Hmm` instance.

    Raises:
        RuntimeError: On malformed input.
    """
    return _core.from_json(src)


def save_json(hmm: Hmm, filepath: str | Path) -> None:
    """Write an HMM as JSON to *filepath*.

    Args:
        hmm: The model to serialize.
        filepath: Destination path.

    Raises:
        RuntimeError: If the file cannot be written.
    """
    _core.save_json(hmm, str(filepath))


def load_json(filepath: str | Path) -> Hmm:
    """Read and deserialize an HMM from a JSON file.

    Args:
        filepath: Path to a JSON model file written by :func:`save_json`.

    Returns:
        Reconstructed :class:`Hmm` instance.

    Raises:
        RuntimeError: If the file cannot be read or parsed.
    """
    return _core.load_json(str(filepath))


def load_hmm(filepath: str | Path):
    """Load an HMM from a legacy XML file written by :func:`save_hmm`.

    .. deprecated::
        Prefer :func:`load_json` for new code.  XML support is retained
        for reading existing files only.

    Args:
        filepath: Path to the XML model file.

    Returns:
        Reconstructed :class:`Hmm` instance.

    Raises:
        RuntimeError: If the file cannot be read or parsed.
    """
    return _core.load_hmm(str(filepath))


def save_hmm(hmm: Hmm, filepath: str | Path) -> None:
    """Save an HMM to a legacy XML file.

    .. deprecated::
        Prefer :func:`save_json` for new code.  XML support is retained
        for producing files readable by older tooling only.

    Args:
        hmm: The model to serialize.
        filepath: Destination path.  Parent directory must exist.

    Raises:
        RuntimeError: If the file cannot be written.
    """
    _core.save_hmm(hmm, str(filepath))


# =============================================================================
# v4 Multivariate API
# =============================================================================

# Re-export MV distribution types directly from the extension.
MVEmissionDistribution = _core.MVEmissionDistribution
DiagonalGaussian       = _core.DiagonalGaussian
FullCovGaussian        = _core.FullCovGaussian
IndependentComponents  = _core.IndependentComponents


def _as_mv_sequence_list(sequences) -> list[np.ndarray]:
    """Validate and coerce each element of *sequences* to a 2-D (T×D) float64 array.

    Args:
        sequences: Iterable of array-like multivariate observation sequences.
            Each element must be convertible to shape (T_i, D) with T_i > 0.

    Returns:
        List of C-contiguous float64 2-D arrays.

    Raises:
        ValueError: If *sequences* is empty or any element fails validation.
    """
    converted: list[np.ndarray] = []
    for i, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"sequences[{i}] must be a 2-D array (T, D)")
        if arr.shape[0] == 0:
            raise ValueError(f"sequences[{i}] must contain at least one observation")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"sequences[{i}] must contain only finite values")
        converted.append(np.ascontiguousarray(arr))
    if not converted:
        raise ValueError("sequences must contain at least one observation sequence")
    return converted


class HmmMV(_core.HmmMV):
    """Multivariate Hidden Markov Model with Python-level input validation.

    Wraps :class:`pylibhmm._core.HmmMV`.  Emission distributions must be
    multivariate types (:class:`DiagonalGaussian`, :class:`FullCovGaussian`,
    or :class:`IndependentComponents`).
    Observation sequences are 2-D float64 NumPy arrays with shape ``(T, D)``.

    Args:
        num_states: Number of hidden states.  Must be > 0.
    """

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


class MVForwardBackwardCalculator(_core.MVForwardBackwardCalculator):
    """Multivariate forward-backward calculator with automatic array coercion.

    Args:
        hmm: A :class:`HmmMV` instance.
        observations: 2-D array-like of shape ``(T, D)``.
    """

    def __init__(self, hmm: HmmMV, observations):
        obs = np.ascontiguousarray(np.asarray(observations, dtype=np.float64))
        if obs.ndim != 2:
            raise ValueError("observations must be a 2-D array (T, D)")
        super().__init__(hmm, obs)


class MVBaumWelchTrainer(_core.MVBaumWelchTrainer):
    """Multivariate Baum-Welch (EM) trainer with automatic sequence coercion.

    Args:
        hmm: The :class:`HmmMV` to train.  Mutated in place by :meth:`train`.
        sequences: Iterable of 2-D array-like sequences, each shape ``(T_i, D)``.
    """

    def __init__(self, hmm: HmmMV, sequences):
        super().__init__(hmm, _as_mv_sequence_list(sequences))


def kmeans_init(hmm: HmmMV, sequences, seed: int = 42) -> None:
    """Initialise a multivariate HMM's emission distributions via k-means++.

    Runs Lloyd's algorithm with k-means++ seeding on all observation vectors.
    Each state's emission distribution is fitted to its assigned cluster members.
    Call before :class:`MVBaumWelchTrainer` to provide a data-driven start.

    Args:
        hmm: The :class:`HmmMV` to initialise.  Emission distributions are
            updated in place.
        sequences: Iterable of 2-D array-like sequences, each shape ``(T_i, D)``.
        seed: Integer RNG seed for reproducible k-means++ seeding (default 42).
    """
    _core.kmeans_init(hmm, _as_mv_sequence_list(sequences), int(seed))


def to_json_mv(hmm: HmmMV) -> str:
    """Serialize a multivariate HMM to a JSON string.

    The JSON schema includes ``\"obs_type\": \"multivariate\"`` so it is
    rejected by the scalar :func:`from_json`.
    """
    return _core.to_json_mv(hmm)


def from_json_mv(src: str) -> HmmMV:
    """Deserialize a multivariate HMM from a JSON string."""
    return _core.from_json_mv(src)


def save_json_mv(hmm: HmmMV, filepath: str | Path) -> None:
    """Write a multivariate HMM as JSON to *filepath*."""
    _core.save_json_mv(hmm, str(filepath))


def load_json_mv(filepath: str | Path) -> HmmMV:
    """Read and deserialize a multivariate HMM from a JSON file."""
    return _core.load_json_mv(str(filepath))


def count_free_parameters_mv(hmm: HmmMV) -> int:
    """Count the free parameters of a fitted multivariate HMM."""
    return _core.count_free_parameters_mv(hmm)


__all__ = [
    # Scalar API (v3 compatible)
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
    "MapBaumWelchTrainer",
    "ModelCriteria",
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
    "VonMises",
    "Weibull",
    "compute_aic",
    "compute_aicc",
    "compute_bic",
    "count_free_parameters",
    "evaluate_model",
    "from_json",
    "load_hmm",
    "load_json",
    "save_hmm",
    "save_json",
    "to_json",
    "training_preset_balanced",
    "training_preset_fast",
    "training_preset_precise",
    # Multivariate API (v4)
    "DiagonalGaussian",
    "FullCovGaussian",
    "HmmMV",
    "IndependentComponents",
    "MVBaumWelchTrainer",
    "MVEmissionDistribution",
    "MVForwardBackwardCalculator",
    "count_free_parameters_mv",
    "from_json_mv",
    "kmeans_init",
    "load_json_mv",
    "save_json_mv",
    "to_json_mv",
]
