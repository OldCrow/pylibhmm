"""Regression tests for Finding 1 (audit 2026-07-04): use-after-free in
calculator bindings.

Before the fix, ForwardBackwardCalculator and MVViterbiCalculator stored their
observation sequence by reference to a temporary that died when the __init__
lambda returned.  Any subsequent compute() / decode() call re-read freed
memory, producing silently-corrupted results rather than a crash.

Each test below:
  1. Constructs a calculator and records the correct log-probability.
  2. Churns the heap to overwrite freed storage with different data.
  3. Re-runs compute() / decode() and asserts the result matches step 1.

A regression means the result changes (silent corruption) or crashes.
"""

import gc

import numpy as np
import pytest

import pylibhmm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _churn_heap(n: int = 2000, size: int = 200) -> None:
    """Allocate and immediately discard n arrays to overwrite freed memory."""
    junk = [np.full(size, 1e300) for _ in range(n)]
    del junk
    gc.collect()


def _make_scalar_hmm() -> pylibhmm.Hmm:
    hmm = pylibhmm.Hmm(2)
    hmm.set_pi(np.array([0.5, 0.5]))
    hmm.set_trans(np.array([[0.9, 0.1], [0.1, 0.9]]))
    hmm.set_distribution(0, pylibhmm.Gaussian(0.0, 1.0))
    hmm.set_distribution(1, pylibhmm.Gaussian(5.0, 1.0))
    return hmm


def _make_mv_hmm() -> pylibhmm.HmmMV:
    hmm = pylibhmm.HmmMV(2)
    hmm.set_pi(np.array([0.5, 0.5]))
    hmm.set_trans(np.array([[0.9, 0.1], [0.1, 0.9]]))
    d0 = pylibhmm.DiagonalGaussian(2)
    d0.set_parameters(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    d1 = pylibhmm.DiagonalGaussian(2)
    d1.set_parameters(np.array([5.0, 5.0]), np.array([1.0, 1.0]))
    hmm.set_distribution(0, d0)
    hmm.set_distribution(1, d1)
    return hmm


# ---------------------------------------------------------------------------
# Scalar ForwardBackwardCalculator
# ---------------------------------------------------------------------------

class TestFBCalcUAF:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.hmm = _make_scalar_hmm()
        self.obs = np.concatenate(
            [rng.normal(0, 1, 50), rng.normal(5, 1, 50)]
        )

    def test_no_arg_compute_stable_after_heap_churn(self):
        """No-arg compute() must re-use the construction-time sequence safely."""
        fb = pylibhmm.ForwardBackwardCalculator(self.hmm, self.obs)
        lp0 = fb.log_probability
        assert np.isfinite(lp0), "initial log_probability must be finite"

        _churn_heap()

        fb.compute()
        assert fb.log_probability == pytest.approx(lp0, rel=1e-9), (
            f"log_probability corrupted after heap churn: {fb.log_probability} != {lp0}"
        )

    def test_compute_with_new_obs_stable(self):
        """compute(new_obs) must produce the same result on identical data."""
        fb = pylibhmm.ForwardBackwardCalculator(self.hmm, self.obs)
        lp0 = fb.log_probability

        _churn_heap()

        fb.compute(self.obs)
        assert fb.log_probability == pytest.approx(lp0, rel=1e-9)

    def test_decode_posterior_stable_after_heap_churn(self):
        """decode_posterior() must return consistent states after heap churn."""
        fb = pylibhmm.ForwardBackwardCalculator(self.hmm, self.obs)
        seq0 = fb.decode_posterior()

        _churn_heap()

        fb.compute()
        seq1 = fb.decode_posterior()
        np.testing.assert_array_equal(seq0, seq1)


# ---------------------------------------------------------------------------
# Scalar ViterbiCalculator
# ---------------------------------------------------------------------------

class TestViterbiCalcUAF:
    def setup_method(self):
        rng = np.random.default_rng(1)
        self.hmm = _make_scalar_hmm()
        self.obs = np.concatenate(
            [rng.normal(0, 1, 50), rng.normal(5, 1, 50)]
        )

    def test_decode_stable_after_heap_churn(self):
        """decode() must re-run Viterbi safely after heap churn."""
        vc = pylibhmm.ViterbiCalculator(self.hmm, self.obs)
        lp0 = vc.log_probability
        seq0 = vc.decode()
        assert np.isfinite(lp0)

        _churn_heap()

        seq1 = vc.decode()
        assert vc.log_probability == pytest.approx(lp0, rel=1e-9)
        np.testing.assert_array_equal(seq0, seq1)


# ---------------------------------------------------------------------------
# MV MVViterbiCalculator
# ---------------------------------------------------------------------------

class TestMVViterbiCalcUAF:
    def setup_method(self):
        rng = np.random.default_rng(2)
        self.hmm = _make_mv_hmm()
        self.obs = np.vstack(
            [rng.normal(0, 1, (50, 2)), rng.normal(5, 1, (50, 2))]
        )

    def test_decode_stable_after_heap_churn(self):
        """MVViterbiCalculator.decode() must re-run safely after heap churn."""
        vc = pylibhmm.MVViterbiCalculator(self.hmm, self.obs)
        lp0 = vc.log_probability
        seq0 = vc.decode()
        assert np.isfinite(lp0), f"initial log_probability not finite: {lp0}"

        _churn_heap()

        seq1 = vc.decode()
        assert vc.log_probability == pytest.approx(lp0, rel=1e-9), (
            f"log_probability corrupted: {vc.log_probability} != {lp0}"
        )
        np.testing.assert_array_equal(seq0, seq1)


# ---------------------------------------------------------------------------
# MV MVForwardBackwardCalculator
# ---------------------------------------------------------------------------

class TestMVFBCalcUAF:
    def setup_method(self):
        rng = np.random.default_rng(3)
        self.hmm = _make_mv_hmm()
        self.obs = np.vstack(
            [rng.normal(0, 1, (50, 2)), rng.normal(5, 1, (50, 2))]
        )

    def test_log_probability_stable_after_heap_churn(self):
        """MVForwardBackwardCalculator log_probability must survive heap churn."""
        fb = pylibhmm.MVForwardBackwardCalculator(self.hmm, self.obs)
        lp0 = fb.log_probability
        assert np.isfinite(lp0)

        _churn_heap()

        # Access log_probability again — reads from calc_ which holds *data_
        assert fb.log_probability == pytest.approx(lp0, rel=1e-9)

    def test_decode_posterior_stable_after_heap_churn(self):
        """MVFBCalc.decode_posterior() must return consistent states."""
        fb = pylibhmm.MVForwardBackwardCalculator(self.hmm, self.obs)
        seq0 = fb.decode_posterior()

        _churn_heap()

        seq1 = fb.decode_posterior()
        np.testing.assert_array_equal(seq0, seq1)
