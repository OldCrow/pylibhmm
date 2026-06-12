"""Tests for pylibhmm v4 multivariate API.

Covers DiagonalGaussian, FullCovGaussian, IndependentComponents,
HmmMV, MVForwardBackwardCalculator, MVBaumWelchTrainer, kmeans_init,
and MV JSON round-trip.
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

import pylibhmm as p

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

def make_sequences(n_seq=5, T=30, D=2, seed=0):
    """Generate synthetic 2-D Gaussian sequences."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((T, D)) for _ in range(n_seq)]

def make_hmm_mv(n_states=2, D=2):
    """Construct a minimal HmmMV with DiagonalGaussian distributions."""
    hmm = p.HmmMV(n_states)
    pi = np.ones(n_states) / n_states
    trans = (np.eye(n_states) * 0.8 + np.ones((n_states, n_states)) * 0.1)
    # Normalise rows
    trans /= trans.sum(axis=1, keepdims=True)
    hmm.set_pi(pi)
    hmm.set_trans(trans)
    for s in range(n_states):
        hmm.set_distribution(s, p.DiagonalGaussian(D))
    return hmm

# ---------------------------------------------------------------------------
# DiagonalGaussian
# ---------------------------------------------------------------------------

class TestDiagonalGaussian:
    D = 3

    def test_construction(self):
        d = p.DiagonalGaussian(self.D)
        assert d.dim == self.D

    def test_default_means_zero(self):
        d = p.DiagonalGaussian(self.D)
        np.testing.assert_array_equal(d.means, np.zeros(self.D))

    def test_default_variances_one(self):
        d = p.DiagonalGaussian(self.D)
        np.testing.assert_array_equal(d.variances, np.ones(self.D))

    def test_set_parameters(self):
        d = p.DiagonalGaussian(self.D)
        mu = np.array([1.0, 2.0, 3.0])
        var = np.array([0.5, 1.0, 2.0])
        d.set_parameters(mu, var)
        np.testing.assert_allclose(d.means, mu)
        np.testing.assert_allclose(d.variances, var)

    def test_set_means(self):
        d = p.DiagonalGaussian(self.D)
        mu = np.array([1.5, -0.5, 2.0])
        d.set_means(mu)
        np.testing.assert_allclose(d.means, mu)

    def test_set_variances(self):
        d = p.DiagonalGaussian(self.D)
        var = np.array([2.0, 3.0, 0.5])
        d.set_variances(var)
        np.testing.assert_allclose(d.variances, var)

    def test_log_pdf_single_at_mean_finite(self):
        d = p.DiagonalGaussian(self.D)
        x = np.zeros(self.D)
        lp = d.log_pdf(x)
        assert math.isfinite(lp)
        # log p(0 | μ=0, σ²=1) = -D/2 * log(2π)
        expected = -self.D / 2.0 * math.log(2 * math.pi)
        assert lp == pytest.approx(expected, rel=1e-6)

    def test_log_pdf_batch(self):
        d = p.DiagonalGaussian(self.D)
        X = np.zeros((5, self.D))
        lp = d.log_pdf(X)
        assert lp.shape == (5,)
        assert np.all(np.isfinite(lp))
        # All rows identical → all log-probs equal
        np.testing.assert_allclose(lp, lp[0])

    def test_fit_recovers_mean(self):
        d = p.DiagonalGaussian(self.D)
        mu = np.array([1.0, 2.0, 3.0])
        X = np.tile(mu, (50, 1)) + RNG.standard_normal((50, self.D)) * 0.01
        d.fit(X)
        np.testing.assert_allclose(d.means, mu, atol=0.05)

    def test_fit_weighted(self):
        d = p.DiagonalGaussian(self.D)
        X = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=np.float64)
        w = np.array([1.0, 0.0])  # concentrate all weight on row 0
        d.fit_weighted(X, w)
        np.testing.assert_allclose(d.means, X[0], atol=1e-9)

    def test_reset(self):
        d = p.DiagonalGaussian(self.D)
        d.set_parameters(np.ones(self.D) * 5, np.ones(self.D) * 2)
        d.reset()
        np.testing.assert_array_equal(d.means, np.zeros(self.D))
        np.testing.assert_array_equal(d.variances, np.ones(self.D))

    def test_sample_mv_shape(self):
        d = p.DiagonalGaussian(self.D)
        s = d.sample_mv()
        assert s.shape == (self.D,)
        assert np.all(np.isfinite(s))

    def test_repr_contains_dim(self):
        d = p.DiagonalGaussian(self.D)
        assert f"D={self.D}" in repr(d)


# ---------------------------------------------------------------------------
# FullCovGaussian
# ---------------------------------------------------------------------------

class TestFullCovGaussian:
    D = 2

    def test_construction(self):
        d = p.FullCovGaussian(self.D)
        assert d.dim == self.D

    def test_default_mean_zero(self):
        d = p.FullCovGaussian(self.D)
        np.testing.assert_array_equal(d.mean, np.zeros(self.D))

    def test_default_covariance_identity(self):
        d = p.FullCovGaussian(self.D)
        np.testing.assert_allclose(d.covariance, np.eye(self.D), atol=1e-5)

    def test_set_mean(self):
        d = p.FullCovGaussian(self.D)
        mu = np.array([1.0, -2.0])
        d.set_mean(mu)
        np.testing.assert_allclose(d.mean, mu)

    def test_set_covariance(self):
        d = p.FullCovGaussian(self.D)
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        d.set_covariance(cov)
        np.testing.assert_allclose(d.covariance[:, :2], cov, atol=1e-4)

    def test_set_parameters(self):
        d = p.FullCovGaussian(self.D)
        mu = np.array([2.0, 3.0])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        d.set_parameters(mu, cov)
        np.testing.assert_allclose(d.mean, mu)

    def test_log_pdf_single_finite(self):
        d = p.FullCovGaussian(self.D)
        x = np.zeros(self.D)
        lp = d.log_pdf(x)
        assert math.isfinite(lp)

    def test_log_pdf_batch(self):
        d = p.FullCovGaussian(self.D)
        X = np.zeros((4, self.D))
        lp = d.log_pdf(X)
        assert lp.shape == (4,)
        assert np.all(np.isfinite(lp))

    def test_fit_recovers_mean(self):
        d = p.FullCovGaussian(self.D)
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, self.D)) + np.array([1.0, -1.0])
        d.fit(X)
        np.testing.assert_allclose(d.mean, [1.0, -1.0], atol=0.3)

    def test_reset(self):
        d = p.FullCovGaussian(self.D)
        d.set_mean(np.array([5.0, 5.0]))
        d.reset()
        np.testing.assert_array_equal(d.mean, np.zeros(self.D))

    def test_sample_mv_shape(self):
        d = p.FullCovGaussian(self.D)
        s = d.sample_mv()
        assert s.shape == (self.D,)
        assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# IndependentComponents
# ---------------------------------------------------------------------------

class TestIndependentComponents:
    D = 2

    def test_construction_default(self):
        d = p.IndependentComponents(self.D)
        assert d.dim == self.D

    def test_construction_from_list(self):
        comps = [p.Gaussian(mu=1.0, sigma=0.5), p.Exponential(lam=2.0)]
        d = p.IndependentComponents(comps)
        assert d.dim == 2

    def test_get_component_type(self):
        d = p.IndependentComponents(self.D)
        c = d.get_component(0)
        # Default components are Gaussian
        assert isinstance(c, p.Gaussian)

    def test_set_component(self):
        d = p.IndependentComponents(self.D)
        d.set_component(0, p.Exponential(lam=2.0))
        c = d.get_component(0)
        assert isinstance(c, p.Exponential)

    def test_log_pdf_single_finite(self):
        d = p.IndependentComponents(self.D)
        x = np.zeros(self.D)
        lp = d.log_pdf(x)
        assert math.isfinite(lp)

    def test_log_pdf_batch(self):
        d = p.IndependentComponents(self.D)
        X = np.zeros((3, self.D))
        lp = d.log_pdf(X)
        assert lp.shape == (3,)
        assert np.all(np.isfinite(lp))

    def test_fit(self):
        d = p.IndependentComponents(self.D)
        X = np.ones((20, self.D)) + np.random.default_rng(1).standard_normal((20, self.D)) * 0.1
        d.fit(X)  # Should not raise

    def test_reset(self):
        d = p.IndependentComponents(self.D)
        d.reset()  # Should not raise


# ---------------------------------------------------------------------------
# HmmMV
# ---------------------------------------------------------------------------

class TestHmmMV:
    N = 2
    D = 2

    def test_construction(self):
        hmm = p.HmmMV(self.N)
        assert hmm.num_states == self.N

    def test_invalid_num_states(self):
        with pytest.raises(ValueError):
            p.HmmMV(0)

    def test_set_get_pi(self):
        hmm = p.HmmMV(self.N)
        pi = np.array([0.3, 0.7])
        hmm.set_pi(pi)
        np.testing.assert_allclose(hmm.get_pi(), pi)

    def test_set_get_trans(self):
        hmm = p.HmmMV(self.N)
        trans = np.array([[0.9, 0.1], [0.2, 0.8]])
        hmm.set_trans(trans)
        np.testing.assert_allclose(hmm.get_trans(), trans)

    def test_set_distribution(self):
        hmm = make_hmm_mv(self.N, self.D)
        d = hmm.get_distribution(0)
        assert isinstance(d, p.DiagonalGaussian)

    def test_train_does_not_raise(self):
        hmm = make_hmm_mv(self.N, self.D)
        seqs = make_sequences(n_seq=5, T=40, D=self.D)
        p.kmeans_init(hmm, seqs, seed=1)
        trainer = p.MVBaumWelchTrainer(hmm, seqs)
        for _ in range(5):
            trainer.train()

    def test_log_probability_finite(self):
        hmm = make_hmm_mv(self.N, self.D)
        seqs = make_sequences(n_seq=3, T=30, D=self.D)
        p.kmeans_init(hmm, seqs, seed=2)
        trainer = p.MVBaumWelchTrainer(hmm, seqs)
        for _ in range(3):
            trainer.train()
        calc = p.MVForwardBackwardCalculator(hmm, seqs[0])
        assert math.isfinite(calc.log_probability)

    def test_log_probability_monotone(self):
        """Baum-Welch EM must be monotone (LL non-decreasing)."""
        hmm = make_hmm_mv(self.N, self.D)
        seqs = make_sequences(n_seq=4, T=50, D=self.D, seed=10)
        p.kmeans_init(hmm, seqs, seed=3)

        def total_ll():
            return sum(
                p.MVForwardBackwardCalculator(hmm, s).log_probability
                for s in seqs
            )

        prev = total_ll()
        trainer = p.MVBaumWelchTrainer(hmm, seqs)
        for _ in range(10):
            trainer.train()
            cur = total_ll()
            assert cur >= prev - 1e-6, f"LL decreased: {prev:.4f} -> {cur:.4f}"
            prev = cur

    def test_full_cov_train(self):
        hmm = p.HmmMV(2)
        hmm.set_pi(np.array([0.5, 0.5]))
        hmm.set_trans(np.array([[0.9, 0.1], [0.1, 0.9]]))
        for s in range(2):
            hmm.set_distribution(s, p.FullCovGaussian(2))
        seqs = make_sequences(n_seq=4, T=30, D=2, seed=20)
        p.kmeans_init(hmm, seqs, seed=4)
        trainer = p.MVBaumWelchTrainer(hmm, seqs)
        for _ in range(5):
            trainer.train()

    def test_count_free_parameters(self):
        hmm = make_hmm_mv(self.N, self.D)
        k = p.count_free_parameters_mv(hmm)
        assert k > 0


# ---------------------------------------------------------------------------
# Kmeans init
# ---------------------------------------------------------------------------

class TestKmeansInit:
    def test_basic(self):
        hmm = make_hmm_mv(3, 2)
        seqs = make_sequences(n_seq=6, T=30, D=2, seed=99)
        p.kmeans_init(hmm, seqs, seed=0)  # Should not raise

    def test_seed_reproducible(self):
        D = 2
        seqs = make_sequences(n_seq=5, T=30, D=D, seed=7)
        hmm1 = make_hmm_mv(2, D)
        hmm2 = make_hmm_mv(2, D)
        p.kmeans_init(hmm1, seqs, seed=123)
        p.kmeans_init(hmm2, seqs, seed=123)
        np.testing.assert_allclose(
            hmm1.get_distribution(0).means,
            hmm2.get_distribution(0).means,
        )


# ---------------------------------------------------------------------------
# MV JSON round-trip
# ---------------------------------------------------------------------------

class TestMVIO:
    def _make_trained_hmm(self):
        hmm = make_hmm_mv(2, 2)
        seqs = make_sequences(n_seq=4, T=30, D=2, seed=55)
        p.kmeans_init(hmm, seqs, seed=5)
        trainer = p.MVBaumWelchTrainer(hmm, seqs)
        for _ in range(5):
            trainer.train()
        return hmm, seqs

    def test_to_from_json_mv(self):
        hmm, seqs = self._make_trained_hmm()
        json_str = p.to_json_mv(hmm)
        assert "multivariate" in json_str
        hmm2 = p.from_json_mv(json_str)
        # Recovered HMM should give same log-probability
        calc1 = p.MVForwardBackwardCalculator(hmm,  seqs[0])
        calc2 = p.MVForwardBackwardCalculator(hmm2, seqs[0])
        assert calc1.log_probability == pytest.approx(calc2.log_probability, rel=1e-6)

    def test_save_load_json_mv(self):
        hmm, seqs = self._make_trained_hmm()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            p.save_json_mv(hmm, path)
            assert path.stat().st_size > 0
            hmm2 = p.load_json_mv(path)
            calc1 = p.MVForwardBackwardCalculator(hmm,  seqs[0])
            calc2 = p.MVForwardBackwardCalculator(hmm2, seqs[0])
            assert calc1.log_probability == pytest.approx(calc2.log_probability, rel=1e-6)
        finally:
            path.unlink(missing_ok=True)

    def test_scalar_json_rejected_by_from_json_mv(self):
        scalar_hmm = p.Hmm(2)
        scalar_json = p.to_json(scalar_hmm)
        with pytest.raises(RuntimeError):
            p.from_json_mv(scalar_json)

    def test_independent_components_round_trip(self):
        hmm = p.HmmMV(2)
        hmm.set_pi(np.array([0.5, 0.5]))
        hmm.set_trans(np.array([[0.8, 0.2], [0.2, 0.8]]))
        # Use Gaussian components only: make_sequences generates standard-normal
        # data (includes negatives), incompatible with Gamma (support x > 0).
        for s in range(2):
            comps = [p.Gaussian(mu=float(s), sigma=1.0), p.Gaussian(mu=float(s) * 2, sigma=0.5)]
            hmm.set_distribution(s, p.IndependentComponents(comps))
        json_str = p.to_json_mv(hmm)
        hmm2 = p.from_json_mv(json_str)
        seqs = make_sequences(n_seq=2, T=10, D=2, seed=88)
        # Both should produce finite scores
        for s in seqs:
            lp = p.MVForwardBackwardCalculator(hmm2, s).log_probability
            assert math.isfinite(lp)
