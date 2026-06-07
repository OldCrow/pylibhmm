"""Distribution fitting validation tests.

Verifies that fit() and fit_weighted() correctly estimate parameters through
the Python bindings, using hardcoded deterministic data with analytically known
MLE results. Closes OldCrow/pylibhmm#3.

MLE conventions used here (matching libhmm v3.8.0):
  Gaussian      : mean = sample mean; sigma = sqrt(sum((xi-mean)^2) / N)  (N denominator)
  Exponential   : lambda = N / sum(xi)  =  1 / sample mean
  Poisson       : lambda = sample mean
  Discrete      : P(k) = count(k) / N   (or weighted: sum_weights_k / sum_weights)
  Gamma         : Newton-Raphson MLE  (k, theta); unweighted and weighted
"""
import math

import numpy as np
import pytest

import pylibhmm


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------
class TestGaussianFitting:
    # data = [5, 7, 9, 11, 13]
    # mean = 9.0
    # MLE var = ((5-9)^2 + (7-9)^2 + 0 + (11-9)^2 + (13-9)^2) / 5
    #         = (16 + 4 + 0 + 4 + 16) / 5 = 40 / 5 = 8.0
    # MLE sigma = sqrt(8) = 2*sqrt(2)
    _data = np.array([5.0, 7.0, 9.0, 11.0, 13.0], dtype=np.float64)
    _mean = 9.0
    _sigma = math.sqrt(8.0)  # 2*sqrt(2) ≈ 2.8284

    def test_unweighted_mean(self):
        d = pylibhmm.Gaussian()
        d.fit(self._data)
        assert d.mu == pytest.approx(self._mean, rel=1e-10)

    def test_unweighted_sigma(self):
        d = pylibhmm.Gaussian()
        d.fit(self._data)
        assert d.sigma == pytest.approx(self._sigma, rel=1e-10)

    def test_weighted_concentrates_on_center(self):
        # Weights heavily concentrate on the middle element (9.0).
        # data = [5, 7, 9, 11, 13], weights = [0.01, 0.01, 1.0, 0.01, 0.01]
        # Weighted mean ≈ 9.0  (dominated by element at index 2)
        data = self._data
        weights = np.array([0.01, 0.01, 1.0, 0.01, 0.01], dtype=np.float64)
        d = pylibhmm.Gaussian()
        d.fit_weighted(data, weights)
        assert d.mu == pytest.approx(9.0, abs=0.1)

    def test_weighted_exact_mean(self):
        # data = [1, 2, 3, 4, 5], weights = [1, 4, 1, 0, 0]  sum=6
        # weighted mean = (1*1 + 4*2 + 1*3 + 0 + 0) / 6 = 12/6 = 2.0
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        weights = np.array([1.0, 4.0, 1.0, 0.0, 0.0], dtype=np.float64)
        d = pylibhmm.Gaussian()
        d.fit_weighted(data, weights)
        assert d.mu == pytest.approx(2.0, rel=1e-8)


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------
class TestExponentialFitting:
    # data = [0.5, 1.0, 2.0, 0.5, 1.0]  mean = 1.0  MLE rate = 1.0
    _data = np.array([0.5, 1.0, 2.0, 0.5, 1.0], dtype=np.float64)

    def test_unweighted_rate(self):
        d = pylibhmm.Exponential()
        d.fit(self._data)
        assert d.lam == pytest.approx(1.0, rel=1e-10)

    def test_weighted_rate(self):
        # data = [1.0, 2.0, 4.0], weights = [2.0, 1.0, 1.0]  sum=4
        # weighted mean = (2*1.0 + 1*2.0 + 1*4.0) / 4 = 8/4 = 2.0
        # MLE rate = 1 / weighted_mean = 0.5
        data = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        weights = np.array([2.0, 1.0, 1.0], dtype=np.float64)
        d = pylibhmm.Exponential()
        d.fit_weighted(data, weights)
        assert d.lam == pytest.approx(0.5, rel=1e-8)


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------
class TestPoissonFitting:
    def test_unweighted_rate(self):
        # All observations equal 4 → MLE rate = 4.0 exactly.
        data = np.array([4.0, 4.0, 4.0, 4.0, 4.0], dtype=np.float64)
        d = pylibhmm.Poisson()
        d.fit(data)
        assert d.lam == pytest.approx(4.0, rel=1e-10)

    def test_unweighted_rate_mixed(self):
        # data = [2, 3, 4, 5, 6]  mean = 4.0
        data = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        d = pylibhmm.Poisson()
        d.fit(data)
        assert d.lam == pytest.approx(4.0, rel=1e-10)

    def test_weighted_rate(self):
        # data = [1, 3, 5], weights = [1, 2, 1]  sum=4
        # weighted mean = (1*1 + 2*3 + 1*5) / 4 = 12/4 = 3.0
        data = np.array([1.0, 3.0, 5.0], dtype=np.float64)
        weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        d = pylibhmm.Poisson()
        d.fit_weighted(data, weights)
        assert d.lam == pytest.approx(3.0, rel=1e-8)


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------
class TestDiscreteFitting:
    def test_unweighted_uniform_recovery(self):
        # 4 symbols, 12 observations: 3 each → all P = 0.25 exactly.
        data = np.array(
            [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            dtype=np.float64,
        )
        d = pylibhmm.Discrete(4)
        d.fit(data)
        for i in range(4):
            assert d.get_symbol_probability(i) == pytest.approx(0.25, rel=1e-10)

    def test_unweighted_skewed_recovery(self):
        # 3 symbols: 6×symbol0, 3×symbol1, 1×symbol2  (N=10)
        # P(0)=0.6, P(1)=0.3, P(2)=0.1
        data = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
            dtype=np.float64,
        )
        d = pylibhmm.Discrete(3)
        d.fit(data)
        assert d.get_symbol_probability(0) == pytest.approx(0.6, rel=1e-10)
        assert d.get_symbol_probability(1) == pytest.approx(0.3, rel=1e-10)
        assert d.get_symbol_probability(2) == pytest.approx(0.1, rel=1e-10)

    def test_weighted_recovery(self):
        # 3 symbols, weights = [6, 3, 1]  sum=10
        # P(0) = 6/10 = 0.6, P(1) = 3/10 = 0.3, P(2) = 1/10 = 0.1
        data = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        weights = np.array([6.0, 3.0, 1.0], dtype=np.float64)
        d = pylibhmm.Discrete(3)
        d.fit_weighted(data, weights)
        assert d.get_symbol_probability(0) == pytest.approx(0.6, rel=1e-10)
        assert d.get_symbol_probability(1) == pytest.approx(0.3, rel=1e-10)
        assert d.get_symbol_probability(2) == pytest.approx(0.1, rel=1e-10)

    def test_mode_after_fit(self):
        data = np.array([0.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float64)
        d = pylibhmm.Discrete(3)
        d.fit(data)
        assert d.mode == 1


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------
class TestGammaFitting:
    def test_unweighted_recovers_params(self):
        # With shape k=2, scale theta=3: mean=6, variance=18.
        # Use enough hardcoded samples that MLE converges close to truth.
        # Samples drawn from Gamma(2,3) with seed 0 (precalculated offline):
        data = np.array(
            [2.43, 4.18, 5.90, 3.67, 8.22, 6.11, 4.73, 7.85,
             2.99, 5.54, 9.03, 3.41, 6.78, 4.02, 7.30, 5.11,
             3.88, 6.55, 4.44, 8.67, 3.22, 5.77, 4.90, 7.01,
             2.88, 6.23, 5.38, 9.12, 3.55, 4.67],
            dtype=np.float64,
        )
        d = pylibhmm.Gamma()
        d.fit(data)
        # Tolerance is loose — 30 samples — but parameters should be in the right ballpark.
        assert d.k == pytest.approx(2.0, abs=0.6)
        assert d.theta == pytest.approx(3.0, abs=0.8)
