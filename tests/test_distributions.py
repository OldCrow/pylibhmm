"""Tests for pylibhmm distribution bindings."""
import math

import numpy as np
import pytest

import pylibhmm


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------
class TestGaussian:
    def test_default_params(self):
        d = pylibhmm.Gaussian()
        assert d.mu == pytest.approx(0.0)
        assert d.sigma == pytest.approx(1.0)
        assert not d.is_discrete

    def test_custom_params(self):
        d = pylibhmm.Gaussian(mu=2.0, sigma=3.0)
        assert d.mu == pytest.approx(2.0)
        assert d.sigma == pytest.approx(3.0)

    def test_scalar_pdf(self):
        d = pylibhmm.Gaussian()
        assert d.pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), rel=1e-6)

    def test_batch_log_pdf(self):
        d = pylibhmm.Gaussian()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        result = d.log_pdf(x)
        assert result.shape == (5,)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result[0], result[4])  # symmetry
        np.testing.assert_allclose(result[1], result[3])

    def test_cdf(self):
        d = pylibhmm.Gaussian()
        assert d.cdf(0.0) == pytest.approx(0.5, rel=1e-6)
        assert d.cdf(1.0) > d.cdf(0.0)

    def test_fit(self):
        d = pylibhmm.Gaussian()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        d.fit(data)
        assert d.mu == pytest.approx(3.0, rel=1e-6)

    def test_fit_weighted(self):
        d = pylibhmm.Gaussian()
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        weights = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        d.fit_weighted(data, weights)
        assert d.mu == pytest.approx(1.0, rel=1e-4)

    def test_reset(self):
        d = pylibhmm.Gaussian(mu=5.0, sigma=2.0)
        d.reset()
        assert d.mu == pytest.approx(0.0)
        assert d.sigma == pytest.approx(1.0)

    def test_moments(self):
        d = pylibhmm.Gaussian(mu=3.0, sigma=2.0)
        assert d.mean == pytest.approx(3.0)
        assert d.variance == pytest.approx(4.0)
        assert d.std == pytest.approx(2.0)

    def test_invalid_sigma(self):
        with pytest.raises(Exception):
            pylibhmm.Gaussian(sigma=0.0)
        with pytest.raises(Exception):
            pylibhmm.Gaussian(sigma=-1.0)

    def test_sample_finite(self):
        d = pylibhmm.Gaussian(mu=3.0, sigma=1.0)
        x = d.sample()
        assert isinstance(x, float)
        assert math.isfinite(x)

    def test_sample_seeded_determinism(self):
        d = pylibhmm.Gaussian(mu=0.0, sigma=1.0)
        assert d.sample(42) == d.sample(42)
        assert d.sample(42) != d.sample(43)


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------
class TestPoisson:
    def test_params(self):
        d = pylibhmm.Poisson(lam=3.0)
        assert d.lam == pytest.approx(3.0)
        assert d.is_discrete

    def test_fit(self):
        d = pylibhmm.Poisson()
        data = np.array([2.0, 3.0, 3.0, 4.0, 3.0], dtype=np.float64)
        d.fit(data)
        assert d.lam == pytest.approx(3.0, rel=1e-6)

    def test_invalid(self):
        with pytest.raises(Exception):
            pylibhmm.Poisson(lam=0.0)
        with pytest.raises(Exception):
            pylibhmm.Poisson(lam=-1.0)

    def test_sample_is_nonneg_int(self):
        d = pylibhmm.Poisson(lam=5.0)
        x = d.sample(99)
        assert isinstance(x, float)
        assert x >= 0.0
        assert x == math.floor(x)  # integral value


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------
class TestExponential:
    def test_params(self):
        d = pylibhmm.Exponential(lam=2.0)
        assert d.lam == pytest.approx(2.0)
        assert not d.is_discrete

    def test_pdf_at_zero(self):
        d = pylibhmm.Exponential(lam=1.0)
        assert d.pdf(0.0) == pytest.approx(1.0, rel=1e-6)

    def test_cdf(self):
        d = pylibhmm.Exponential(lam=1.0)
        assert d.cdf(0.0) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------
class TestGamma:
    def test_params(self):
        d = pylibhmm.Gamma(k=2.0, theta=3.0)
        assert d.k == pytest.approx(2.0)
        assert d.theta == pytest.approx(3.0)

    def test_moments(self):
        d = pylibhmm.Gamma(k=2.0, theta=3.0)
        assert d.mean == pytest.approx(6.0)
        assert d.variance == pytest.approx(18.0)
        assert d.std == pytest.approx(math.sqrt(18.0), rel=1e-6)

    def test_cdf_at_zero(self):
        d = pylibhmm.Gamma(k=2.0, theta=1.0)
        assert d.cdf(0.0) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------
class TestBinomial:
    def test_params(self):
        d = pylibhmm.Binomial(n=10, p=0.3)
        assert d.n == 10
        assert d.p == pytest.approx(0.3)
        assert d.is_discrete


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------
class TestDiscrete:
    def test_fit_and_mode(self):
        d = pylibhmm.Discrete(3)
        data = np.array([0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float64)
        d.fit(data)
        assert d.mode == 1
        assert d.get_symbol_probability(1) > d.get_symbol_probability(0)

    def test_sample_valid_symbol(self):
        d = pylibhmm.Discrete(4)
        for i in range(4):
            d.set_probability(i, 0.25)
        x = d.sample(7)
        assert isinstance(x, float)
        assert x in {0.0, 1.0, 2.0, 3.0}

    def test_num_symbols_and_discrete(self):
        d = pylibhmm.Discrete(6)
        assert d.num_symbols == 6
        assert d.is_discrete


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------
class TestBeta:
    def test_params(self):
        d = pylibhmm.Beta(alpha=2.0, beta=5.0)
        assert d.alpha == pytest.approx(2.0)
        assert d.beta == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------
class TestUniform:
    def test_params_and_pdf(self):
        d = pylibhmm.Uniform(a=0.0, b=2.0)
        assert d.a == pytest.approx(0.0)
        assert d.b == pytest.approx(2.0)
        assert d.pdf(1.0) == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# Weibull
# ---------------------------------------------------------------------------
class TestWeibull:
    def test_params(self):
        d = pylibhmm.Weibull(k=2.0, lam=1.0)
        assert d.k == pytest.approx(2.0)
        assert d.lam == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rayleigh
# ---------------------------------------------------------------------------
class TestRayleigh:
    def test_params(self):
        d = pylibhmm.Rayleigh(sigma=2.0)
        assert d.sigma == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# LogNormal
# ---------------------------------------------------------------------------
class TestLogNormal:
    def test_params(self):
        d = pylibhmm.LogNormal(mu=0.0, sigma=1.0)
        assert d.mu == pytest.approx(0.0)
        assert d.sigma == pytest.approx(1.0)

    def test_distribution_mean(self):
        # E[X] = exp(mu + sigma^2 / 2)
        d = pylibhmm.LogNormal(mu=0.0, sigma=1.0)
        assert d.distribution_mean == pytest.approx(math.exp(0.5), rel=1e-6)


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------
class TestPareto:
    def test_params(self):
        d = pylibhmm.Pareto(k=2.0, xm=1.0)
        assert d.k == pytest.approx(2.0)
        assert d.xm == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NegativeBinomial
# ---------------------------------------------------------------------------
class TestNegativeBinomial:
    def test_params(self):
        d = pylibhmm.NegativeBinomial(r=5.0, p=0.4)
        assert d.r == pytest.approx(5.0)
        assert d.p == pytest.approx(0.4)
        assert d.is_discrete


# ---------------------------------------------------------------------------
# StudentT
# ---------------------------------------------------------------------------
class TestStudentT:
    def test_default_constructor(self):
        d = pylibhmm.StudentT()
        assert d.nu > 0

    def test_params(self):
        d = pylibhmm.StudentT(nu=3.0, location=1.0, scale=2.0)
        assert d.nu == pytest.approx(3.0)
        assert d.location == pytest.approx(1.0)
        assert d.scale == pytest.approx(2.0)

    def test_cdf_symmetry(self):
        d = pylibhmm.StudentT(nu=10.0)
        assert d.cdf(0.0) == pytest.approx(0.5, rel=1e-4)

    def test_invalid(self):
        with pytest.raises(Exception):
            pylibhmm.StudentT(nu=0.0)
        with pytest.raises(Exception):
            pylibhmm.StudentT(nu=1.0, location=0.0, scale=0.0)


# ---------------------------------------------------------------------------
# ChiSquared
# ---------------------------------------------------------------------------
class TestChiSquared:
    def test_params(self):
        d = pylibhmm.ChiSquared(k=4.0)
        assert d.k == pytest.approx(4.0)

    def test_moments(self):
        d = pylibhmm.ChiSquared(k=4.0)
        assert d.mean == pytest.approx(4.0)
        assert d.variance == pytest.approx(8.0)

    def test_cdf_at_zero(self):
        d = pylibhmm.ChiSquared(k=2.0)
        assert d.cdf(0.0) == pytest.approx(0.0, abs=1e-10)

    def test_invalid(self):
        with pytest.raises(Exception):
            pylibhmm.ChiSquared(k=0.0)
        with pytest.raises(Exception):
            pylibhmm.ChiSquared(k=-1.0)


# ---------------------------------------------------------------------------
# VonMises
# ---------------------------------------------------------------------------
class TestVonMises:
    def test_default_params(self):
        d = pylibhmm.VonMises()
        assert d.mu == pytest.approx(0.0)
        assert d.kappa == pytest.approx(1.0)
        assert not d.is_discrete

    def test_custom_params(self):
        d = pylibhmm.VonMises(mu=1.5, kappa=2.0)
        assert d.mu == pytest.approx(1.5)
        assert d.kappa == pytest.approx(2.0)

    def test_pdf_at_mean(self):
        # PDF is maximised at x == mu
        import math
        d = pylibhmm.VonMises(mu=0.0, kappa=2.0)
        peak = d.pdf(0.0)
        assert peak > d.pdf(1.0)
        assert peak > d.pdf(-1.0)

    def test_batch_log_pdf(self):
        d = pylibhmm.VonMises(mu=0.0, kappa=1.0)
        x = np.linspace(-math.pi, math.pi, 9, dtype=np.float64)
        result = d.log_pdf(x)
        assert result.shape == (9,)
        assert np.isfinite(result).all()
        # symmetry around mu=0
        np.testing.assert_allclose(result[0], result[-1], rtol=1e-6)

    def test_cdf_at_mean(self):
        # CDF(mu) should be 0.5 by symmetry
        d = pylibhmm.VonMises(mu=0.0, kappa=1.0)
        assert d.cdf(0.0) == pytest.approx(0.5, abs=1e-3)

    def test_mean_and_circular_variance(self):
        d = pylibhmm.VonMises(mu=1.0, kappa=3.0)
        assert d.mean == pytest.approx(1.0)
        # circular_variance in (0, 1); decreases as kappa increases
        cv = d.circular_variance
        assert 0.0 < cv < 1.0
        # kappa=0 -> uniform -> circular_variance=1
        d0 = pylibhmm.VonMises(mu=0.0, kappa=0.0)
        assert d0.circular_variance == pytest.approx(1.0, abs=1e-6)

    def test_variance_property(self):
        # variance is an alias for circular_variance
        d = pylibhmm.VonMises(mu=0.0, kappa=2.0)
        assert d.variance == pytest.approx(d.circular_variance)

    def test_fit_unweighted(self):
        import math
        rng = np.random.default_rng(42)
        # Concentrated around pi/2
        angles = rng.vonmises(math.pi / 2, 3.0, size=500)
        d = pylibhmm.VonMises()
        d.fit(angles.astype(np.float64))
        # mu should be near pi/2
        assert d.mu == pytest.approx(math.pi / 2, abs=0.15)
        assert d.kappa > 0.5

    def test_fit_weighted(self):
        import math
        data = np.array([0.0, math.pi, math.pi], dtype=np.float64)
        weights = np.array([0.01, 1.0, 1.0], dtype=np.float64)
        d = pylibhmm.VonMises()
        d.fit_weighted(data, weights)
        # mu should be near pi
        assert abs(d.mu) == pytest.approx(math.pi, abs=0.3)

    def test_reset(self):
        d = pylibhmm.VonMises(mu=1.0, kappa=5.0)
        d.reset()
        assert d.mu == pytest.approx(0.0)
        assert d.kappa == pytest.approx(1.0)

    def test_invalid_kappa(self):
        with pytest.raises(Exception):
            pylibhmm.VonMises(kappa=-1.0)

    def test_repr(self):
        d = pylibhmm.VonMises(mu=0.0, kappa=1.0)
        assert "Von Mises" in repr(d) or "VonMises" in repr(d)
