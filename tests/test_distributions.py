import numpy as np
import pytest

import pylibhmm


def test_gaussian_scalar_and_batch_log_pdf():
    dist = pylibhmm.Gaussian(mu=0.0, sigma=1.0)
    assert dist.pdf(0.0) == pytest.approx(0.3989, rel=1e-3)

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    log_pdf = dist.log_pdf(x)
    assert log_pdf.shape == (5,)
    assert np.isfinite(log_pdf).all()


def test_discrete_distribution_fit_and_mode():
    dist = pylibhmm.Discrete(3)
    data = np.array([0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float64)
    dist.fit(data)

    assert dist.mode == 1
    assert dist.get_symbol_probability(1) > dist.get_symbol_probability(0)


def test_distribution_validation_errors():
    with pytest.raises(Exception):
        pylibhmm.Gaussian(mu=0.0, sigma=0.0)
    with pytest.raises(Exception):
        pylibhmm.Poisson(lam=-1.0)
