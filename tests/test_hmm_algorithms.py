"""Tests for HMM algorithm, calculator, and trainer bindings."""
from pathlib import Path
import tempfile

import numpy as np
import pytest

import pylibhmm


def test_forward_backward(simple_hmm):
    obs = np.array([0, 1, 5, 4, 2], dtype=np.float64)
    calc = pylibhmm.ForwardBackwardCalculator(simple_hmm, obs)

    assert np.isfinite(calc.log_probability)
    assert 0.0 < calc.probability <= 1.0

    alpha = calc.get_log_forward_variables()
    beta = calc.get_log_backward_variables()
    assert alpha.shape == (obs.shape[0], 2)
    assert beta.shape == (obs.shape[0], 2)


def test_forward_backward_recompute(simple_hmm):
    obs1 = np.array([0, 1, 5], dtype=np.float64)
    obs2 = np.array([5, 4, 3, 2], dtype=np.float64)
    calc = pylibhmm.ForwardBackwardCalculator(simple_hmm, obs1)
    lp1 = calc.log_probability
    calc.compute(obs2)
    lp2 = calc.log_probability
    # Different length sequences must produce different log-probabilities.
    assert lp1 != pytest.approx(lp2)


def test_viterbi_decode(simple_hmm):
    obs = np.array([0, 1, 5, 5, 5, 4], dtype=np.float64)
    calc = pylibhmm.ViterbiCalculator(simple_hmm, obs)
    states = calc.decode()
    assert states.shape == (obs.shape[0],)
    assert np.isfinite(calc.log_probability)
    assert set(states.tolist()).issubset({0, 1})


def test_baum_welch_training_runs(simple_hmm):
    sequences = [
        np.array([0, 1, 5, 4, 3, 5], dtype=np.float64),
        np.array([5, 5, 4, 5, 0, 1], dtype=np.float64),
    ]
    trainer = pylibhmm.BaumWelchTrainer(simple_hmm, sequences)
    trainer.train()
    simple_hmm.validate()


def test_baum_welch_last_log_probability(simple_hmm):
    sequences = [
        np.array([0, 1, 5, 4, 3, 5], dtype=np.float64),
        np.array([5, 5, 4, 5, 0, 1], dtype=np.float64),
    ]
    trainer = pylibhmm.BaumWelchTrainer(simple_hmm, sequences)
    assert not np.isfinite(trainer.last_log_probability)
    trainer.train()
    assert np.isfinite(trainer.last_log_probability)
    assert trainer.last_log_probability < 0.0


def test_viterbi_trainer_config(simple_hmm):
    cfg = pylibhmm.training_preset_fast()
    trainer = pylibhmm.ViterbiTrainer(
        simple_hmm,
        [np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)],
        cfg,
    )
    trainer.train()
    assert trainer.has_converged or trainer.reached_max_iterations


def test_viterbi_trainer_convergence(simple_hmm):
    cfg = pylibhmm.training_preset_precise()
    sequences = [np.array([i % 6 for i in range(30)], dtype=np.float64)]
    trainer = pylibhmm.ViterbiTrainer(simple_hmm, sequences, cfg)
    trainer.train()
    assert np.isfinite(trainer.last_log_probability)


def test_segmental_kmeans_training_runs(simple_hmm):
    sequences = [
        np.array([0, 1, 5, 4, 3, 5], dtype=np.float64),
        np.array([5, 5, 4, 5, 0, 1], dtype=np.float64),
    ]
    trainer = pylibhmm.SegmentalKMeansTrainer(simple_hmm, sequences)
    trainer.train()
    assert isinstance(trainer.is_terminated, bool)
    simple_hmm.validate()


def test_json_roundtrip(simple_hmm):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.json"
        pylibhmm.save_json(simple_hmm, model_path)
        assert model_path.exists()
        assert model_path.stat().st_size > 0
        loaded = pylibhmm.load_json(model_path)
    assert loaded.num_states == simple_hmm.num_states


def test_json_roundtrip_preserves_pi(simple_hmm):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.json"
        pylibhmm.save_json(simple_hmm, path)
        loaded = pylibhmm.load_json(path)
    np.testing.assert_allclose(loaded.get_pi(), simple_hmm.get_pi(), rtol=1e-10)


def test_to_from_json_roundtrip(simple_hmm):
    json_str = pylibhmm.to_json(simple_hmm)
    assert isinstance(json_str, str)
    assert "states" in json_str
    loaded = pylibhmm.from_json(json_str)
    assert loaded.num_states == simple_hmm.num_states
    np.testing.assert_allclose(loaded.get_pi(), simple_hmm.get_pi(), rtol=1e-10)


def test_xml_roundtrip(simple_hmm):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.xml"
        pylibhmm.save_hmm(simple_hmm, model_path)
        assert model_path.exists()
        assert model_path.stat().st_size > 0
        loaded = pylibhmm.load_hmm(model_path)
    assert loaded.num_states == simple_hmm.num_states


def test_xml_roundtrip_preserves_pi(simple_hmm):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.xml"
        pylibhmm.save_hmm(simple_hmm, path)
        loaded = pylibhmm.load_hmm(path)
    np.testing.assert_allclose(loaded.get_pi(), simple_hmm.get_pi(), rtol=1e-6)


def test_training_config_fields():
    cfg = pylibhmm.TrainingConfig()
    cfg.max_iterations = 100
    cfg.convergence_tolerance = 1e-4
    cfg.convergence_window = 5
    cfg.enable_progress_reporting = False
    assert cfg.max_iterations == 100
    assert cfg.convergence_tolerance == pytest.approx(1e-4)
    assert cfg.convergence_window == 5
    assert not cfg.enable_progress_reporting


def test_empty_sequences_raises(simple_hmm):
    with pytest.raises(Exception):
        pylibhmm.BaumWelchTrainer(simple_hmm, [])


# ---------------------------------------------------------------------------
# decode_posterior
# ---------------------------------------------------------------------------
def test_decode_posterior_shape(simple_hmm):
    obs = np.array([0, 1, 5, 4, 2, 3], dtype=np.float64)
    calc = pylibhmm.ForwardBackwardCalculator(simple_hmm, obs)
    posterior = calc.decode_posterior()
    assert posterior.shape == (obs.shape[0],)
    assert posterior.dtype == np.int64


def test_decode_posterior_valid_states(simple_hmm):
    obs = np.array([0, 1, 5, 5, 5, 0, 1, 2], dtype=np.float64)
    calc = pylibhmm.ForwardBackwardCalculator(simple_hmm, obs)
    posterior = calc.decode_posterior()
    assert set(posterior.tolist()).issubset({0, 1})


def test_decode_posterior_vs_viterbi_same_length(simple_hmm):
    obs = np.array([0, 1, 5, 4, 5, 0], dtype=np.float64)
    fb = pylibhmm.ForwardBackwardCalculator(simple_hmm, obs)
    vt = pylibhmm.ViterbiCalculator(simple_hmm, obs)
    posterior = fb.decode_posterior()
    viterbi = vt.decode()
    assert posterior.shape == viterbi.shape


# ---------------------------------------------------------------------------
# MapBaumWelchTrainer
# ---------------------------------------------------------------------------
def test_map_baum_welch_runs(simple_hmm):
    sequences = [
        np.array([0, 1, 5, 4, 3, 5], dtype=np.float64),
        np.array([5, 5, 4, 5, 0, 1], dtype=np.float64),
    ]
    trainer = pylibhmm.MapBaumWelchTrainer(simple_hmm, sequences, pseudo_count=1.0)
    trainer.train()
    simple_hmm.validate()


def test_map_baum_welch_default_pseudo_count(simple_hmm):
    sequences = [np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)]
    trainer = pylibhmm.MapBaumWelchTrainer(simple_hmm, sequences)
    assert trainer.pseudo_count == pytest.approx(1.0)


def test_map_baum_welch_pseudo_count_setter(simple_hmm):
    sequences = [np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)]
    trainer = pylibhmm.MapBaumWelchTrainer(simple_hmm, sequences, pseudo_count=2.0)
    assert trainer.pseudo_count == pytest.approx(2.0)
    trainer.pseudo_count = 0.5
    assert trainer.pseudo_count == pytest.approx(0.5)


def test_map_baum_welch_compute_log_prior(simple_hmm):
    sequences = [np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)]
    trainer = pylibhmm.MapBaumWelchTrainer(simple_hmm, sequences, pseudo_count=1.0)
    lp = trainer.compute_log_prior()
    assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# P-4: from_json / load_json must return the Python Hmm subclass
# ---------------------------------------------------------------------------

def test_from_json_returns_python_subclass(simple_hmm):
    json_str = pylibhmm.to_json(simple_hmm)
    result = pylibhmm.from_json(json_str)
    assert isinstance(result, pylibhmm.Hmm)
    assert type(result) is pylibhmm.Hmm


def test_load_json_returns_python_subclass(simple_hmm):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.json"
        pylibhmm.save_json(simple_hmm, path)
        result = pylibhmm.load_json(path)
    assert isinstance(result, pylibhmm.Hmm)
    assert type(result) is pylibhmm.Hmm


def test_from_json_subclass_can_set_pi(simple_hmm):
    """Deserialized Hmm must have Python-validated set_pi available."""
    json_str = pylibhmm.to_json(simple_hmm)
    result = pylibhmm.from_json(json_str)
    pi = result.get_pi()
    # Re-setting the same pi via the Python wrapper should not raise.
    result.set_pi(pi)
    np.testing.assert_allclose(result.get_pi(), pi)


def test_map_baum_welch_zero_pseudo_count_matches_mle(simple_hmm):
    # c=0 must not raise; it recovers standard Baum-Welch
    sequences = [
        np.array([0, 1, 5, 4, 3, 5], dtype=np.float64),
        np.array([5, 5, 4, 5, 0, 1], dtype=np.float64),
    ]
    trainer = pylibhmm.MapBaumWelchTrainer(simple_hmm, sequences, pseudo_count=0.0)
    trainer.train()
    simple_hmm.validate()


def test_map_baum_welch_empty_sequences_raises(simple_hmm):
    with pytest.raises(Exception):
        pylibhmm.MapBaumWelchTrainer(simple_hmm, [])


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def test_count_free_parameters(simple_hmm):
    # 2-state discrete-emission HMM:
    # transitions: 2*(2-1)=2, initial: (2-1)=1
    # each Discrete(6) has 5 free params (6-1 probabilities) x 2 states = 10
    # total: 2 + 1 + 10 = 13
    k = pylibhmm.count_free_parameters(simple_hmm)
    assert k == 13


def test_compute_aic():
    logL = -100.0
    k = 5
    aic = pylibhmm.compute_aic(logL, k)
    assert aic == pytest.approx(2 * k - 2 * logL)


def test_compute_bic():
    logL = -100.0
    k = 5
    n = 200
    import math
    bic = pylibhmm.compute_bic(logL, k, n)
    assert bic == pytest.approx(k * math.log(n) - 2 * logL)


def test_compute_aicc():
    logL = -100.0
    k = 5
    n = 200
    aic = pylibhmm.compute_aic(logL, k)
    aicc = pylibhmm.compute_aicc(logL, k, n)
    # AICc = AIC + 2k(k+1)/(n-k-1)
    correction = 2 * k * (k + 1) / (n - k - 1)
    assert aicc == pytest.approx(aic + correction)


def test_evaluate_model_returns_model_criteria(simple_hmm):
    # k=13 free params; need n > k+1=14 for AICc to be finite.
    obs = np.array([0, 1, 5, 4, 2, 3] * 10, dtype=np.float64)  # n=60
    calc = pylibhmm.ForwardBackwardCalculator(simple_hmm, obs)
    mc = pylibhmm.evaluate_model(simple_hmm, calc.log_probability, len(obs))
    assert isinstance(mc, pylibhmm.ModelCriteria)
    assert np.isfinite(mc.aic)
    assert np.isfinite(mc.bic)
    assert np.isfinite(mc.aicc)


def test_evaluate_model_ordering(simple_hmm):
    # For n >> k, AICc ≈ AIC < BIC (BIC penalises more heavily for large n)
    obs = np.array([0, 1, 5, 4, 2, 3] * 100, dtype=np.float64)
    calc = pylibhmm.ForwardBackwardCalculator(simple_hmm, obs)
    mc = pylibhmm.evaluate_model(simple_hmm, calc.log_probability, len(obs))
    # AIC < BIC for large n
    assert mc.aic < mc.bic
    # AICc converges toward AIC as n grows
    assert abs(mc.aicc - mc.aic) < abs(mc.bic - mc.aic)
