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
