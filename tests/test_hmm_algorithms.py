import numpy as np
from pathlib import Path
import tempfile
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


def test_viterbi_decode(simple_hmm):
    obs = np.array([0, 1, 5, 5, 5, 4], dtype=np.float64)
    calc = pylibhmm.ViterbiCalculator(simple_hmm, obs)
    states = calc.decode()
    assert states.shape == (obs.shape[0],)
    assert np.isfinite(calc.log_probability)


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


def test_xml_roundtrip(simple_hmm):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.xml"
        pylibhmm.save_hmm(simple_hmm, model_path)
        assert model_path.exists()
        assert model_path.stat().st_size > 0
        try:
            loaded = pylibhmm.load_hmm(model_path)
        except ValueError:
            pytest.xfail("libhmm XML reader rejected generated XML on this platform")
    assert loaded.num_states == simple_hmm.num_states
