"""Tests for the libhmm v4.4.0 model-level bindings.

Covers clone_hmm / clone_hmm_mv, sample / sample_mv, fit_best_of_n /
fit_best_of_n_mv, the HmmTopology constraints, and the validateInitialized
error message surfaced through existing bindings (issues #25 / #26).
"""

import numpy as np
import pytest

import pylibhmm


def _mv_hmm(dim: int = 2) -> pylibhmm.HmmMV:
    hmm = pylibhmm.HmmMV(2)
    hmm.set_pi(np.array([0.5, 0.5]))
    hmm.set_trans(np.array([[0.9, 0.1], [0.1, 0.9]]))
    lo = pylibhmm.DiagonalGaussian(dim)
    lo.set_parameters(np.zeros(dim), np.ones(dim))
    hi = pylibhmm.DiagonalGaussian(dim)
    hi.set_parameters(np.full(dim, 5.0), np.ones(dim))
    hmm.set_distribution(0, lo)
    hmm.set_distribution(1, hi)
    return hmm


# ---------------------------------------------------------------------------
# clone_hmm
# ---------------------------------------------------------------------------
class TestCloneHmm:
    def test_returns_wrapper_type(self, simple_hmm):
        clone = pylibhmm.clone_hmm(simple_hmm)
        assert isinstance(clone, pylibhmm.Hmm)
        assert clone.num_states == simple_hmm.num_states

    def test_copies_parameters(self, simple_hmm):
        clone = pylibhmm.clone_hmm(simple_hmm)
        np.testing.assert_allclose(clone.get_pi(), simple_hmm.get_pi())
        np.testing.assert_allclose(clone.get_trans(), simple_hmm.get_trans())

    def test_clone_is_independent(self, simple_hmm):
        clone = pylibhmm.clone_hmm(simple_hmm)
        original_pi = simple_hmm.get_pi().copy()
        clone.set_pi(np.array([0.5, 0.5]))
        np.testing.assert_allclose(simple_hmm.get_pi(), original_pi)

    def test_distributions_are_deep_copied(self, simple_hmm):
        clone = pylibhmm.clone_hmm(simple_hmm)
        d = clone.get_distribution(0)
        d.set_probability(0, 0.9)
        # The original's state-0 distribution must be untouched.
        assert simple_hmm.get_distribution(0).pdf(0.0) == pytest.approx(1.0 / 6.0)

    def test_mv_clone(self):
        hmm = _mv_hmm()
        clone = pylibhmm.clone_hmm_mv(hmm)
        assert isinstance(clone, pylibhmm.HmmMV)
        np.testing.assert_allclose(clone.get_trans(), hmm.get_trans())

    def test_mv_clone_preserves_unset_slots(self):
        hmm = pylibhmm.HmmMV(2)
        hmm.set_pi(np.array([0.5, 0.5]))
        hmm.set_trans(np.array([[0.5, 0.5], [0.5, 0.5]]))
        clone = pylibhmm.clone_hmm_mv(hmm)
        with pytest.raises(RuntimeError):
            clone.get_distribution(0)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------
class TestSample:
    def test_shapes_and_dtypes(self, simple_hmm):
        obs, states = pylibhmm.sample(simple_hmm, 25, seed=7)
        assert obs.shape == (25,)
        assert states.shape == (25,)
        assert obs.dtype == np.float64
        assert states.dtype == np.int64

    def test_states_in_range(self, simple_hmm):
        _, states = pylibhmm.sample(simple_hmm, 100, seed=1)
        assert states.min() >= 0
        assert states.max() < simple_hmm.num_states

    def test_seed_reproducible(self, simple_hmm):
        obs1, states1 = pylibhmm.sample(simple_hmm, 50, seed=42)
        obs2, states2 = pylibhmm.sample(simple_hmm, 50, seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(states1, states2)

    def test_unseeded_draws(self, simple_hmm):
        obs, states = pylibhmm.sample(simple_hmm, 10)
        assert obs.shape == (10,)
        assert np.all(np.isfinite(obs))

    def test_zero_length(self, simple_hmm):
        obs, states = pylibhmm.sample(simple_hmm, 0, seed=1)
        assert obs.shape == (0,)
        assert states.shape == (0,)

    def test_negative_length_raises(self, simple_hmm):
        with pytest.raises(ValueError):
            pylibhmm.sample(simple_hmm, -1, seed=1)

    def test_uninitialised_raises(self):
        # A fresh Hmm zero-initialises pi; sampling must not fail silently.
        with pytest.raises(RuntimeError, match="sums to zero"):
            pylibhmm.sample(pylibhmm.Hmm(2), 5, seed=1)

    def test_mv_shapes(self):
        hmm = _mv_hmm(dim=3)
        obs, states = pylibhmm.sample_mv(hmm, 20, seed=5)
        assert obs.shape == (20, 3)
        assert states.shape == (20,)
        assert obs.dtype == np.float64
        assert states.dtype == np.int64

    def test_mv_seed_reproducible(self):
        hmm = _mv_hmm()
        obs1, _ = pylibhmm.sample_mv(hmm, 30, seed=9)
        obs2, _ = pylibhmm.sample_mv(hmm, 30, seed=9)
        np.testing.assert_array_equal(obs1, obs2)

    def test_mv_zero_length(self):
        obs, states = pylibhmm.sample_mv(_mv_hmm(), 0, seed=1)
        assert obs.shape[0] == 0
        assert states.shape == (0,)

    def test_round_trip_with_training(self, simple_hmm):
        # Sequences sampled from the model must be trainable on it.
        seqs = [pylibhmm.sample(simple_hmm, 60, seed=s)[0] for s in range(3)]
        trainer = pylibhmm.BaumWelchTrainer(simple_hmm, seqs)
        trainer.train()
        assert np.isfinite(trainer.last_log_probability)


# ---------------------------------------------------------------------------
# fit_best_of_n
# ---------------------------------------------------------------------------
class TestFitBestOfN:
    def setup_method(self):
        self.hmm = pylibhmm.Hmm(2)
        self.hmm.set_pi(np.array([0.5, 0.5]))
        self.hmm.set_trans(np.array([[0.9, 0.1], [0.1, 0.9]]))
        self.hmm.set_distribution(0, pylibhmm.Gaussian(mu=-1.0, sigma=1.0))
        self.hmm.set_distribution(1, pylibhmm.Gaussian(mu=1.0, sigma=1.0))
        rng = np.random.default_rng(0)
        self.seqs = [
            np.concatenate([rng.normal(0, 1, 40), rng.normal(5, 1, 40)]),
            np.concatenate([rng.normal(5, 1, 40), rng.normal(0, 1, 40)]),
        ]

    def test_returns_finite_log_likelihood(self):
        logl = pylibhmm.fit_best_of_n(self.hmm, self.seqs, n_restarts=3, seed=1, max_iters=100)
        assert np.isfinite(logl)

    def test_at_least_as_good_as_single_run(self):
        # Restart 0 is the unrandomised single run, so the best-of-n result
        # is by construction >= a single training run from the same start.
        single = pylibhmm.clone_hmm(self.hmm)
        trainer = pylibhmm.BaumWelchTrainer(single, self.seqs)
        for _ in range(100):
            trainer.train()
        single_total = sum(
            pylibhmm.ForwardBackwardCalculator(single, s).log_probability for s in self.seqs
        )
        best = pylibhmm.fit_best_of_n(self.hmm, self.seqs, n_restarts=4, seed=3, max_iters=100)
        assert best >= single_total - 1e-6

    def test_seed_reproducible(self):
        h1 = pylibhmm.clone_hmm(self.hmm)
        h2 = pylibhmm.clone_hmm(self.hmm)
        l1 = pylibhmm.fit_best_of_n(h1, self.seqs, n_restarts=3, seed=11, max_iters=50)
        l2 = pylibhmm.fit_best_of_n(h2, self.seqs, n_restarts=3, seed=11, max_iters=50)
        assert l1 == pytest.approx(l2)
        np.testing.assert_allclose(h1.get_trans(), h2.get_trans())

    def test_mutates_in_place(self):
        before = self.hmm.get_trans().copy()
        pylibhmm.fit_best_of_n(self.hmm, self.seqs, n_restarts=2, seed=1, max_iters=100)
        assert not np.allclose(self.hmm.get_trans(), before)

    def test_zero_restarts_raises(self):
        with pytest.raises(ValueError):
            pylibhmm.fit_best_of_n(self.hmm, self.seqs, n_restarts=0)

    def test_empty_sequences_raises(self):
        with pytest.raises(ValueError):
            pylibhmm.fit_best_of_n(self.hmm, [], n_restarts=2)

    def test_mv(self):
        hmm = _mv_hmm()
        rng = np.random.default_rng(1)
        seqs = [
            np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(5, 1, (30, 2))]),
            np.vstack([rng.normal(5, 1, (30, 2)), rng.normal(0, 1, (30, 2))]),
        ]
        logl = pylibhmm.fit_best_of_n_mv(hmm, seqs, n_restarts=2, seed=1, max_iters=50)
        assert np.isfinite(logl)

    def test_mv_zero_restarts_raises(self):
        with pytest.raises(ValueError):
            pylibhmm.fit_best_of_n_mv(_mv_hmm(), [np.zeros((5, 2))], n_restarts=0)


# ---------------------------------------------------------------------------
# Topology constraints
# ---------------------------------------------------------------------------
def _valid_mask(topology: "pylibhmm.HmmTopology", n: int, max_skip: int = 1) -> np.ndarray:
    i, j = np.indices((n, n))
    if topology == pylibhmm.HmmTopology.LeftToRight:
        return j >= i
    if topology == pylibhmm.HmmTopology.LeftToRightSkip:
        return (j >= i) & (j <= i + max_skip)
    if topology == pylibhmm.HmmTopology.Banded:
        return np.abs(i - j) <= max_skip
    return np.ones((n, n), dtype=bool)


class TestTopology:
    @pytest.mark.parametrize(
        "topology",
        [
            pylibhmm.HmmTopology.Ergodic,
            pylibhmm.HmmTopology.LeftToRight,
            pylibhmm.HmmTopology.LeftToRightSkip,
            pylibhmm.HmmTopology.Banded,
        ],
    )
    def test_initialize_rows_stochastic_and_masked(self, topology):
        hmm = pylibhmm.Hmm(4)
        pylibhmm.initialize_topology(hmm, topology, max_skip=1)
        trans = hmm.get_trans()
        np.testing.assert_allclose(trans.sum(axis=1), np.ones(4))
        mask = _valid_mask(topology, 4)
        assert np.all(trans[~mask] == 0.0)
        # Valid transitions are uniform over each row's valid set.
        counts = mask.sum(axis=1)
        np.testing.assert_allclose(trans[mask], np.repeat(1.0 / counts, counts))

    def test_left_to_right_structure(self):
        hmm = pylibhmm.Hmm(3)
        pylibhmm.initialize_topology(hmm, pylibhmm.HmmTopology.LeftToRight)
        expected = np.array(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(hmm.get_trans(), expected)

    def test_max_skip_validation(self):
        hmm = pylibhmm.Hmm(3)
        with pytest.raises(ValueError):
            pylibhmm.initialize_topology(hmm, pylibhmm.HmmTopology.LeftToRightSkip, max_skip=0)
        with pytest.raises(ValueError):
            pylibhmm.enforce_topology(hmm, pylibhmm.HmmTopology.Banded, max_skip=0)

    def test_enforce_zeroes_and_renormalises(self):
        hmm = pylibhmm.Hmm(3)
        hmm.set_pi(np.array([1.0, 0.0, 0.0]))
        # Deliberately violates LeftToRight: mass below the diagonal.
        hmm.set_trans(
            np.array(
                [
                    [0.5, 0.25, 0.25],
                    [0.5, 0.25, 0.25],
                    [0.2, 0.2, 0.6],
                ]
            )
        )
        pylibhmm.enforce_topology(hmm, pylibhmm.HmmTopology.LeftToRight)
        trans = hmm.get_trans()
        assert np.all(trans[~_valid_mask(pylibhmm.HmmTopology.LeftToRight, 3)] == 0.0)
        np.testing.assert_allclose(trans.sum(axis=1), np.ones(3))
        # Row 1's valid mass {0.25, 0.25} renormalises to {0.5, 0.5}.
        np.testing.assert_allclose(trans[1], [0.0, 0.5, 0.5])

    def test_enforce_resets_degenerate_row_to_uniform(self):
        hmm = pylibhmm.Hmm(3)
        # Row 0 has all its mass on invalid (below-diagonal is fine for row 0,
        # so use row 2 of a Banded(1) topology: only j in {1, 2} are valid).
        hmm.set_trans(
            np.array(
                [
                    [0.5, 0.5, 0.0],
                    [0.3, 0.4, 0.3],
                    [1.0, 0.0, 0.0],
                ]
            )
        )
        pylibhmm.enforce_topology(hmm, pylibhmm.HmmTopology.Banded, max_skip=1)
        # Row 2's only valid entries {1, 2} had zero mass: reset to uniform.
        np.testing.assert_allclose(hmm.get_trans()[2], [0.0, 0.5, 0.5])

    def test_enforce_ergodic_is_identity(self):
        hmm = pylibhmm.Hmm(2)
        trans = np.array([[0.7, 0.3], [0.4, 0.6]])
        hmm.set_trans(trans)
        pylibhmm.enforce_topology(hmm, pylibhmm.HmmTopology.Ergodic)
        np.testing.assert_allclose(hmm.get_trans(), trans)

    def test_constrained_training_round_trip(self):
        # Issue #26: initialize -> train -> enforce keeps the mask.
        topology = pylibhmm.HmmTopology.LeftToRightSkip
        hmm = pylibhmm.Hmm(3)
        pylibhmm.initialize_topology(hmm, topology, max_skip=1)
        hmm.set_pi(np.array([1.0, 0.0, 0.0]))
        hmm.set_distribution(0, pylibhmm.Gaussian(mu=0.0, sigma=1.0))
        hmm.set_distribution(1, pylibhmm.Gaussian(mu=5.0, sigma=1.0))
        hmm.set_distribution(2, pylibhmm.Gaussian(mu=10.0, sigma=1.0))
        rng = np.random.default_rng(4)
        seqs = [
            np.concatenate([rng.normal(0, 1, 20), rng.normal(5, 1, 20), rng.normal(10, 1, 20)])
            for _ in range(3)
        ]
        trainer = pylibhmm.BaumWelchTrainer(hmm, seqs)
        for _ in range(20):
            trainer.train()
            pylibhmm.enforce_topology(hmm, topology, max_skip=1)
        trans = hmm.get_trans()
        mask = _valid_mask(topology, 3)
        assert np.all(trans[~mask] == 0.0)
        np.testing.assert_allclose(trans.sum(axis=1), np.ones(3))
        assert np.isfinite(trainer.last_log_probability)

    def test_mv_variants(self):
        hmm = _mv_hmm()
        pylibhmm.initialize_topology_mv(hmm, pylibhmm.HmmTopology.LeftToRight)
        trans = hmm.get_trans()
        assert trans[1, 0] == 0.0
        np.testing.assert_allclose(trans.sum(axis=1), np.ones(2))
        hmm.set_trans(np.array([[0.5, 0.5], [0.5, 0.5]]))
        pylibhmm.enforce_topology_mv(hmm, pylibhmm.HmmTopology.LeftToRight)
        np.testing.assert_allclose(hmm.get_trans(), [[0.5, 0.5], [0.0, 1.0]])


# ---------------------------------------------------------------------------
# validateInitialized (libhmm #78) — surfaced through existing bindings.
# ---------------------------------------------------------------------------
class TestValidateInitialized:
    def test_uninitialised_scoring_message(self):
        # A fresh Hmm zero-initialises pi and trans; scoring entry points
        # must reject it with the actionable message, not a numerics error.
        with pytest.raises(RuntimeError, match=r"pi.*all zero.*setPi"):
            pylibhmm.ForwardBackwardCalculator(pylibhmm.Hmm(2), np.array([1.0, 2.0, 3.0]))

    def test_uninitialised_training_message(self):
        # Validation fires at training entry, not at trainer construction.
        trainer = pylibhmm.BaumWelchTrainer(pylibhmm.Hmm(2), [np.array([1.0, 2.0])])
        with pytest.raises(RuntimeError, match=r"pi.*all zero"):
            trainer.train()
