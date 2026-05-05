"""Basic pylibhmm usage — model, inference, and JSON I/O."""

import json
import tempfile
from pathlib import Path

import numpy as np

import pylibhmm

# ---------------------------------------------------------------------------
# Build a 2-state HMM (fair vs. loaded die)
# ---------------------------------------------------------------------------
hmm = pylibhmm.Hmm(2)
hmm.set_pi(np.array([0.75, 0.25], dtype=np.float64))
hmm.set_trans(np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float64))

fair = pylibhmm.Discrete(6)
for i in range(6):
    fair.set_probability(i, 1.0 / 6.0)

loaded = pylibhmm.Discrete(6)
for i in range(5):
    loaded.set_probability(i, 0.125)
loaded.set_probability(5, 0.375)

hmm.set_distribution(0, fair)
hmm.set_distribution(1, loaded)

# repr(d) returns the distribution's text representation.
# Use named properties for programmatic access, not the repr string.
print("State 0 distribution:", repr(fair))
print("num_symbols:", fair.num_symbols)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
observations = np.array([0, 1, 5, 4, 2], dtype=np.float64)

fb = pylibhmm.ForwardBackwardCalculator(hmm, observations)
print("log P(O|λ):", fb.log_probability)
print("P(O|λ):   ", fb.probability)

vt = pylibhmm.ViterbiCalculator(hmm, observations)
path = vt.decode()
print("Viterbi path:", path)

# ---------------------------------------------------------------------------
# JSON I/O (recommended as of libhmm v3.4.0)
# ---------------------------------------------------------------------------

# Serialize to / deserialize from a string.
json_str = pylibhmm.to_json(hmm)
print("JSON keys:", list(json.loads(json_str).keys()))

hmm2 = pylibhmm.from_json(json_str)
print("Round-tripped num_states:", hmm2.num_states)
np.testing.assert_allclose(hmm2.get_pi(), hmm.get_pi())

# Save to / load from a file.
with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "model.json"
    pylibhmm.save_json(hmm, path)
    hmm3 = pylibhmm.load_json(path)
    print("Loaded from file, num_states:", hmm3.num_states)
