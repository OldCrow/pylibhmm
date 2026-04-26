"""Basic pylibhmm usage."""

import numpy as np

import pylibhmm

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

observations = np.array([0, 1, 5, 4, 2], dtype=np.float64)

fb = pylibhmm.ForwardBackwardCalculator(hmm, observations)
print("log P(O|λ):", fb.log_probability)
print("P(O|λ):", fb.probability)

vt = pylibhmm.ViterbiCalculator(hmm, observations)
path = vt.decode()
print("Viterbi path:", path)
