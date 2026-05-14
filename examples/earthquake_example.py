"""
Earthquake regime detection — Poisson HMM
==========================================
Fits a 2-state Poisson HMM to annual major earthquake counts (magnitude ≥ 7)
worldwide, 1900–2006. The canonical running example from Zucchini & MacDonald
(2009), benchmarked against the HiddenMarkov R package.

Dataset
-------
107 annual counts from the USGS earthquake catalogue, published in
Zucchini & MacDonald Table 1.1. Data are embedded — no download needed.

HiddenMarkov R reference (BaumWelch, April 2026)
    λ_low  = 15.418  (low seismicity,  65 years)
    λ_high = 26.013  (high seismicity, 42 years)
    Transition: [[0.928, 0.072], [0.119, 0.881]]
    Log-likelihood: -341.879
    Wall time: ~0.02 s

Reference
---------
Zucchini W, MacDonald IL (2009). Hidden Markov Models for Time Series:
An Introduction Using R. CRC Press. (Table 1.1, Chapters 3–4.)
Harte D (2025). HiddenMarkov: Hidden Markov Models. CRAN.
"""

import time

import numpy as np
import pylibhmm

# Annual major earthquake counts 1900–2006 (Zucchini & MacDonald Table 1.1)
EQ_DATA = np.array([
    13, 14,  8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18, 25, 21, 21, 14,
     8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26, 13, 14, 22, 24, 21, 22, 26, 21,
    23, 24, 27, 41, 31, 27, 35, 26, 28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15,
    22, 18, 15, 20, 15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
    18, 14, 10, 15,  8, 15,  6, 11,  8,  7, 18, 16, 13, 12, 13, 20, 15, 16, 12, 18,
    15, 16, 13, 15, 16, 11, 11,
], dtype=np.float64)

print("Major Earthquake Counts 1900–2006 — Poisson HMM")
print("=" * 50)
print(f"\nDataset: {len(EQ_DATA)} annual counts (1900–2006)")
print(f"  mean={EQ_DATA.mean():.3f}  min={int(EQ_DATA.min())}  max={int(EQ_DATA.max())}\n")

# 2-state HMM: low and high seismicity
hmm = pylibhmm.Hmm(2)
hmm.set_pi(np.array([0.5, 0.5]))
hmm.set_trans(np.array([[0.9, 0.1], [0.1, 0.9]]))
hmm.set_distribution(0, pylibhmm.Poisson(lam=15.0))  # low seismicity
hmm.set_distribution(1, pylibhmm.Poisson(lam=25.0))  # high seismicity

print("Initial: λ_low=15.0, λ_high=25.0\n")

# Baum-Welch EM
sequences = [EQ_DATA]
trainer = pylibhmm.BaumWelchTrainer(hmm, sequences)

prev_ll = pylibhmm.ForwardBackwardCalculator(hmm, EQ_DATA).log_probability
print(f"{'iter':>6}  {'logL':>14}  {'delta':>12}")
print("-" * 36)
print(f"{'0':>6}  {prev_ll:>14.4f}  {'(initial)':>12}")

t0 = time.perf_counter()
conv_iter = -1
for i in range(1, 201):
    trainer.train()
    ll = pylibhmm.ForwardBackwardCalculator(hmm, EQ_DATA).log_probability
    delta = ll - prev_ll
    converged = i > 1 and abs(delta) < 1e-5
    note = "  <- converged" if (converged and conv_iter < 0) else ""
    print(f"{i:>6}  {ll:>14.4f}  {delta:>12.6f}{note}")
    if converged and conv_iter < 0:
        conv_iter = i
    if conv_iter > 0 and i >= conv_iter + 2:
        break
    prev_ll = ll

wall_ms = (time.perf_counter() - t0) * 1000
final_ll = pylibhmm.ForwardBackwardCalculator(hmm, EQ_DATA).log_probability

# Sort states by λ ascending (low seismicity first)
lam0 = hmm.get_distribution(0).lam
lam1 = hmm.get_distribution(1).lam
lo, hi = (0, 1) if lam0 < lam1 else (1, 0)
lam_lo = hmm.get_distribution(lo).lam
lam_hi = hmm.get_distribution(hi).lam
T = hmm.get_trans()

print(f"\n=== pylibhmm results ===")
print(f"Wall time: {wall_ms:.1f} ms\n")
print(f"State {lo} (low  seismicity): λ = {lam_lo:.4f}")
print(f"State {hi} (high seismicity): λ = {lam_hi:.4f}\n")
print("Transition matrix:")
for row in T:
    print("  [" + "  ".join(f"{v:8.5f}" for v in row) + " ]")
print(f"\nLog-likelihood: {final_ll:.3f}")

# Viterbi and posterior decoding
fb = pylibhmm.ForwardBackwardCalculator(hmm, EQ_DATA)
vc = pylibhmm.ViterbiCalculator(hmm, EQ_DATA)
viterbi_states  = vc.decode()
posterior_states = fb.decode_posterior()

v_cnt  = [int((viterbi_states  == s).sum()) for s in range(2)]
p_cnt  = [int((posterior_states == s).sum()) for s in range(2)]
print(f"\nViterbi   occupancy: low={v_cnt[lo]:3d} yr  high={v_cnt[hi]:3d} yr")
print(f"Posterior occupancy: low={p_cnt[lo]:3d} yr  high={p_cnt[hi]:3d} yr")

# Model selection
mc = pylibhmm.evaluate_model(hmm, final_ll, len(EQ_DATA))
print(f"\nModel selection (k={pylibhmm.count_free_parameters(hmm)}, n={len(EQ_DATA)}):")
print(f"  AIC  = {mc.aic:.3f}")
print(f"  BIC  = {mc.bic:.3f}")
print(f"  AICc = {mc.aicc:.3f}")

print(f"\n=== Comparison: pylibhmm vs HiddenMarkov (R) ===\n")
print(f"{'':>22} {'pylibhmm':>12} {'HiddenMarkov':>14}")
print("-" * 50)
print(f"{'λ low':>22} {lam_lo:>12.4f} {15.418:>14.3f}")
print(f"{'λ high':>22} {lam_hi:>12.4f} {26.013:>14.3f}")
print(f"{'A[low→low]':>22} {T[lo, lo]:>12.4f} {0.9283:>14.4f}")
print(f"{'A[high→low]':>22} {T[hi, lo]:>12.4f} {0.1189:>14.4f}")
print(f"{'Log-likelihood':>22} {final_ll:>12.2f} {-341.879:>14.3f}")
print(f"{'Wall time':>22} {wall_ms:.0f} ms{'~20 ms':>18}")
print()
print("Notes:")
print("  Both fit the same 2-state Poisson HMM via Baum-Welch EM.")
print("  HiddenMarkov uses nlm()-based maximisation; pylibhmm uses direct EM.")
print("  The earthquake dataset is the canonical example in Zucchini &")
print("  MacDonald (2009), used throughout Chapters 3–7.")
