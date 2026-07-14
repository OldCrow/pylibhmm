"""
Wind direction regime detection — VonMises HMM
===============================================
Fits a 2-state VonMisesDistribution HMM to hourly wind directions at
Chicago O'Hare International Airport, 2015. Benchmarked against the
HiddenMarkov R package (which uses a Normal approximation — incorrect
for circular data because angles near 0 and 2π are close but the Normal
treats them as far apart).

Dataset
-------
NOAA Integrated Surface Database (ISD), station 725300-14819
(Chicago O'Hare), year 2015. 11,894 valid hourly wind direction
observations (calm/missing excluded).

Data preparation
----------------
Run from the libhmm repository root:

    Rscript scripts/prepare_wind_data.R [output_dir]

This exports ohare_wind_2015.csv to /tmp/ (or the specified dir).
Pass the same directory as the first argument to this script.

Usage
-----
    python wind_direction_example.py [data_dir]

HiddenMarkov R reference (Normal approximation, April 2026)
    State 1 (prevailing):  mean=49.6°  sd=0.468 rad
    State 2 (variable):    mean=239.1° sd=1.093 rad
    LL: -16830.8  (Normal approx — not directly comparable)
    Wall time: ~0.24 s

Note: VonMisesDistribution is the correct model for circular data; the Normal
approximation fails at the 0/2π wrap-around boundary.

Reference
---------
NOAA NCEI (2001). Global Surface Hourly [ISD]. NCEI.
Zucchini W, MacDonald IL, Langrock R (2017). Hidden Markov Models for
  Time Series: An Introduction Using R, 2nd ed. CRC Press. (Ch. 10.)
"""

import math
import sys
import time

import numpy as np

import pylibhmm

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
csv_path = f"{data_dir}/ohare_wind_2015.csv"

print("Wind Direction Regime Detection — Chicago O'Hare 2015")
print("VonMises HMM vs HiddenMarkov (Normal approximation)")
print("=" * 56)

# Load data: first column is direction in radians [0, 2π]
try:
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=0)
except OSError:
    print(f"\nERROR: cannot open {csv_path}")
    print("Run:  Rscript scripts/prepare_wind_data.R  (from the libhmm repo root)")
    sys.exit(1)

directions = raw.astype(np.float64)
print(f"\nNOAA ISD, O'Hare (725300): {len(directions)} hourly wind directions (2015)")
print("Directions in radians [0, 2π]\n")

# 2-state HMM: prevailing SW/W wind vs variable/N wind
hmm = pylibhmm.Hmm(2)
hmm.set_pi(np.array([0.4, 0.6]))
hmm.set_trans(np.array([[0.95, 0.05], [0.05, 0.95]]))
# State 0: prevailing SW (~225° = 3.93 rad), concentrated
# State 1: variable/N  (~0°  = 0.00 rad), dispersed
hmm.set_distribution(0, pylibhmm.VonMises(mu=3.93, kappa=2.0))
hmm.set_distribution(1, pylibhmm.VonMises(mu=0.00, kappa=0.5))

print("Initial parameters:")
print("  State 0: mu=3.93 rad (225°, SW)  kappa=2.0 (concentrated)")
print("  State 1: mu=0.00 rad (0°,   N)   kappa=0.5 (dispersed)\n")

# Baum-Welch EM
sequences = [directions]
trainer = pylibhmm.BaumWelchTrainer(hmm, sequences)

prev_ll = pylibhmm.ForwardBackwardCalculator(hmm, directions).log_probability
print(f"{'iter':>6}  {'log-likelihood':>15}  {'delta':>11}")
print("-" * 36)
print(f"{'0':>6}  {prev_ll:>15.3f}  {'(initial)':>11}")

t0 = time.perf_counter()
conv_iter = -1
for i in range(1, 201):
    trainer.train()
    ll = pylibhmm.ForwardBackwardCalculator(hmm, directions).log_probability
    delta = ll - prev_ll
    converged = i > 1 and abs(delta) < 1.0
    note = "  <- converged" if (converged and conv_iter < 0) else ""
    print(f"{i:>6}  {ll:>15.3f}  {delta:>11.3f}{note}")
    if converged and conv_iter < 0:
        conv_iter = i
    if conv_iter > 0 and i >= conv_iter + 2:
        break
    prev_ll = ll

wall_ms = (time.perf_counter() - t0) * 1000
final_ll = pylibhmm.ForwardBackwardCalculator(hmm, directions).log_probability

# Sort by kappa descending: concentrated (prevailing) first
d0, d1 = hmm.get_distribution(0), hmm.get_distribution(1)
conc, disp = (0, 1) if d0.kappa > d1.kappa else (1, 0)

def to_deg(r: float) -> float:
    return r * 180.0 / math.pi

dc = hmm.get_distribution(conc)
dd = hmm.get_distribution(disp)

print("\n=== pylibhmm results ===")
print(f"Wall time: {wall_ms:.1f} ms\n")
print(f"State {conc} (prevailing):  mu={dc.mu:.4f} rad ({to_deg(dc.mu):.1f}°)"
      f"  kappa={dc.kappa:.4f}  circ_var={dc.circular_variance:.4f}")
print(f"State {disp} (variable):    mu={dd.mu:.4f} rad ({to_deg(dd.mu):.1f}°)"
      f"  kappa={dd.kappa:.4f}  circ_var={dd.circular_variance:.4f}")

T = hmm.get_trans()
print("\nTransition matrix:")
for row in T:
    print("  [" + "  ".join(f"{v:9.5f}" for v in row) + " ]")
print(f"\nLog-likelihood (VonMises): {final_ll:.1f}")

# State occupancy
vc = pylibhmm.ViterbiCalculator(hmm, directions)
viterbi_states = vc.decode()
fb = pylibhmm.ForwardBackwardCalculator(hmm, directions)
posterior_states = fb.decode_posterior()
v_cnt = [int((viterbi_states  == s).sum()) for s in range(2)]
p_cnt = [int((posterior_states == s).sum()) for s in range(2)]
print(f"\nViterbi   occupancy: prevailing={v_cnt[conc]:5d}  variable={v_cnt[disp]:5d} hr")
print(f"Posterior occupancy: prevailing={p_cnt[conc]:5d}  variable={p_cnt[disp]:5d} hr")

# Model selection
mc = pylibhmm.evaluate_model(hmm, final_ll, len(directions))
print(f"\nModel selection (k={pylibhmm.count_free_parameters(hmm)}, n={len(directions)}):")
print(f"  AIC  = {mc.aic:.1f}")
print(f"  BIC  = {mc.bic:.1f}")
print(f"  AICc = {mc.aicc:.1f}")

print("\nNote: HiddenMarkov (R) LL=-16830.8 uses a Normal approximation, which")
print("fails at the 0/2π boundary. VonMisesDistribution is the correct model")
print("for circular data and is not directly comparable to the Normal LL.")
