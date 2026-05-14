"""
Elk movement state detection — joint Gamma + VonMises HMM
==========================================================
Fits a 2-state, joint Gamma + VonMises HMM to elk GPS step lengths and
turning angles from Morales et al. (2004). Directly comparable to the
canonical moveHMM (R package) reference fit.

The joint emission model cannot be expressed as a single pylibhmm
EmissionDistribution, so this example implements the forward-backward
pass and M-step directly with NumPy, using pylibhmm distributions only
for the weighted MLE fits (the M-step). This is a common pattern in
movement ecology where each observation is a bivariate (step, angle) pair.

Dataset
-------
4 elk GPS tracks, Morales et al. (2004). Step lengths in metres, turning
angles in radians (−π, π].

Data preparation
----------------
Run from the libhmm repository root:

    Rscript scripts/prepare_elk_data.R [output_dir]

This exports elk_<id>_obs.csv (one per animal) to /tmp/.
Pass the same directory as the first argument to this script.

Usage
-----
    python elk_movement_example.py [data_dir]

moveHMM reference fit (Gamma + VonMises, April 2026)
    State 0 (encamped):   step mean=373.8 m  sd=399.0 m  angle kappa=0.592
    State 1 (travelling): step mean=3247.3 m sd=4393.5 m angle kappa=0.208
    Transition: [[0.912, 0.088], [0.200, 0.800]]
    moveHMM log-likelihood: -6935.6  wall time: ~2000 ms

Reference
---------
Morales JM et al. (2004). Extracting more out of relocation data: building
  movement models as mixtures of random walks. Ecology, 85(9), 2436–2445.
Michelot T, Langrock R, Patterson TA (2016). moveHMM: an R package for
  analysing animal movement data as mixtures of random walks. Methods Ecol
  Evol, 7(11), 1308–1315.
"""

import glob
import math
import sys
import time

import numpy as np
import pylibhmm

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

print("Elk Movement State Detection — Joint Gamma + VonMises HMM")
print("vs moveHMM (R)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Load all elk tracks
# ---------------------------------------------------------------------------
csv_files = sorted(glob.glob(f"{data_dir}/elk_*_obs.csv"))
if not csv_files:
    print(f"\nERROR: no elk_*_obs.csv files in {data_dir}")
    print("Run:  Rscript scripts/prepare_elk_data.R  (from the libhmm repo root)")
    sys.exit(1)

all_steps:  list[np.ndarray] = []
all_angles: list[np.ndarray] = []
for path in csv_files:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    all_steps.append(data[:, 0].astype(np.float64))
    all_angles.append(data[:, 1].astype(np.float64))

n_seqs = len(all_steps)
n_total = sum(len(s) for s in all_steps)
print(f"\n{n_seqs} elk GPS tracks, {n_total} total observations\n")

# ---------------------------------------------------------------------------
# HMM parameters
# ---------------------------------------------------------------------------
N = 2  # states
pi  = np.array([0.5, 0.5])
A   = np.array([[0.9, 0.1], [0.1, 0.9]])
# Gamma distributions for step lengths
gamma_dists = [pylibhmm.Gamma(k=1.0, theta=300.0),
               pylibhmm.Gamma(k=1.0, theta=3000.0)]
# VonMises distributions for turning angles
vm_dists = [pylibhmm.VonMises(mu=0.0, kappa=0.5),
            pylibhmm.VonMises(mu=0.0, kappa=0.2)]

print("Initial parameters:")
for s in range(N):
    g, v = gamma_dists[s], vm_dists[s]
    lbl = "encamped " if s == 0 else "travelling"
    print(f"  State {s} ({lbl}): step mean={g.mean:.1f} m  "
          f"angle kappa={v.kappa:.3f}")
print()

# ---------------------------------------------------------------------------
# Custom forward-backward EM for joint distributions
# ---------------------------------------------------------------------------
NEG_INF = -np.inf

def log_emission(steps: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """(T, N) log-emission matrix for one sequence."""
    T = len(steps)
    logE = np.empty((T, N))
    for s in range(N):
        logE[:, s] = (gamma_dists[s].log_pdf(steps) +
                      vm_dists[s].log_pdf(angles))
    return logE


def forward_backward(logE: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Log-space forward-backward for one sequence.

    Returns (gamma, xi_sum, log_likelihood) where
      gamma[t, s]  = P(q_t=s | O, λ)  (T x N)
      xi_sum[i, j] = Σ_t ξ_t(i,j)    (N x N)
    """
    T, _ = logE.shape
    log_A = np.log(A)

    # Forward
    la = np.full((T, N), NEG_INF)
    la[0] = np.log(pi) + logE[0]
    for t in range(1, T):
        for j in range(N):
            la[t, j] = np.logaddexp.reduce(la[t-1] + log_A[:, j]) + logE[t, j]
    ll = np.logaddexp.reduce(la[-1])

    # Backward
    lb = np.zeros((T, N))
    for t in range(T - 2, -1, -1):
        for i in range(N):
            lb[t, i] = np.logaddexp.reduce(log_A[i] + logE[t+1] + lb[t+1])

    # Gamma
    log_gamma = la + lb
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    # Xi sum
    xi_sum = np.zeros((N, N))
    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                xi_sum[i, j] += np.exp(
                    la[t, i] + log_A[i, j] + logE[t+1, j] + lb[t+1, j] - ll
                )

    return gamma, xi_sum, ll


print("Baum-Welch EM:")
print(f"{'iter':>6}  {'log-likelihood':>16}  {'delta':>12}")
print("-" * 38)

t0 = time.perf_counter()
prev_ll = sum(
    forward_backward(log_emission(s, a))[2]
    for s, a in zip(all_steps, all_angles)
)
print(f"{'0':>6}  {prev_ll:>16.3f}  {'(initial)':>12}")

conv_iter = -1
for it in range(1, 201):
    # E-step: accumulate across all sequences
    gamma_all   = [[] for _ in range(N)]
    steps_all   = []
    angles_all  = []
    xi_sum_tot  = np.zeros((N, N))
    pi_num      = np.zeros(N)
    total_ll    = 0.0

    for steps, angles in zip(all_steps, all_angles):
        logE = log_emission(steps, angles)
        gamma, xi_sum, ll = forward_backward(logE)
        total_ll  += ll
        pi_num    += gamma[0]
        xi_sum_tot += xi_sum
        steps_all.extend(steps)
        angles_all.extend(angles)
        for s in range(N):
            gamma_all[s].extend(gamma[:, s])

    steps_np  = np.array(steps_all)
    angles_np = np.array(angles_all)
    for s in range(N):
        gamma_all[s] = np.array(gamma_all[s])

    # M-step: update π and A
    pi = pi_num / pi_num.sum()
    for i in range(N):
        row = xi_sum_tot[i]
        A[i] = row / row.sum()

    # M-step: update emission distributions via weighted MLE
    for s in range(N):
        gamma_dists[s].fit_weighted(steps_np,  gamma_all[s])
        vm_dists[s].fit_weighted(angles_np, gamma_all[s])

    delta = total_ll - prev_ll
    converged = it > 1 and abs(delta) < 0.01
    note = "  <- converged" if (converged and conv_iter < 0) else ""
    print(f"{it:>6}  {total_ll:>16.3f}  {delta:>12.4f}{note}")
    if converged and conv_iter < 0:
        conv_iter = it
    if conv_iter > 0 and it >= conv_iter + 2:
        break
    prev_ll = total_ll

wall_ms = (time.perf_counter() - t0) * 1000

# ---------------------------------------------------------------------------
# Sort states by step mean ascending: encamped first
# ---------------------------------------------------------------------------
enc, trav = (0, 1) if gamma_dists[0].mean < gamma_dists[1].mean else (1, 0)

print(f"\n=== pylibhmm results ===")
print(f"Wall time: {wall_ms:.0f} ms\n")
for idx, lbl in [(enc, "encamped  "), (trav, "travelling")]:
    g, v = gamma_dists[idx], vm_dists[idx]
    print(f"State {idx} ({lbl}): step mean={g.mean:7.1f} m  "
          f"sd={g.std:7.1f} m  angle kappa={v.kappa:.3f}")

print("\nTransition matrix:")
for row in A:
    print("  [" + "  ".join(f"{v:8.5f}" for v in row) + " ]")
print(f"\nLog-likelihood: {prev_ll:.1f}")

print(f"\n=== Comparison: pylibhmm vs moveHMM (R) ===\n")
print(f"{'':>26} {'pylibhmm':>10} {'moveHMM':>12}")
print("-" * 50)
ge, ve = gamma_dists[enc],  vm_dists[enc]
gt, vt = gamma_dists[trav], vm_dists[trav]
print(f"{'step mean encamped (m)':>26} {ge.mean:>10.1f} {373.8:>12.1f}")
print(f"{'step sd encamped (m)':>26} {ge.std:>10.1f} {399.0:>12.1f}")
print(f"{'step mean travelling (m)':>26} {gt.mean:>10.1f} {3247.3:>12.1f}")
print(f"{'step sd travelling (m)':>26} {gt.std:>10.1f} {4393.5:>12.1f}")
print(f"{'angle kappa encamped':>26} {ve.kappa:>10.3f} {0.592:>12.3f}")
print(f"{'angle kappa travelling':>26} {vt.kappa:>10.3f} {0.208:>12.3f}")
print(f"{'A[enc→enc]':>26} {A[enc,enc]:>10.4f} {0.912:>12.3f}")
print(f"{'A[trav→trav]':>26} {A[trav,trav]:>10.4f} {0.800:>12.3f}")
print(f"{'Log-likelihood':>26} {prev_ll:>10.1f} {-6935.6:>12.1f}")
print(f"{'Wall time':>26} {wall_ms:.0f} ms   {'~2000 ms':>12}")
