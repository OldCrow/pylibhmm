"""
DAX market regime detection — 3-state Student-t HMM
=====================================================
Fits a 3-state location-scale Student-t HMM to DAX daily log-returns
(2000–2022) using Baum-Welch EM. Directly comparable to the fHMM
R package reference fit.

Data source
-----------
Primary:  yfinance — fetched automatically if installed:

    pip install yfinance

Fallback: dax_logreturns.csv produced by the libhmm R script:

    Rscript scripts/prepare_dax_data.R [output_dir]

Usage
-----
    python dax_regime_example.py [data_dir]

If yfinance is available the data_dir argument is ignored.

fHMM reference fit (3-state Student-t, fHMM 1.2.0, April 2026)
    State 1 (bearish):  mu=-0.00180  sigma=0.02629  nu=11.2  | 702 days
    State 2 (bullish):  mu=+0.00126  sigma=0.00600  nu= 5.3  | 2362 days
    State 3 (neutral):  mu=-0.00031  sigma=0.01330  nu=91.2  | 2774 days
    Log-likelihood: 17485.7
    fHMM wall time: ~1360 s

Note: fHMM uses a numerical optimizer (nlm) on the full likelihood, not
direct EM — this is the source of the large wall-time difference.

Reference
---------
Oelschläger L, Adam T, Michels R (2024). fHMM: Hidden Markov Models for
Financial Time Series in R. J. Statistical Software, 109(9).
https://doi.org/10.18637/jss.v109.i09
"""

import sys
import time

import numpy as np
import pylibhmm


def load_returns(ticker: str, start: str, end: str,
                 csv_fallback: str) -> np.ndarray:
    """Load log-returns from yfinance (preferred) or a CSV fallback."""
    try:
        import yfinance as yf  # type: ignore[import]
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        prices = df["Close"].dropna().to_numpy(dtype=np.float64)
        returns = np.diff(np.log(prices))
        print(f"Downloaded {ticker} ({start} to {end}) via yfinance: "
              f"{len(returns)} daily log-returns")
        return returns
    except ImportError:
        pass
    except Exception as exc:
        print(f"yfinance fetch failed ({exc}); falling back to CSV")

    try:
        raw = np.loadtxt(csv_fallback, skiprows=1, dtype=np.float64)
        print(f"Loaded {len(raw)} log-returns from {csv_fallback}")
        return raw
    except OSError:
        print(f"\nERROR: cannot open {csv_fallback}")
        print("Install yfinance (pip install yfinance) or run:")
        print("  Rscript scripts/prepare_dax_data.R  (from the libhmm repo root)")
        sys.exit(1)


def run_regime_hmm(returns: np.ndarray, label: str,
                   ref_ll: float, ref_time_s: float) -> None:
    """Fit a 3-state StudentT HMM and print a comparison table."""
    print(f"\n{'Baum-Welch training:':}")
    print(f"{'iter':>7}  {'log-likelihood':>16}  {'delta':>12}")
    print("-" * 38)

    hmm = pylibhmm.Hmm(3)
    hmm.set_pi(np.array([0.333, 0.334, 0.333]))
    hmm.set_trans(np.array([
        [0.95, 0.03, 0.02],
        [0.02, 0.95, 0.03],
        [0.02, 0.03, 0.95],
    ]))
    hmm.set_distribution(0, pylibhmm.StudentT(nu=10.0, location=-0.002, scale=0.025))
    hmm.set_distribution(1, pylibhmm.StudentT(nu=30.0, location=0.000,  scale=0.012))
    hmm.set_distribution(2, pylibhmm.StudentT(nu= 5.0, location=0.001,  scale=0.006))

    sequences = [returns]
    trainer = pylibhmm.BaumWelchTrainer(hmm, sequences)

    prev_ll = pylibhmm.ForwardBackwardCalculator(hmm, returns).log_probability
    print(f"{'0':>7}  {prev_ll:>16.4f}  {'(initial)':>12}")

    t0 = time.perf_counter()
    conv_iter = -1
    for i in range(1, 201):
        trainer.train()
        ll = pylibhmm.ForwardBackwardCalculator(hmm, returns).log_probability
        delta = ll - prev_ll
        converged = i > 1 and abs(delta) < 1e-4
        note = "  <- converged" if (converged and conv_iter < 0) else ""
        print(f"{i:>7}  {ll:>16.4f}  {delta:>12.6f}{note}")
        if converged and conv_iter < 0:
            conv_iter = i
        if conv_iter > 0 and i >= conv_iter + 2:
            break
        prev_ll = ll

    wall_s = time.perf_counter() - t0
    final_ll = pylibhmm.ForwardBackwardCalculator(hmm, returns).log_probability

    # Sort states by sigma descending: bearish (most volatile) first
    sigmas = [hmm.get_distribution(s).scale for s in range(3)]
    order = sorted(range(3), key=lambda s: sigmas[s], reverse=True)
    labels_sorted = ["bearish", "neutral", "bullish"]

    print(f"\n=== pylibhmm {label} results ===")
    print(f"Wall time: {wall_s:.1f} s\n")
    for rank, (state, lbl) in enumerate(zip(order, labels_sorted)):
        d = hmm.get_distribution(state)
        print(f"State {state} ({lbl}):  nu={d.nu:6.1f}  "
              f"mu={d.location:+.5f}  sigma={d.scale:.5f}")

    T = hmm.get_trans()
    print("\nTransition matrix:")
    for row in T:
        print("  [" + "  ".join(f"{v:9.5f}" for v in row) + " ]")
    print(f"\nLog-likelihood: {final_ll:.1f}")

    # Viterbi and posterior decoding occupancy
    vc = pylibhmm.ViterbiCalculator(hmm, returns)
    fb = pylibhmm.ForwardBackwardCalculator(hmm, returns)
    viterbi   = vc.decode()
    posterior = fb.decode_posterior()
    v_cnt = [int((viterbi   == s).sum()) for s in order]
    p_cnt = [int((posterior == s).sum()) for s in order]
    print("\nViterbi   occupancy:  " +
          "  ".join(f"{lbl}={v_cnt[r]:4d}" for r, lbl in enumerate(labels_sorted)))
    print("Posterior occupancy:  " +
          "  ".join(f"{lbl}={p_cnt[r]:4d}" for r, lbl in enumerate(labels_sorted)))

    # Model selection
    mc = pylibhmm.evaluate_model(hmm, final_ll, len(returns))
    k  = pylibhmm.count_free_parameters(hmm)
    print(f"\nModel selection (k={k}, n={len(returns)}):")
    print(f"  AIC  = {mc.aic:.1f}")
    print(f"  BIC  = {mc.bic:.1f}")
    print(f"  AICc = {mc.aicc:.1f}")

    print(f"\n=== pylibhmm vs fHMM (R) ===\n")
    print(f"{'':>20} {'pylibhmm':>12} {'fHMM':>10}")
    print("-" * 44)
    print(f"{'Log-likelihood':>20} {final_ll:>12.1f} {ref_ll:>10.1f}")
    print(f"{'Wall time':>20} {wall_s:.1f} s    {ref_time_s:>7.0f} s")
    print(f"{'Speedup':>20} {ref_time_s/max(wall_s,0.1):>11.0f}x {'(1x)':>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
data_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

print("DAX Market Regime Detection — pylibhmm vs fHMM")
print("=" * 50)

returns = load_returns(
    ticker="^GDAXI",
    start="2000-01-01",
    end="2022-12-31",
    csv_fallback=f"{data_dir}/dax_logreturns.csv",
)
print(f"  min={returns.min():.6f}  max={returns.max():.6f}"
      f"  mean={returns.mean():.6f}")
print("\nInitial parameters (same as C++ dax_regime_example):")
print("  State 0 (bearish):  nu=10  mu=-0.002  sigma=0.025")
print("  State 1 (neutral):  nu=30  mu= 0.000  sigma=0.012")
print("  State 2 (bullish):  nu= 5  mu= 0.001  sigma=0.006")

run_regime_hmm(returns, label="DAX",
               ref_ll=17485.7, ref_time_s=1360.0)
