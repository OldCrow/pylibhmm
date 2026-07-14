"""
S&P 500 market regime detection — 3-state Student-t HMM
=========================================================
Fits the same 3-state location-scale Student-t HMM as dax_regime_example
to S&P 500 daily log-returns (2000–2022). Comparing DAX and S&P 500 results
reveals differences in US vs German equity market regimes.

Data source
-----------
Primary:  yfinance — fetched automatically if installed:

    pip install yfinance

Fallback: sp500_logreturns.csv produced by the libhmm R script:

    Rscript scripts/prepare_dax_data.R [output_dir]

(The DAX R script also exports S&P 500 data to the same directory.)

Usage
-----
    python sp500_regime_example.py [data_dir]

If yfinance is available the data_dir argument is ignored.

DAX reference (pylibhmm 0.4.0, for cross-market comparison)
    Bearish: mu=-0.00179  sigma=0.02628  nu=11.1
    Neutral: mu=-0.00028  sigma=0.01305  nu=36.1
    Bullish: mu=+0.00126  sigma=0.00599  nu= 5.4
    DAX LL: ~17487

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
        prices = df["Close"].squeeze().dropna().to_numpy(dtype=np.float64).flatten()
        returns = np.diff(np.log(prices), axis=0).flatten()
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


def run_regime_hmm(returns: np.ndarray, label: str) -> None:
    """Fit a 3-state StudentT HMM and print results with model selection."""
    print("\nBaum-Welch training:")
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

    # Sort states by sigma descending: bearish first
    order = sorted(range(3),
                   key=lambda s: hmm.get_distribution(s).scale,
                   reverse=True)
    labels_sorted = ["bearish", "neutral", "bullish"]

    print(f"\n=== pylibhmm {label} results ===")
    print(f"Wall time: {wall_s:.1f} s\n")
    for state, lbl in zip(order, labels_sorted):
        d = hmm.get_distribution(state)
        print(f"State {state} ({lbl}):  nu={d.nu:6.1f}  "
              f"mu={d.location:+.5f}  sigma={d.scale:.5f}")

    T = hmm.get_trans()
    print("\nTransition matrix:")
    for row in T:
        print("  [" + "  ".join(f"{v:9.5f}" for v in row) + " ]")
    print(f"\nLog-likelihood: {final_ll:.1f}")

    # Viterbi and posterior decoding
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

    # Model selection — compare 2-state vs 3-state via AIC/BIC
    mc3 = pylibhmm.evaluate_model(hmm, final_ll, len(returns))
    k3  = pylibhmm.count_free_parameters(hmm)
    print(f"\nModel selection — 3-state model (k={k3}, n={len(returns)}):")
    print(f"  AIC  = {mc3.aic:.1f}")
    print(f"  BIC  = {mc3.bic:.1f}")
    print(f"  AICc = {mc3.aicc:.1f}")

    # Quick 2-state fit for comparison
    hmm2 = pylibhmm.Hmm(2)
    hmm2.set_pi(np.array([0.5, 0.5]))
    hmm2.set_trans(np.array([[0.95, 0.05], [0.05, 0.95]]))
    hmm2.set_distribution(0, pylibhmm.StudentT(nu=10.0, location=-0.001, scale=0.020))
    hmm2.set_distribution(1, pylibhmm.StudentT(nu=10.0, location=0.001,  scale=0.008))
    t2 = pylibhmm.BaumWelchTrainer(hmm2, sequences)
    ll2_prev = pylibhmm.ForwardBackwardCalculator(hmm2, returns).log_probability
    for _ in range(100):
        t2.train()
        ll2 = pylibhmm.ForwardBackwardCalculator(hmm2, returns).log_probability
        if abs(ll2 - ll2_prev) < 1e-4:
            break
        ll2_prev = ll2
    mc2 = pylibhmm.evaluate_model(hmm2, ll2_prev, len(returns))
    k2  = pylibhmm.count_free_parameters(hmm2)
    print(f"\nModel selection — 2-state model (k={k2}, n={len(returns)}):")
    print(f"  AIC  = {mc2.aic:.1f}")
    print(f"  BIC  = {mc2.bic:.1f}")
    print(f"  AICc = {mc2.aicc:.1f}")
    winner = "3-state" if mc3.bic < mc2.bic else "2-state"
    print(f"\nBIC selects: {winner} model")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
data_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

print("S&P 500 Market Regime Detection — pylibhmm")
print("=" * 50)

returns = load_returns(
    ticker="^GSPC",
    start="2000-01-01",
    end="2022-12-31",
    csv_fallback=f"{data_dir}/sp500_logreturns.csv",
)
print(f"  min={returns.min():.6f}  max={returns.max():.6f}"
      f"  mean={returns.mean():.6f}")
print("\nInitial parameters (same as DAX example):")
print("  State 0 (bearish):  nu=10  mu=-0.002  sigma=0.025")
print("  State 1 (neutral):  nu=30  mu= 0.000  sigma=0.012")
print("  State 2 (bullish):  nu= 5  mu= 0.001  sigma=0.006")

run_regime_hmm(returns, label="S&P 500")
