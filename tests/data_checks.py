"""Pre-build data checks that GATE the model refonte.

Two cheap, decisive questions the advisor flagged before writing any model:

1. **Is volatility "rough"?** (Gatheral-Jaisson-Rosenbaum 2018). Estimate the
   smoothness/Hurst exponent H of the log realized-variance series from the
   scaling of its q-th structure function:  m(q, Δ) = E[|log σ_{t+Δ} − log σ_t|^q]
   ∝ Δ^{ζ_q}, with ζ_q = q·H for a fractional process. If H is materially below
   0.5 (the GJR claim is H≈0.1), a rough/fractional vol model is justified over
   plain HAR. If H≈0.5 it buys nothing — don't build it.

2. **Are regimes statistically real?** Compare the hardcoded heuristic regime
   labelling (core.market._detect_regimes) against a quick Gaussian-mixture-style
   separation on [r, |r|] space: persistence (diagonal of the empirical
   transition matrix) and state separation. A proper HMM is only worth building
   if returns genuinely cluster into persistent, separable states.

Run:  PYTHONPATH=. .venv/bin/python tests/data_checks.py [SYM ...]
"""
from __future__ import annotations

import sys

import numpy as np

from core.market import _detect_regimes, load_market_simulator
from ml.vol_model import parkinson_var, real_ohlc


# --------------------------------------------------------------------------- #
# 1. Roughness / Hurst of log realized variance                               #
# --------------------------------------------------------------------------- #
def _logrv_series(o, h, l, c) -> np.ndarray:
    """Per-bar log realized variance. Range-based (Parkinson) when OHLC is real,
    else squared close-returns. This is the 'σ²_t' series whose roughness we test."""
    c = np.asarray(c, dtype=np.float64)
    if real_ohlc(o, h, l, c):
        v = parkinson_var(np.asarray(h, dtype=np.float64), np.asarray(l, dtype=np.float64))
    else:
        r = np.diff(np.log(c))
        v = np.concatenate([[np.var(r[:20])], r * r])
    return np.log(np.maximum(v, 1e-12))


def hurst_structure(logrv: np.ndarray, qs=(0.5, 1.0, 1.5, 2.0, 3.0),
                    lags=range(1, 50)) -> dict:
    """Estimate H via the GJR structure-function method.

    For each moment order q, regress log m(q,Δ) on log Δ; the slope is ζ_q. For a
    fractional process ζ_q = q·H, so H_q = ζ_q / q. We report the per-q H and the
    pooled estimate (slope of ζ_q vs q). A monotone, linear ζ_q in q with small
    intercept is the fBm signature; H well below 0.5 ⇒ rough.
    """
    x = np.asarray(logrv, dtype=np.float64)
    x = x[np.isfinite(x)]
    lags = list(lags)
    zeta = []
    hq = []
    for q in qs:
        m = []
        for d in lags:
            diff = np.abs(x[d:] - x[:-d]) ** q
            m.append(np.mean(diff))
        m = np.asarray(m)
        good = m > 0
        slope = np.polyfit(np.log(np.asarray(lags)[good]), np.log(m[good]), 1)[0]
        zeta.append(slope)
        hq.append(slope / q)
    zeta = np.asarray(zeta)
    # Pooled H = slope of ζ_q vs q (forced through the q axis).
    H_pooled = float(np.polyfit(np.asarray(qs), zeta, 1)[0])
    return {"qs": list(qs), "zeta": [float(z) for z in zeta],
            "H_per_q": [float(z) for z in hq], "H_pooled": H_pooled,
            "H_q2": float(zeta[list(qs).index(2.0)] / 2.0) if 2.0 in qs else float("nan")}


# --------------------------------------------------------------------------- #
# 2. Regime reality: heuristic persistence + simple 3-state separation        #
# --------------------------------------------------------------------------- #
def _transition_diag(labels: np.ndarray, k: int) -> float:
    """Mean diagonal mass of the empirical transition matrix (persistence)."""
    counts = np.ones((k, k))
    for i in range(len(labels) - 1):
        counts[labels[i], labels[i + 1]] += 1
    P = counts / counts.sum(axis=1, keepdims=True)
    return float(np.mean(np.diag(P)))


def regime_check(prices: np.ndarray) -> dict:
    """Persistence of the hardcoded heuristic regimes vs a naive vol-tercile
    labelling, and the separation of mean |r| across vol terciles (a quick proxy
    for 'do states separate?'). High persistence + separation ⇒ HMM worthwhile."""
    from core.market import Regime
    reg = _detect_regimes(prices)
    idx = {r: i for i, r in enumerate(Regime)}
    heur = np.array([idx[r] for r in reg[20:]])  # drop warmup
    heur_pers = _transition_diag(heur, 4)

    # Naive 3-state vol labelling on a rolling 20-bar σ (low/mid/high vol).
    r = np.diff(np.log(prices))
    sig = np.array([np.std(r[max(0, i - 20):i]) if i >= 5 else r[:i].std() if i >= 2 else 0.0
                    for i in range(1, len(r) + 1)])
    sig = sig[19:]
    qlo, qhi = np.quantile(sig, [1 / 3, 2 / 3])
    vol_lab = np.where(sig < qlo, 0, np.where(sig > qhi, 2, 1))
    vol_pers = _transition_diag(vol_lab, 3)

    # Separation: next-bar |return| conditional on current vol state — volatility
    # clustering means high-vol state predicts a larger next move.
    fwd_absr = np.abs(r[20:])
    m = min(len(vol_lab), len(fwd_absr))
    vl, fa = vol_lab[:m], fwd_absr[:m]
    sep = {int(s): float(np.mean(fa[vl == s])) for s in (0, 1, 2) if np.any(vl == s)}
    return {"heuristic_persistence": heur_pers, "volstate_persistence": vol_pers,
            "next_absret_by_volstate": sep,
            "separation_ratio": (sep.get(2, 0) / sep.get(0, 1e-9)) if sep else float("nan")}


def main() -> None:
    assets = sys.argv[1:] or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    tf = "1h"
    print("=" * 78)
    print("DATA CHECKS — do rough-vol and HMM regimes earn their complexity?")
    print("=" * 78)
    for sym in assets:
        sim = load_market_simulator(sym, tf)
        p = sim.prices
        print(f"\n### {sym} {tf}  [{sim.source}, {len(p)} bars] ###")
        if sim.source == "synthetic":
            print("  ⚠ synthetic — skip")
            continue
        lrv = _logrv_series(sim.opens, sim.highs, sim.lows, p)
        H = hurst_structure(lrv)
        print(f"  ROUGHNESS (log-RV structure function):")
        print(f"    ζ_q  = {['%.3f' % z for z in H['zeta']]}  at q={H['qs']}")
        print(f"    H_q  = {['%.3f' % h for h in H['H_per_q']]}")
        print(f"    H (pooled slope ζ_q vs q) = {H['H_pooled']:.3f}   "
              f"H_q2 = {H['H_q2']:.3f}")
        verdict = ("ROUGH ⇒ fractional vol justified" if H["H_pooled"] < 0.4
                   else "borderline" if H["H_pooled"] < 0.45
                   else "NOT rough ⇒ HAR suffices, skip rough-vol")
        print(f"    → {verdict}")
        rc = regime_check(p)
        print(f"  REGIMES:")
        print(f"    heuristic persistence (diag) = {rc['heuristic_persistence']:.3f}")
        print(f"    vol-state persistence (diag) = {rc['volstate_persistence']:.3f}")
        print(f"    next |r| by vol-state        = "
              f"{ {k: round(v, 5) for k, v in rc['next_absret_by_volstate'].items()} }")
        print(f"    separation (hi/lo)           = {rc['separation_ratio']:.2f}x")
        print(f"    → {'states separate & persist ⇒ HMM worthwhile' if rc['separation_ratio'] > 1.5 and rc['volstate_persistence'] > 0.45 else 'weak — HMM may not beat heuristic'}")


if __name__ == "__main__":
    main()
