"""Measurement harness — honest before/after metrics for the candle model.

Run as:  PYTHONPATH=. .venv/bin/python tests/measure_model.py [SYMBOL] [TF]

Reports, on **real** data when available:
  * data source (synthetic ⇒ the whole premise is void — see ANALYSE_CRITIQUE)
  * single 20% holdout: val accuracy vs majority baseline, edge
  * Brier score (probability calibration) on the holdout
  * mean |return error| of the magnitude head (live_eval-style)
  * an expanding-window WALK-FORWARD CV: refit on bars 0..t, predict t+1's
    direction with NO leak, aggregated accuracy / edge / Brier over the tail.

The walk-forward number is the headline: it is the only out-of-sample estimate
that doesn't depend on a single arbitrary split.
"""
from __future__ import annotations

import copy
import math
import sys

import numpy as np

from core.market import load_market_simulator
from core.thresholds import adaptive_threshold, classify_idx
from ml.candle_model import build_dataset, candle_features, train_candle_model


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between the predicted
    probability vector and the one-hot truth (0 = perfect, lower is better)."""
    if len(y) == 0:
        return float("nan")
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _holdout_metrics(prices: np.ndarray, symbol: str, timeframe: str) -> dict:
    model, rep = train_candle_model(
        prices, symbol=symbol, timeframe=timeframe, epochs=400, lr=0.3
    )
    if not model.trained:
        return {"trained": False}

    # Rebuild the same chronological split the trainer used, score the val tail.
    X, y = build_dataset(prices)
    n = len(y)
    n_val = max(1, int(n * 0.2))
    Xva, yva = X[-n_val:], y[-n_val:]
    Xva_s = (Xva - model.feat_mean) / model.feat_std
    z = (Xva_s @ model.weights + model.bias) / model.temperature  # calibrated
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    pred = np.argmax(probs, axis=1)

    return {
        "trained": True,
        "n": n,
        "val_acc": float(np.mean(pred == yva)),
        "val_majority": rep.val_majority,
        "brier": _brier(probs, yva),
        "temperature": model.temperature,
        "class_counts": rep.class_counts,
        "final_loss": rep.final_loss,
        "train_acc": rep.train_accuracy,
    }


def _walk_forward(
    prices: np.ndarray,
    symbol: str,
    timeframe: str,
    *,
    n_test: int = 120,
    refit_every: int = 10,
) -> dict:
    """Expanding-window walk-forward: for the last ``n_test`` bars, refit on
    everything strictly before, predict the next bar's direction. No leak."""
    p = np.asarray(prices, dtype=np.float64)
    n = len(p)
    thr = adaptive_threshold(p)
    start = max(200, n - n_test - 1)
    preds, probs_list, truth = [], [], []
    model = None
    for t in range(start, n - 1):
        if model is None or (t - start) % refit_every == 0:
            model, _ = train_candle_model(
                p[: t + 1], symbol=symbol, timeframe=timeframe, epochs=250, lr=0.3
            )
            if not model.trained:
                continue
        pr = model.predict_proba(p[: t + 1])
        preds.append(int(np.argmax(pr)))
        probs_list.append(pr)
        truth.append(classify_idx(float(np.log(p[t + 1] / p[t])), thr))

    if not preds:
        return {"n_test": 0}
    preds_a = np.array(preds)
    truth_a = np.array(truth)
    probs_a = np.array(probs_list)
    counts = np.bincount(truth_a, minlength=3)
    baseline = float(counts.max() / len(truth_a))
    acc = float(np.mean(preds_a == truth_a))
    return {
        "n_test": len(preds),
        "wf_acc": acc,
        "wf_baseline": baseline,
        "wf_edge": acc - baseline,
        "wf_brier": _brier(probs_a, truth_a),
    }


def _online_vs_frozen_eval(
    prices: np.ndarray,
    symbol: str,
    timeframe: str,
    *,
    lr0: float = 0.05,
    anchor: float = 0.02,
    max_drift: float = 0.75,
    tau: float = 200.0,
) -> dict:
    """A/B the Live online learner against the frozen batch model, out-of-sample.

    Train one batch model, then walk the OOS tail (strictly after the model's
    training split). At each bar score BOTH the frozen model and an online copy
    on the realised next direction (Brier + NLL), THEN feed that settled candle
    to the online copy as one SGD step. Predict-before-update ⇒ no leak. The
    headline is the cumulative Brier/NLL delta: online "helps" only if it lowers
    them over a meaningful sample (≥100 settled bars). Both models are scored
    with their (identical) temperature, so it mirrors what the UI shows.
    """
    p = np.asarray(prices, dtype=np.float64)
    n = len(p)
    frozen, _ = train_candle_model(p, symbol=symbol, timeframe=timeframe, epochs=400, lr=0.3)
    if not frozen.trained:
        return {"n": 0}
    thr = frozen.dir_threshold
    start = max(frozen.train_end_step + 1, 205)
    if start >= n - 1:
        return {"n": 0}

    online = copy.deepcopy(frozen)
    online.online_lr0, online.online_anchor = lr0, anchor
    online.online_max_drift, online.online_tau = max_drift, tau
    online.ensure_anchor()

    bf = bo = nf = no = 0.0
    cf = co = m = 0
    for t in range(start, n - 1):
        sl = p[: t + 1]
        y = classify_idx(float(np.log(p[t + 1] / p[t])), thr)
        oh = np.eye(3)[y]
        pf = frozen.predict_proba(sl)
        po = online.predict_proba(sl)
        bf += float(np.sum((pf - oh) ** 2))
        bo += float(np.sum((po - oh) ** 2))
        nf += float(-math.log(max(float(pf[y]), 1e-12)))
        no += float(-math.log(max(float(po[y]), 1e-12)))
        cf += int(np.argmax(pf) == y)
        co += int(np.argmax(po) == y)
        m += 1
        online.online_update(online.features_asof(sl), y)  # settle → 1 SGD step

    if m == 0:
        return {"n": 0}
    return {
        "n": m, "lr0": lr0,
        "frozen_brier": bf / m, "online_brier": bo / m,
        "frozen_nll": nf / m, "online_nll": no / m,
        "frozen_acc": cf / m, "online_acc": co / m,
        "drift": online.online_drift(),
    }


def online_main() -> None:
    """Sweep the online learning rate across assets — honest online-vs-frozen A/B.

    Run:  PYTHONPATH=. .venv/bin/python tests/measure_model.py online [SYM ...]
    """
    assets = sys.argv[2:] or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    tf = "1h"
    grid = [0.003, 0.01, 0.03, 0.05]
    print("=== Online (Live self-reinforcement) A/B — online vs frozen, OOS tail ===")
    print("Brier/NLL lower = better; Δ<0 ⇒ online helps. dir 1-bar ≈ 50/50 ⇒ expect ~neutral.\n")
    agg = {lr: {"db": [], "dn": [], "da": []} for lr in grid}
    for sym in assets:
        sim = load_market_simulator(sym, tf)
        prices = sim.prices
        print(f"-- {sym} {tf}  [{sim.source}, {len(prices)} bars] --")
        if sim.source == "synthetic":
            print("   ⚠ synthetic — skipping (premise void)\n")
            continue
        print(f"   {'lr0':>6} {'n':>4}  {'Brier f→o(Δ)':>22}  {'NLL f→o(Δ)':>20}  "
              f"{'acc f→o(Δ)':>20}  {'drift':>5}")
        for lr in grid:
            r = _online_vs_frozen_eval(prices, sym, tf, lr0=lr)
            if not r.get("n"):
                print(f"   {lr:>6}  insufficient OOS")
                continue
            db = r["online_brier"] - r["frozen_brier"]
            dn = r["online_nll"] - r["frozen_nll"]
            da = r["online_acc"] - r["frozen_acc"]
            agg[lr]["db"].append(db)
            agg[lr]["dn"].append(dn)
            agg[lr]["da"].append(da)
            print(f"   {lr:>6} {r['n']:>4}  "
                  f"{r['frozen_brier']:.4f}→{r['online_brier']:.4f}({db:+.4f})  "
                  f"{r['frozen_nll']:.3f}→{r['online_nll']:.3f}({dn:+.3f})  "
                  f"{r['frozen_acc']:.3f}→{r['online_acc']:.3f}({da:+.3f})  {r['drift']:>5.2f}")
        print()
    print("=== mean Δ across assets (online − frozen; negative = online better) ===")
    print(f"{'lr0':>6} {'ΔBrier':>9} {'ΔNLL':>9} {'Δacc':>9}")
    for lr in grid:
        if not agg[lr]["db"]:
            continue
        print(f"{lr:>6} {float(np.mean(agg[lr]['db'])):>+9.4f} "
              f"{float(np.mean(agg[lr]['dn'])):>+9.4f} {float(np.mean(agg[lr]['da'])):>+9.4f}")
    print("\nHonest read: pick the lr0 with the best mean ΔBrier; if no lr0 clears "
          "frozen, online is ~neutral (regime-tracking/calibration, not alpha).")


def _flat_sigma(r: np.ndarray, window: int = 20) -> float:
    w = r[-window:]
    return float(np.std(w)) if len(w) else 0.01


def _ewma_sigma(r: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA volatility: σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}."""
    if len(r) == 0:
        return 0.01
    var = float(r[0] ** 2)
    for x in r[1:]:
        var = lam * var + (1.0 - lam) * float(x) ** 2
    return float(np.sqrt(max(var, 1e-12)))


def _vol_forecast_eval(prices: np.ndarray, *, n_test: int = 250) -> dict:
    """One-step-ahead volatility forecast quality: flat 20-window σ vs EWMA σ.

    Scored by Gaussian negative log-likelihood of the realised next return
    r_{t+1} ~ N(0, σ_t²) — a proper scoring rule, so a sharper σ that tracks
    volatility clustering wins. Also reports mean |·| band error. No leak: σ_t
    uses only returns up to bar t.
    """
    p = np.asarray(prices, dtype=np.float64)
    r = np.diff(np.log(p))
    n = len(r)
    if n < 60:
        return {"n_test": 0}
    start = max(40, n - n_test)

    def nll(sig: float, nxt: float) -> float:
        s2 = max(sig * sig, 1e-10)
        return 0.5 * (np.log(2 * np.pi * s2) + nxt * nxt / s2)

    flat_nll, ewma_nll, flat_abs, ewma_abs = [], [], [], []
    for t in range(start, n):
        hist = r[:t]
        nxt = float(r[t])
        sf, se = _flat_sigma(hist), _ewma_sigma(hist)
        flat_nll.append(nll(sf, nxt))
        ewma_nll.append(nll(se, nxt))
        flat_abs.append(abs(abs(nxt) - sf))
        ewma_abs.append(abs(abs(nxt) - se))
    return {
        "n_test": n - start,
        "flat_nll": float(np.mean(flat_nll)),
        "ewma_nll": float(np.mean(ewma_nll)),
        "flat_abs": float(np.mean(flat_abs)),
        "ewma_abs": float(np.mean(ewma_abs)),
    }


def _vol_models_eval(opens, highs, lows, closes, *, n_test: int = 250, refit_every: int = 25) -> dict:
    """Walk-forward shoot-out of every σ forecaster in ``ml.vol_model``.

    For each test bar *t*, every model forecasts σ_t from bars strictly before
    *t* (no leak; fitted models — GARCH, HAR — refit on the expanding prefix
    every ``refit_every`` bars). Each σ_t is scored against the realised
    close-to-close return r_t by three proper/robust rules, all lower-is-better:

    * **NLL**   — Gaussian negative log-likelihood of r_t ~ N(0, σ_t²). The
      unbiased headline: scored on the *close* return, so range models get no
      free lunch — they must forecast the close-move magnitude better.
    * **QLIKE** — r_t²/σ_t² − ln(r_t²/σ_t²) − 1, robust to the noisy r² proxy
      (Patton 2011): a consistent volatility ranking even with imperfect truth.
    * **MAE**   — mean |σ_t − |r_t||, the average band error (the metric on
      which EWMA *lost* to flat σ — so we watch it explicitly).
    """
    from ml.vol_model import (
        GarchModel, HARVol, RealizedGarch, close_ewma_next, flat_next,
        range_ewma_next, real_ohlc,
    )
    from ml.rough_vol import RoughVol

    c = np.asarray(closes, dtype=np.float64)
    n = len(c)
    if n < 80:
        return {"n_test": 0}
    o = None if opens is None else np.asarray(opens, dtype=np.float64)
    h = None if highs is None else np.asarray(highs, dtype=np.float64)
    low = None if lows is None else np.asarray(lows, dtype=np.float64)
    real = real_ohlc(o, h, low, c)
    start = max(60, n - n_test)

    names = ["flat", "close_ewma", "garch", "har", "rough"]
    if real:
        names = ["flat", "close_ewma", "park_ewma", "gk_ewma", "garch", "rgarch", "har", "rough"]
    acc = {nm: {"nll": [], "qlike": [], "mae": []} for nm in names}

    garch = har = rgarch = rough = None
    for t in range(start, n):
        co = c[:t]
        oo = None if o is None else o[:t]
        hh = None if h is None else h[:t]
        ll = None if low is None else low[:t]
        rco = np.diff(np.log(co))
        if garch is None or (t - start) % refit_every == 0:
            garch = GarchModel.fit(rco)
            har = HARVol.fit(oo, hh, ll, co)
            rough = RoughVol.fit(oo, hh, ll, co)
            if real:
                rgarch = RealizedGarch.fit(oo, hh, ll, co)
        sig = {
            "flat": flat_next(co),
            "close_ewma": close_ewma_next(co),
            "garch": garch.forecast_next(rco),
            "har": har.forecast_next(oo, hh, ll, co),
            "rough": rough.forecast_next(oo, hh, ll, co),
        }
        if real:
            sig["park_ewma"] = range_ewma_next(oo, hh, ll, co, estimator="parkinson")
            sig["gk_ewma"] = range_ewma_next(oo, hh, ll, co, estimator="garman_klass")
            sig["rgarch"] = rgarch.forecast_next(oo, hh, ll, co)

        r_t = float(np.log(c[t] / c[t - 1]))
        rt2 = max(r_t * r_t, 1e-12)
        for nm in names:
            s2 = max(sig[nm] ** 2, 1e-12)
            acc[nm]["nll"].append(0.5 * (math.log(2 * math.pi * s2) + rt2 / s2))
            acc[nm]["qlike"].append(rt2 / s2 - math.log(rt2 / s2) - 1.0)
            acc[nm]["mae"].append(abs(sig[nm] - abs(r_t)))

    out = {"n_test": n - start, "real_ohlc": real, "names": names}
    for nm in names:
        out[nm] = {k: float(np.mean(v)) for k, v in acc[nm].items()}
    return out


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf (no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _scenario_engine_eval(
    sim, *, horizon: int = 12, n_scenarios: int = 100, n_test: int = 150
) -> dict:
    """Walk-forward OOS quality of the ScenarioEngine's *terminal* distribution.

    The vol head (§8.3) was validated; the drift/direction head was never
    measured. This harness isolates exactly that. For each test bar *t* (state
    built from bars 0..t only — no leak), the engine produces a distribution of
    *h-bar* terminal returns. We score it against the realised terminal return
    ``R = P[t+h]/P[t] − 1`` two ways:

    * **Gaussian NLL** of R under the scenario distribution, comparing the
      **full drift** (mean μ = the MC cone's centre, shaped by momentum,
      regime_bias, REGIME_DRIFTS, DIR_DRIFT_SCALE) against a **martingale**
      (μ = 0) **at the identical σ** (the cone width). Holding σ fixed isolates
      the *drift*: if full-drift doesn't beat the martingale, the directional
      machinery is decorative (§1-2: direction is ≈50/50, so this is the
      expected, honest outcome).
    * **Directional Brier** of the predicted (P↑, P●, P↓) against the realised
      up/flat/down class, for the engine's **weighted** masses (what the UI
      shows) and the **equal-weight** MC path frequencies (the raw generative
      density), vs a **climatology** baseline (running empirical frequencies).

    σ and μ are the **equal-weight** empirical moments of the N sampled terminal
    returns: each MC path is an equally-likely draw from the generative process,
    so that empirical distribution *is* the predictive density. The per-scenario
    ``probability`` is a posterior re-weighting for display/selection, not the
    predictive spread — using it as σ collapses toward zero when the softmax
    concentrates and produces meaningless NLLs.
    """
    from core.scenarios import ScenarioEngine

    p = np.asarray(sim.prices, dtype=np.float64)
    n = len(p)
    if n < 260 + horizon:
        return {"n_test": 0}

    engine = ScenarioEngine(n_scenarios=n_scenarios, horizon=horizon,
                            timeframe=sim.timeframe)
    start = max(200, n - n_test - horizon)
    full_nll, mart_nll = [], []
    brier_w, brier_eq, brier_clim = [], [], []
    sign_hits = sign_n = 0
    # Running climatology counts (Laplace-smoothed), updated online → no leak.
    clim = np.ones(3)

    def _nll(mu: float, sig: float, x: float) -> float:
        s2 = max(sig * sig, 1e-10)
        return 0.5 * (math.log(2 * math.pi * s2) + (x - mu) ** 2 / s2)

    for t in range(start, n - horizon):
        state = sim.state_at(t)
        bundle = engine.generate(state)
        terms = np.array([s.terminal_return for s in bundle.scenarios])
        dirs = np.array([s.direction for s in bundle.scenarios])
        probs = np.array([s.probability for s in bundle.scenarios])
        probs = probs / max(probs.sum(), 1e-12)
        mu = float(terms.mean())          # equal-weight MC cone centre (drift)
        sig = float(terms.std())          # equal-weight MC cone width (vol head)

        realized = float(p[t + horizon] / p[t] - 1.0)
        thr = adaptive_threshold(state.history) * math.sqrt(horizon)

        # --- NLL: full drift vs martingale, identical σ ---
        full_nll.append(_nll(mu, sig, realized))
        mart_nll.append(_nll(0.0, sig, realized))

        # --- realised direction class (same band the engine classifies with) ---
        y = 0 if realized > thr else (2 if realized < -thr else 1)
        onehot = np.eye(3)[y]

        # softmax-weighted masses, computed directly from the scenario probs so
        # this stays the *weighted* baseline regardless of _FORECAST_EQUAL_WEIGHT.
        pw = np.array([probs[dirs == "up"].sum(), probs[dirs == "flat"].sum(),
                       probs[dirs == "down"].sum()])
        pw = pw / max(pw.sum(), 1e-12)
        brier_w.append(float(np.sum((pw - onehot) ** 2)))

        # equal-weight MC path frequencies (the raw generative directional call)
        peq = np.array([(dirs == "up").mean(), (dirs == "flat").mean(),
                        (dirs == "down").mean()])
        brier_eq.append(float(np.sum((peq - onehot) ** 2)))

        pc = clim / clim.sum()
        brier_clim.append(float(np.sum((pc - onehot) ** 2)))
        clim[y] += 1.0  # update AFTER scoring

        if y != 1:  # sign accuracy on non-flat realisations only
            sign_n += 1
            if (mu > 0) == (realized > 0):
                sign_hits += 1

    if not full_nll:
        return {"n_test": 0}
    fn, mn = np.array(full_nll), np.array(mart_nll)
    return {
        "n_test": len(full_nll),
        "horizon": horizon,
        "full_nll": float(fn.mean()),
        "mart_nll": float(mn.mean()),
        "full_nll_med": float(np.median(fn)),
        "mart_nll_med": float(np.median(mn)),
        "brier_w": float(np.mean(brier_w)),
        "brier_eq": float(np.mean(brier_eq)),
        "brier_clim": float(np.mean(brier_clim)),
        "sign_acc": (sign_hits / sign_n) if sign_n else float("nan"),
        "sign_n": sign_n,
    }


def _crps_empirical(samples: np.ndarray, y: float) -> float:
    """Empirical CRPS via the sorted closed form (same estimator for every cone
    variant, so the finite-N bias cancels in the ranking):

        CRPS = (1/N)Σ|xᵢ−y| − (1/(2N²))ΣΣ|xᵢ−xⱼ|

    with the pairwise term = (1/N²)Σᵢ(2i−N−1)x₍ᵢ₎ on the ascending sort."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(x)
    if n == 0:
        return float("nan")
    term1 = float(np.mean(np.abs(x - y)))
    i = np.arange(1, n + 1)
    term2 = float(np.sum((2 * i - n - 1) * x)) / (n * n)
    return term1 - 0.5 * term2


def _cone_ladder_eval(
    sim, *, horizon: int = 12, n_paths: int = 600, n_test: int = 250,
    refit_every: int = 25, seed: int = 7,
) -> dict:
    """A/B the Monte-Carlo cone up the ladder, walk-forward, no leak:

        gauss_flat → gauss_rough → t_rough → t_rough_lev

    Each rung samples ``n_paths`` *terminal* h-bar log-returns from bars ≤ t and
    is scored against the realised terminal log-return with the SAME estimators:

    * **CRPS** (proper scoring rule for the whole predictive density; lower=better)
    * **coverage** of the central 50/80/95/99 % intervals (empirical hit-rate vs
      nominal — fat tails should fix the 95/99 under-coverage)
    * **directional Brier** of the equal-weight up/flat/down masses vs climatology
      — the §8.4 honesty guard: leverage must not skew the cone into a directional
      tilt (drop it if direction regresses even when CRPS improves).
    """
    from core.cone import ConeModel, ConeParams

    p = np.asarray(sim.prices, dtype=np.float64)
    n = len(p)
    if n < 260 + horizon:
        return {"n_test": 0}
    o, h, l = sim.opens, sim.highs, sim.lows
    start = max(200, n - n_test - horizon)
    levels = [0.5, 0.8, 0.95, 0.99]
    rungs = {
        "gauss_flat": lambda m: ConeParams(nu=float("inf"), leverage=0.0, rough=False),
        "gauss_rough": lambda m: ConeParams(nu=float("inf"), leverage=0.0, rough=True),
        "t_rough": lambda m: ConeParams(nu=m.nu, leverage=0.0, rough=True),
        "t_rough_lev": lambda m: ConeParams(nu=m.nu, leverage=m.leverage, rough=True),
        "t_rough_hawkes": lambda m: ConeParams(nu=m.nu, leverage=0.0, rough=True, hawkes=True),
    }
    acc = {k: {"crps": [], "cov": {lv: [] for lv in levels},
               "brier": [], "var": []} for k in rungs}
    brier_clim = []
    clim = np.ones(3)
    model = None
    nus, levs = [], []

    for t in range(start, n - horizon):
        sl = slice(0, t + 1)
        oo = None if o is None else np.asarray(o)[sl]
        hh = None if h is None else np.asarray(h)[sl]
        ll = None if l is None else np.asarray(l)[sl]
        cc = p[: t + 1]
        if model is None or (t - start) % refit_every == 0:
            model = ConeModel.fit(oo, hh, ll, cc)
            nus.append(model.nu); levs.append(model.leverage)
        # var paths (flat & rough) computed once per bar, shared across rungs.
        vp_rough = model.var_path(oo, hh, ll, cc, horizon, rough=True)
        vp_flat = model.var_path(oo, hh, ll, cc, horizon, rough=False)
        realized = float(np.log(p[t + horizon] / p[t]))
        thr = adaptive_threshold(cc) * math.sqrt(horizon)
        y = 0 if realized > thr else (2 if realized < -thr else 1)
        onehot = np.eye(3)[y]
        rng = np.random.default_rng(seed + t)
        for name, mk in rungs.items():
            params = mk(model)
            vp = vp_rough if params.rough else vp_flat
            logpaths = model.sample_logpaths(rng, vp, n_paths, params)
            term = logpaths[:, -1]                       # terminal log-returns
            acc[name]["crps"].append(_crps_empirical(term, realized))
            acc[name]["var"].append(float(np.var(term)))
            for lv in levels:
                lo, hi = np.quantile(term, [(1 - lv) / 2, (1 + lv) / 2])
                acc[name]["cov"][lv].append(1.0 if lo <= realized <= hi else 0.0)
            # equal-weight directional masses from the same sample
            d_up = float(np.mean(term > thr)); d_dn = float(np.mean(term < -thr))
            peq = np.array([d_up, 1 - d_up - d_dn, d_dn])
            acc[name]["brier"].append(float(np.sum((peq - onehot) ** 2)))
        pc = clim / clim.sum()
        brier_clim.append(float(np.sum((pc - onehot) ** 2)))
        clim[y] += 1.0

    if not acc["t_rough"]["crps"]:
        return {"n_test": 0}
    out = {"n_test": len(acc["t_rough"]["crps"]), "horizon": horizon,
           "nu_med": float(np.median(nus)), "lev_med": float(np.median(levs)),
           "brier_clim": float(np.mean(brier_clim)), "rungs": list(rungs)}
    for name in rungs:
        out[name] = {
            "crps": float(np.mean(acc[name]["crps"])),
            "brier": float(np.mean(acc[name]["brier"])),
            "term_sd": float(np.sqrt(np.mean(acc[name]["var"]))),
            "cov": {lv: float(np.mean(acc[name]["cov"][lv])) for lv in levels},
        }
    return out


def _heuristic_regime_at(prices: np.ndarray, t: int, window: int = 20) -> str:
    """Causal replica of core.market._detect_regimes for the single bar t."""
    if t < window + 1:
        return "range"
    r = np.diff(np.log(prices[t - window:t + 1]))
    vol = float(np.std(r))
    mom = float((prices[t] - prices[t - window]) / prices[t - window])
    if vol > 0.025:
        return "volatile"
    if mom > 0.03:
        return "bull"
    if mom < -0.03:
        return "bear"
    return "range"


def _regime_eval(sim, *, n_test: int = 600, refit_every: int = 50,
                 fit_window: int = 4000) -> dict:
    """Walk-forward A/B: learned Gaussian-HMM regime vs the hardcoded heuristic.

    At each test bar t (causal — regime read from bars ≤ t), record the regime
    each model assigns and the realised NEXT-bar |return|. Two honest questions:

    * **Stability** — how often the label persists bar-to-bar (the heuristic
      whipsaws; see tests/data_checks.py). Higher = steadier regimes.
    * **Predictiveness** — does the regime separate next-bar volatility? We report
      (a) the between-regime share of next-|r| variance (ANOVA η², higher = the
      regime explains more of tomorrow's magnitude) and (b) a regime-conditional
      vol-forecast QLIKE: σ²(regime) = running mean realised var in that regime
      (causal), scored on r²_{t+1}, vs an unconditional baseline. Lower QLIKE and
      higher η² ⇒ the learned regime is genuinely more informative.
    """
    from core.hmm import RegimeHMM

    p = np.asarray(sim.prices, dtype=np.float64)
    o, h, l = sim.opens, sim.highs, sim.lows
    n = len(p)
    if n < 400:
        return {"n_test": 0}
    start = max(200, n - n_test - 1)
    hmm = None
    rows = []  # (hmm_regime, heur_regime, next_abs_r, next_r2)
    for t in range(start, n - 1):
        if hmm is None or (t - start) % refit_every == 0:
            s = max(0, t + 1 - fit_window)
            hmm = RegimeHMM.fit(
                None if o is None else np.asarray(o)[s:t + 1],
                None if h is None else np.asarray(h)[s:t + 1],
                None if l is None else np.asarray(l)[s:t + 1],
                p[s:t + 1],
            )
        if hmm is None:
            continue
        hr = hmm.regime_at(p[: t + 1],
                           None if h is None else np.asarray(h)[: t + 1],
                           None if l is None else np.asarray(l)[: t + 1])
        er = _heuristic_regime_at(p, t)
        nr = float(np.log(p[t + 1] / p[t]))
        rows.append((hr, er, abs(nr), nr * nr))
    if len(rows) < 50:
        return {"n_test": 0}

    def _persistence(labels):
        return float(np.mean([labels[i] == labels[i - 1] for i in range(1, len(labels))]))

    def _eta2(labels, vals):
        vals = np.asarray(vals)
        grand = vals.mean()
        ss_tot = float(np.sum((vals - grand) ** 2)) + 1e-12
        ss_bet = 0.0
        for g in set(labels):
            mask = np.array([x == g for x in labels])
            if mask.sum() > 0:
                ss_bet += mask.sum() * (vals[mask].mean() - grand) ** 2
        return float(ss_bet / ss_tot)

    def _cond_qlike(labels, r2):
        """Causal regime-conditional vol QLIKE: σ²(regime)=running mean r² in that
        regime up to t−1, scored on r²_t. Laplace-seeded with the global mean."""
        r2 = np.asarray(r2)
        glob = float(np.mean(r2))
        sums = {}
        cnts = {}
        out = []
        for g, v in zip(labels, r2):
            s2 = (sums.get(g, glob) + glob) / (cnts.get(g, 0) + 1)  # shrink to global
            s2 = max(s2, 1e-12)
            out.append(v / s2 - math.log(v / s2 + 1e-300) - 1.0 if v > 0 else 0.0)
            sums[g] = sums.get(g, 0.0) + v
            cnts[g] = cnts.get(g, 0) + 1
        return float(np.mean(out))

    hreg = [r[0] for r in rows]
    ereg = [r[1] for r in rows]
    absr = [r[2] for r in rows]
    r2 = [r[3] for r in rows]
    glob = float(np.mean(r2))
    uncond_qlike = float(np.mean([v / glob - math.log(v / glob + 1e-300) - 1.0
                                  for v in r2 if v > 0]))
    return {
        "n_test": len(rows),
        "hmm_persistence": _persistence(hreg),
        "heur_persistence": _persistence(ereg),
        "hmm_eta2": _eta2(hreg, absr),
        "heur_eta2": _eta2(ereg, absr),
        "hmm_qlike": _cond_qlike(hreg, r2),
        "heur_qlike": _cond_qlike(ereg, r2),
        "uncond_qlike": uncond_qlike,
        "hmm_regime_counts": {g: hreg.count(g) for g in set(hreg)},
        "heur_regime_counts": {g: ereg.count(g) for g in set(ereg)},
    }


def _conformal_eval(sim, *, n_test: int = 800, refit_every: int = 25) -> dict:
    """Walk-forward: do the conformal next-bar return intervals actually achieve
    their nominal coverage? For each level we run a ConformalCalibrator fed
    (μ=0, σ̂=rough-vol one-step) and compare empirical coverage to nominal, vs a
    raw Gaussian band z_{level}·σ̂ (what an uncalibrated model would emit). A good
    conformal layer tracks nominal tightly under volatility clustering; the
    Gaussian band typically under-covers the tails (fat tails) — the gap is the
    value the conformal layer adds. No leak: σ̂ and the conformal quantile use
    only bars < t; the bar is settled into the calibrator AFTER scoring."""
    from core.conformal import ConformalCalibrator, _erfinv
    from ml.rough_vol import RoughVol

    p = np.asarray(sim.prices, dtype=np.float64)
    o, h, l = sim.opens, sim.highs, sim.lows
    r = np.diff(np.log(p))
    n = len(p)
    if n < 400:
        return {"n_test": 0}
    start = max(200, n - n_test - 1)
    levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    cal = {lv: ConformalCalibrator(target_alpha=1 - lv, window=300, gamma=0.01) for lv in levels}
    conf_cov = {lv: [] for lv in levels}
    gauss_cov = {lv: [] for lv in levels}
    conf_w = {lv: [] for lv in levels}
    rough = None
    for t in range(start, n - 1):
        if rough is None or (t - start) % refit_every == 0:
            oo = None if o is None else np.asarray(o)[: t + 1]
            hh = None if h is None else np.asarray(h)[: t + 1]
            ll = None if l is None else np.asarray(l)[: t + 1]
            rough = RoughVol.fit(oo, hh, ll, p[: t + 1])
        oo = None if o is None else np.asarray(o)[: t + 1]
        hh = None if h is None else np.asarray(h)[: t + 1]
        ll = None if l is None else np.asarray(l)[: t + 1]
        sigma = rough.forecast_next(oo, hh, ll, p[: t + 1])
        realized = float(r[t])
        for lv in levels:
            lo, hi = cal[lv].interval(0.0, sigma)        # conformal (ACI) band
            conf_cov[lv].append(1.0 if lo <= realized <= hi else 0.0)
            conf_w[lv].append(hi - lo)
            z = math.sqrt(2.0) * _erfinv(lv)             # Gaussian band half-width
            gauss_cov[lv].append(1.0 if abs(realized) <= z * sigma else 0.0)
            cal[lv].update(realized, 0.0, sigma)         # settle AFTER scoring
    return {
        "n_test": len(conf_cov[0.9]),
        "levels": levels,
        "conf_cov": {lv: float(np.mean(conf_cov[lv])) for lv in levels},
        "gauss_cov": {lv: float(np.mean(gauss_cov[lv])) for lv in levels},
        "conf_width": {lv: float(np.mean(conf_w[lv])) for lv in levels},
    }


def _perf_metrics(log_returns: np.ndarray, bpy: float) -> dict:
    """Rendement total, vol annualisée, Sharpe, maxDD d'une série de log-returns."""
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) == 0:
        return {"ret_pct": 0.0, "vol_ann": 0.0, "sharpe": 0.0, "maxdd_pct": 0.0}
    eq = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))
    dd = (np.concatenate([[1.0], eq]) - peak) / peak
    sd = float(np.std(r))
    return {
        "ret_pct": float((eq[-1] - 1.0) * 100.0),
        "vol_ann": float(sd * math.sqrt(bpy)),
        "sharpe": float(np.mean(r) / sd * math.sqrt(bpy)) if sd > 0 else 0.0,
        "maxdd_pct": float(dd.min() * 100.0),
    }


def _risk_overlay_eval(sim, *, n_bars: int = 400, seed: int = 1) -> dict:
    """Walk-forward net de coûts de la politique de budget de risque
    (``core/decision.py``) contre deux benchmarks honnêtes sur les MÊMES barres :

    * **buy & hold** (exposition 1, coût d'entrée payé une fois) ;
    * **exposition constante = l'exposition moyenne réalisée de la stratégie** —
      le test juste : le sizing *dynamique* (rough-vol + régime HMM) apporte-t-il
      quelque chose face au sizing *statique* à exposition égale ? Comparer au
      seul B&H plein flatterait mécaniquement la stratégie (moins exposée).

    Métriques : rendement net, vol annualisée réalisée vs cible (le tracking est
    LE job du bot), maxDD, Sharpe, turnover et coûts. Déterministe : politique
    sans état appris, moteur seedé.
    """
    from config.market_config import WARMUP_BARS
    from core.bot import TradingBot
    from core.decision import bars_per_year

    engine = TradingBot(sim, n_scenarios=100, horizon=12)
    # Fenêtre = la QUEUE de l'historique (les n_bars dernières barres), pas le
    # début : démarrer à WARMUP mesurait les barres ~100-500 du cache — la zone
    # sur laquelle les têtes ont été réglées, et accessoirement la fenêtre la
    # plus favorable. Le chiffre du rapport doit venir de barres non touchées.
    tail_start = max(WARMUP_BARS, sim.n_bars - n_bars - 2)
    bot = engine.new_episode(episode=seed, start_step=tail_start)
    tf = getattr(sim, "timeframe", "1h")
    bpy = bars_per_year(tf)
    start_step = int(bot.market.step)
    limit = min(n_bars, sim.n_bars - start_step - 2)
    if limit < 30:
        return {}

    equity = [bot.portfolio.mark_to_market(bot.market.price)]
    exposures: list[float] = []
    steps = 0
    while steps < limit and engine.step(bot):
        equity.append(bot.portfolio.mark_to_market(bot.market.price))
        exposures.append(bot.portfolio.exposure(bot.market.price))
        steps += 1

    eq = np.asarray(equity, dtype=np.float64)
    r_strat = np.diff(np.log(eq))
    p = np.asarray(sim.prices[start_step:start_step + steps + 1], dtype=np.float64)
    r_mkt = np.diff(np.log(p))
    m = min(len(r_strat), len(r_mkt))
    r_strat, r_mkt = r_strat[:m], r_mkt[:m]
    avg_e = float(np.mean(exposures)) if exposures else 0.0
    cost_side = bot.portfolio.cost_per_side(0.0)

    # Benchmarks (coût d'entrée une fois, pas de rebalancement ensuite).
    arith = np.exp(r_mkt) - 1.0
    bh = np.log1p(arith)
    bh[0] += math.log1p(-cost_side)
    const = np.log1p(avg_e * arith)
    if len(const):
        const[0] += math.log1p(-avg_e * cost_side)

    strat = _perf_metrics(r_strat, bpy)
    strat.update({
        "avg_exposure": avg_e,
        "n_trades": len(bot.portfolio.trades),
        "turnover": len(bot.portfolio.trades) / max(steps, 1),
        "costs": float(bot.portfolio.fees_paid + bot.portfolio.slippage_paid),
        "target_vol": float(bot.risk.sigma_target_ann),
    })
    return {
        "bars": steps,
        "strat": strat,
        "buy_hold": _perf_metrics(bh, bpy),
        "const_exp": _perf_metrics(const, bpy),
    }


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"
    sim = load_market_simulator(symbol, timeframe)
    prices = sim.prices
    print(f"=== {symbol} {timeframe} ===")
    print(f"source         : {sim.source}   ({len(prices)} bars)")
    if sim.source == "synthetic":
        print("⚠ SYNTHETIC fallback — directional edge is impossible by construction.")
    thr = adaptive_threshold(prices)
    print(f"adaptive thr   : {thr:.5f}  (k·σ flat band)")

    ho = _holdout_metrics(prices, symbol, timeframe)
    if ho.get("trained"):
        print("\n-- 20% holdout --")
        print(f"samples        : {ho['n']}   class counts {ho['class_counts']}")
        print(f"train acc      : {ho['train_acc']:.3f}")
        print(f"val acc        : {ho['val_acc']:.3f}   vs majority {ho['val_majority']:.3f}"
              f"   (edge {ho['val_acc'] - ho['val_majority']:+.3f})")
        print(f"Brier (val)    : {ho['brier']:.4f}   (lower = better calibrated)"
              f"   T={ho['temperature']:.2f}")
        print(f"final loss     : {ho['final_loss']:.4f}")

    wf = _walk_forward(prices, symbol, timeframe)
    if wf.get("n_test"):
        print("\n-- walk-forward CV (expanding window, no leak) --")
        print(f"test bars      : {wf['n_test']}")
        print(f"WF acc         : {wf['wf_acc']:.3f}   vs baseline {wf['wf_baseline']:.3f}"
              f"   (edge {wf['wf_edge']:+.3f})")
        print(f"WF Brier       : {wf['wf_brier']:.4f}")

    vf = _vol_forecast_eval(prices)
    if vf.get("n_test"):
        d_nll = (vf["flat_nll"] - vf["ewma_nll"]) / abs(vf["flat_nll"]) * 100
        d_abs = (vf["flat_abs"] - vf["ewma_abs"]) / vf["flat_abs"] * 100
        print("\n-- volatility forecast: flat 20-window vs EWMA (magnitude head) --")
        print(f"test bars      : {vf['n_test']}")
        print(f"Gaussian NLL   : flat {vf['flat_nll']:.4f}  →  EWMA {vf['ewma_nll']:.4f}"
              f"   ({d_nll:+.1f}% {'better' if d_nll > 0 else 'worse'})")
        print(f"mean band err  : flat {vf['flat_abs']:.5f}  →  EWMA {vf['ewma_abs']:.5f}"
              f"   ({d_abs:+.1f}%)")

    vm = _vol_models_eval(
        getattr(sim, "opens", None), getattr(sim, "highs", None),
        getattr(sim, "lows", None), prices,
    )
    if vm.get("n_test"):
        print("\n-- volatility model shoot-out (walk-forward, no leak; lower = better) --")
        print(f"test bars      : {vm['n_test']}   real OHLC: {vm['real_ohlc']}")
        base = vm["close_ewma"]
        print(f"{'model':<12} {'NLL':>9} {'QLIKE':>8} {'MAE×1e3':>9}   vs close-EWMA")
        for nm in vm["names"]:
            m = vm[nm]
            d_nll = (base["nll"] - m["nll"]) / abs(base["nll"]) * 100
            tag = "—" if nm == "close_ewma" else f"NLL {d_nll:+.1f}%"
            star = "  ◀ best NLL" if m["nll"] == min(vm[x]["nll"] for x in vm["names"]) else ""
            print(f"{nm:<12} {m['nll']:>9.4f} {m['qlike']:>8.4f} {m['mae']*1e3:>9.4f}   {tag}{star}")

    se = _scenario_engine_eval(sim)
    if se.get("n_test"):
        print("\n-- scenario engine: terminal h-bar distribution (walk-forward, no leak) --")
        print(f"horizon        : {se['horizon']} bars   test bars: {se['n_test']}")
        d_nll = (se["mart_nll_med"] - se["full_nll_med"]) / abs(se["mart_nll_med"]) * 100
        print(f"NLL (median)   : full-drift {se['full_nll_med']:+.4f}  vs  "
              f"martingale {se['mart_nll_med']:+.4f}   ({d_nll:+.1f}%)")
        print(f"NLL (mean)     : full-drift {se['full_nll']:+.4f}  vs  "
              f"martingale {se['mart_nll']:+.4f}")
        verdict = "drift HELPS" if d_nll > 1.0 else (
            "drift HURTS" if d_nll < -1.0 else "drift ≈ neutral (martingale)")
        print(f"               → {verdict} (same σ; lower NLL = better)")
        print(f"dir-Brier      : weighted {se['brier_w']:.4f}  ·  "
              f"equal-wt MC {se['brier_eq']:.4f}  ·  climatology {se['brier_clim']:.4f}"
              f"   (lower = better)")
        print(f"sign acc (E[r] vs realised, non-flat): {se['sign_acc']:.3f}  (n={se['sign_n']})"
              "  [≈0.5 noise once drift=0: μ≈0, sign is undefined — not a signal]")

    cl = _cone_ladder_eval(sim, n_test=400)
    if cl.get("n_test"):
        print("\n-- cône Monte-Carlo: échelle gaussien→rough→Student-t→+levier→+Hawkes --")
        print(f"horizon {cl['horizon']}  test {cl['n_test']}  ν_med {cl['nu_med']:.1f}  "
              f"(CRPS↓ ; couverture cible = niveau)")
        print(f"{'rung':<15} {'CRPS':>9} {'Brier':>7} {'cov95':>6} {'cov99':>6}")
        for nm in cl["rungs"]:
            m = cl[nm]
            print(f"{nm:<15} {m['crps']:>9.5f} {m['brier']:>7.4f} "
                  f"{m['cov'][0.95]:>6.3f} {m['cov'][0.99]:>6.3f}")
        print("  → t_rough = défaut validé (queues+structure terme) ; levier DROP "
              "(régression directionnelle) ; Hawkes OFF (corrige 99% mais coûte CRPS)")

    cf = _conformal_eval(sim, n_test=500)
    if cf.get("n_test"):
        print("\n-- calibration conforme: couverture empirique vs nominale (next-bar) --")
        print(f"test {cf['n_test']}  (|err| = |empirique − nominal| ; conforme ≈ 0)")
        print(f"{'niveau':>7} {'conforme':>9} {'gaussien':>9}")
        for lv in cf["levels"]:
            print(f"{lv:>7.2f} {cf['conf_cov'][lv]:>9.3f} {cf['gauss_cov'][lv]:>9.3f}")

    rg = _regime_eval(sim, n_test=300, refit_every=60, fit_window=3000)
    if rg.get("n_test"):
        print("\n-- régime HMM (vol3) vs heuristique codée en dur (walk-forward) --")
        print(f"test {rg['n_test']}")
        print(f"persistance  HMM {rg['hmm_persistence']:.3f}  vs heur {rg['heur_persistence']:.3f}")
        print(f"η² next|r|   HMM {rg['hmm_eta2']:.4f}  vs heur {rg['heur_eta2']:.4f}  "
              "(HMM sépare mieux la magnitude ; descriptif/génératif, pas un edge directionnel)")
        print(f"HMM régimes  {rg['hmm_regime_counts']}  (équilibré vs heur {rg['heur_regime_counts']})")

    ro = _risk_overlay_eval(sim)
    if ro:
        s = ro["strat"]
        print("\n-- politique de budget de risque: walk-forward net de coûts "
              f"({ro['bars']} barres) --")
        print(f"{'stratégie':<22} {'ret %':>8} {'vol ann':>8} {'Sharpe':>7} {'maxDD %':>8}")
        print(f"{'risk-budget (bot)':<22} {s['ret_pct']:>8.2f} {s['vol_ann']:>8.1%}"
              f" {s['sharpe']:>7.2f} {s['maxdd_pct']:>8.2f}")
        for label, key in (("buy & hold", "buy_hold"),
                           (f"expo constante {s['avg_exposure']:.0%}", "const_exp")):
            m = ro[key]
            print(f"{label:<22} {m['ret_pct']:>8.2f} {m['vol_ann']:>8.1%}"
                  f" {m['sharpe']:>7.2f} {m['maxdd_pct']:>8.2f}")
        print(f"tracking vol   : réalisée {s['vol_ann']:.1%} vs cible {s['target_vol']:.0%}"
              f"   (LE job du bot — pas le sur-rendement)")
        print(f"exécution      : {s['n_trades']} trades ({s['turnover']:.3f}/barre)"
              f" · coûts {s['costs']:,.0f}$ (frais+slippage)")
        print("→ lecture honnête: pas d'edge directionnel revendiqué — le test juste est"
              " la ligne « expo constante » (sizing dynamique vs statique, exposition égale)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "online":
        online_main()
    else:
        main()
