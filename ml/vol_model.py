"""Volatility forecasting — the one head with a *genuine* out-of-sample edge.

The candle-direction model is structurally capped at ≤ 0 edge on 1-bar moves
(see ``ANALYSE_CRITIQUE_MODELE.md`` §1-2: the direction of a single bar is
≈ 50/50 on real data as on the GBM substrate). The *magnitude* of a bar — its
volatility — is the opposite: it **clusters** (calm follows calm, storm follows
storm), so next-bar σ is genuinely forecastable. Every scenario cone, every
Monte-Carlo path spread and every probability weight in ``core/scenarios.py`` is
sized by a single σ, so a sharper σ improves the calibration of the *whole* bot.

This module collects several one-step-ahead σ forecasters and a learned one,
all **look-ahead-free** (a forecast for bar *t* uses only bars < *t*):

* ``close_ewma_sigma`` — RiskMetrics EWMA on squared close-to-close returns.
  This is the model's *previous* best (``MarketState.ewma_volatility``); it is
  the bar to beat.
* ``range_ewma_sigma`` — the same EWMA recursion, but the per-bar innovation is
  a **range-based** variance estimate (Parkinson / Garman-Klass / Rogers-
  Satchell) computed from the bar's open/high/low/close. The intrabar high-low
  range is a far less noisy proxy of a bar's variance than a single squared
  close-return (5-8× more statistically efficient), so it gives a cleaner state
  estimate to roll forward — *provided the high/low are real* (they are, on the
  cached Binance data; the synthetic GBM fallback fabricates them and is gated
  out in :func:`real_ohlc`).
* :class:`GarchModel` — GARCH(1,1) by variance-targeting MLE (numpy-only, no
  scipy). Adds mean-reversion to a long-run variance that EWMA (an IGARCH with
  ω=0, α+β=1) lacks.
* :class:`HARVol` — a *learned* Heterogeneous-AutoRegressive model (Corsi 2009)
  on **log** realised variance, regressing next-bar log-var on the average
  log-var over the last 1 / 5 / 22 bars. Captures the long-memory of volatility
  that a single EWMA decay cannot.

The winner is chosen empirically by walk-forward NLL/QLIKE in
``tests/measure_model.py`` — not asserted here. :class:`VolForecaster` wraps the
chosen method with a robust fallback chain so the live pipeline always gets a
finite σ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# σ floor (per-bar log-return units). Volatility never collapses to zero on a
# flat patch; mirrors the ``max(…, 0.003)`` floor the scenario engine used.
SIGMA_FLOOR = 1e-4
_LAM = 0.94          # RiskMetrics decay
_2LN2 = 2.0 * math.log(2.0)
# Per-tick recursions seed at the long-run variance and are contractive (β<1),
# so the seed's influence vanishes within a handful of bars. Capping the live
# recursion to the most recent bars keeps the TUI hot-path O(1) in history
# length without changing the forecast (400 ≫ any realistic mixing time).
_RECURSION_CAP = 400


# --------------------------------------------------------------------------- #
# Real-vs-synthetic OHLC gate                                                  #
# --------------------------------------------------------------------------- #
def real_ohlc(opens, highs, lows, closes) -> bool:
    """True iff high/low carry genuine intrabar information.

    ``core/market.py`` fabricates OHLC from closes when a feed has none
    (``high = max(o,c) + 0.35·|c−o|``). On that synthetic range a Parkinson
    estimate is a deterministic function of the closes and adds *nothing* — so
    the range forecasters must fall back to the close-based EWMA. This detects
    the fabrication formula (and degenerate H==L) cheaply on a recent window.
    """
    if opens is None or highs is None or lows is None or closes is None:
        return False
    o = np.asarray(opens, dtype=np.float64)[-200:]
    h = np.asarray(highs, dtype=np.float64)[-200:]
    low = np.asarray(lows, dtype=np.float64)[-200:]
    c = np.asarray(closes, dtype=np.float64)[-200:]
    n = min(len(o), len(h), len(low), len(c))
    if n < 8:
        return False
    o, h, low, c = o[-n:], h[-n:], low[-n:], c[-n:]
    if np.any(h <= 0) or np.any(low <= 0) or np.any(h < low):
        return False
    span = np.abs(c - o)
    synth_h = np.maximum(o, c) + 0.35 * span
    synth_l = np.minimum(o, c) - 0.35 * span
    if np.allclose(h, synth_h, rtol=1e-6, atol=1e-9) and np.allclose(low, synth_l, rtol=1e-6, atol=1e-9):
        return False
    # Range must be non-trivially wider than the body for a decent fraction of
    # bars, otherwise there is no intrabar information to exploit.
    body = np.maximum(span, 1e-12)
    wick = (h - low) - body
    return float(np.mean(wick > 0.1 * body)) > 0.3


# --------------------------------------------------------------------------- #
# Per-bar variance proxies (one number per bar, from that bar's OHLC)          #
# --------------------------------------------------------------------------- #
def parkinson_var(highs, lows) -> np.ndarray:
    """Parkinson (1980): σ² ≈ (ln(H/L))² / (4 ln 2). Uses the high-low range."""
    h = np.asarray(highs, dtype=np.float64)
    low = np.asarray(lows, dtype=np.float64)
    hl = np.log(np.maximum(h, 1e-12) / np.maximum(low, 1e-12))
    return (hl * hl) / (4.0 * math.log(2.0))


def garman_klass_var(opens, highs, lows, closes) -> np.ndarray:
    """Garman-Klass (1980): adds the open→close body to the high-low range."""
    o = np.asarray(opens, dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64)
    low = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    hl = np.log(np.maximum(h, 1e-12) / np.maximum(low, 1e-12))
    co = np.log(np.maximum(c, 1e-12) / np.maximum(o, 1e-12))
    v = 0.5 * hl * hl - (_2LN2 - 1.0) * co * co
    return np.maximum(v, 1e-12)


def rogers_satchell_var(opens, highs, lows, closes) -> np.ndarray:
    """Rogers-Satchell (1991): drift-independent (unbiased under a trend)."""
    o = np.asarray(opens, dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64)
    low = np.asarray(lows, dtype=np.float64)
    c = np.asarray(closes, dtype=np.float64)
    ho = np.log(np.maximum(h, 1e-12) / np.maximum(o, 1e-12))
    hc = np.log(np.maximum(h, 1e-12) / np.maximum(c, 1e-12))
    lo = np.log(np.maximum(low, 1e-12) / np.maximum(o, 1e-12))
    lc = np.log(np.maximum(low, 1e-12) / np.maximum(c, 1e-12))
    return np.maximum(ho * hc + lo * lc, 1e-12)


_RANGE_ESTIMATORS = {
    "parkinson": lambda o, h, l, c: parkinson_var(h, l),
    "garman_klass": garman_klass_var,
    "rogers_satchell": rogers_satchell_var,
}


# --------------------------------------------------------------------------- #
# One-step-ahead σ for the bar *after* the given (fully observed) arrays.      #
# This is exactly what both consumers want: the live engine sizes the next     #
# candle, and the walk-forward harness scores σ_t against the realised r_t by  #
# calling these on the prefix of bars strictly before t. No leak by design.    #
# --------------------------------------------------------------------------- #
def _ewma_next_var(innov: np.ndarray, lam: float = _LAM) -> float:
    """RiskMetrics next-step variance from a per-bar variance proxy series.

    ``innov[k]`` is bar *k*'s realised variance proxy (r_k² or a range estimate).
    Recurses ``v ← λ·v + (1-λ)·innov[k]`` over **all** proxies including the last
    one, so the result is the forecast for the *next* (unobserved) bar — the same
    convention as the legacy ``MarketState.ewma_volatility`` this replaces.
    """
    if len(innov) == 0:
        return SIGMA_FLOOR ** 2
    var = float(innov[0])
    for x in innov[1:]:
        var = lam * var + (1.0 - lam) * float(x)
    return max(var, SIGMA_FLOOR ** 2)


def close_ewma_next(closes, lam: float = _LAM) -> float:
    """Next-bar σ from squared close-to-close returns (RiskMetrics EWMA)."""
    c = np.asarray(closes, dtype=np.float64)
    if len(c) < 3:
        return SIGMA_FLOOR
    r = np.diff(np.log(c))
    return math.sqrt(_ewma_next_var(r * r, lam))


def range_ewma_next(opens, highs, lows, closes, lam: float = _LAM, estimator: str = "parkinson") -> float:
    """Next-bar σ from an EWMA over range-based per-bar variances.

    Falls back to the close-EWMA when the high/low are synthetic (no intrabar
    information to exploit).
    """
    c = np.asarray(closes, dtype=np.float64)
    if not real_ohlc(opens, highs, lows, c):
        return close_ewma_next(c, lam)
    vfun = _RANGE_ESTIMATORS.get(estimator, _RANGE_ESTIMATORS["parkinson"])
    bar_var = vfun(np.asarray(opens, dtype=np.float64),
                   np.asarray(highs, dtype=np.float64),
                   np.asarray(lows, dtype=np.float64), c)
    return math.sqrt(_ewma_next_var(bar_var, lam))


def flat_next(closes, window: int = 20) -> float:
    """Next-bar σ as the std of the last ``window`` close returns (naive)."""
    c = np.asarray(closes, dtype=np.float64)
    if len(c) < 3:
        return SIGMA_FLOOR
    r = np.diff(np.log(c))[-window:]
    return max(float(np.std(r)) if len(r) >= 2 else abs(float(r[-1])), SIGMA_FLOOR)


# --------------------------------------------------------------------------- #
# GARCH(1,1) — variance targeting, numpy-only grid MLE                         #
# --------------------------------------------------------------------------- #
@dataclass
class GarchModel:
    """GARCH(1,1): σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}, ω = (1-α-β)·Var(r).

    Variance targeting pins ω to the sample variance so only (α, β) are searched
    — robust on a few hundred bars and needs no external optimiser. α+β<1 gives
    the mean-reversion EWMA lacks.
    """

    alpha: float = 0.08
    beta: float = 0.90
    uncond_var: float = 1e-4
    trained: bool = False

    @classmethod
    def fit(cls, returns: np.ndarray) -> "GarchModel":
        r = np.asarray(returns, dtype=np.float64)
        r = r[np.isfinite(r)]
        if len(r) < 60:
            v = float(np.var(r)) if len(r) else 1e-4
            return cls(uncond_var=max(v, SIGMA_FLOOR ** 2))
        uncond = float(np.var(r))
        uncond = max(uncond, SIGMA_FLOOR ** 2)

        def nll(alpha: float, beta: float) -> float:
            omega = (1.0 - alpha - beta) * uncond
            if omega <= 0:
                return np.inf
            var = uncond
            total = 0.0
            for x in r:
                total += math.log(var) + (x * x) / var
                var = omega + alpha * (x * x) + beta * var
                if var <= 1e-300:
                    return np.inf
            return 0.5 * total

        best = (np.inf, 0.08, 0.90)
        # Coarse grid over the stationary simplex, then a local refine.
        for a in np.linspace(0.01, 0.4, 14):
            for b in np.linspace(0.5, 0.98, 14):
                if a + b >= 0.999:
                    continue
                v = nll(a, b)
                if v < best[0]:
                    best = (v, a, b)
        _, a0, b0 = best
        for a in np.linspace(max(0.001, a0 - 0.04), a0 + 0.04, 9):
            for b in np.linspace(max(0.3, b0 - 0.04), min(0.985, b0 + 0.04), 9):
                if a + b >= 0.999:
                    continue
                v = nll(a, b)
                if v < best[0]:
                    best = (v, a, b)
        _, a, b = best
        return cls(alpha=float(a), beta=float(b), uncond_var=uncond, trained=True)

    def forecast_next(self, returns: np.ndarray) -> float:
        """Next-bar σ: recurse the GARCH variance over recent observed returns."""
        r = np.asarray(returns, dtype=np.float64)[-_RECURSION_CAP:]
        omega = (1.0 - self.alpha - self.beta) * self.uncond_var
        var = self.uncond_var
        for x in r:
            var = omega + self.alpha * (x * x) + self.beta * var
        return math.sqrt(max(var, SIGMA_FLOOR ** 2))


@dataclass
class RealizedGarch:
    """GARCH(1,1) driven by a **range-based** realised variance (Realized GARCH,
    Hansen-Huang-Shek 2012, simplified): σ²_t = ω + α·RV_{t-1} + β·σ²_{t-1}.

    The α term innovates on the Parkinson per-bar variance ``RV`` (an efficient
    intrabar estimate) instead of a single squared close-return, while β keeps
    GARCH's mean-reversion. So it unites the two effects that each won on one
    asset above (range-efficiency on ETH, mean-reversion on BTC). ``RV`` is
    rescaled on the fit window so its mean matches the close-return variance the
    likelihood is computed against — no scale bias.
    """

    alpha: float = 0.10
    beta: float = 0.85
    uncond_var: float = 1e-4
    scale: float = 1.0
    trained: bool = False

    @staticmethod
    def _aligned_rv(highs, lows, closes) -> np.ndarray:
        """Parkinson variance per bar, aligned to close-to-close returns:
        ``rv[j]`` is the range variance of the same bar whose return is
        ``diff(log(closes))[j]`` — i.e. drop bar 0 (which has no return)."""
        rv = parkinson_var(np.asarray(highs, dtype=np.float64),
                           np.asarray(lows, dtype=np.float64))
        return rv[1:]

    @classmethod
    def fit(cls, opens, highs, lows, closes) -> "RealizedGarch":
        c = np.asarray(closes, dtype=np.float64)
        r = np.diff(np.log(c))
        r = r[np.isfinite(r)]
        if len(r) < 60 or not real_ohlc(opens, highs, lows, c):
            return cls(uncond_var=max(float(np.var(r)) if len(r) else 1e-4, SIGMA_FLOOR ** 2))
        rv = cls._aligned_rv(highs, lows, c)
        m = min(len(r), len(rv))
        r, rv = r[-m:], rv[-m:]
        uncond = max(float(np.mean(r * r)), SIGMA_FLOOR ** 2)
        scale = uncond / max(float(np.mean(rv)), 1e-12)   # match RV mean to r² mean
        rv_s = rv * scale

        def nll(alpha: float, beta: float) -> float:
            omega = (1.0 - alpha - beta) * uncond
            if omega <= 0:
                return np.inf
            var = uncond
            total = 0.0
            for j in range(len(r)):
                total += math.log(var) + (r[j] * r[j]) / var
                var = omega + alpha * rv_s[j] + beta * var
                if var <= 1e-300:
                    return np.inf
            return 0.5 * total

        best = (np.inf, 0.10, 0.85)
        for a in np.linspace(0.02, 0.6, 16):
            for b in np.linspace(0.3, 0.95, 16):
                if a + b >= 0.999:
                    continue
                v = nll(a, b)
                if v < best[0]:
                    best = (v, a, b)
        _, a0, b0 = best
        for a in np.linspace(max(0.005, a0 - 0.05), a0 + 0.05, 9):
            for b in np.linspace(max(0.2, b0 - 0.05), min(0.985, b0 + 0.05), 9):
                if a + b >= 0.999:
                    continue
                v = nll(a, b)
                if v < best[0]:
                    best = (v, a, b)
        _, a, b = best
        return cls(alpha=float(a), beta=float(b), uncond_var=uncond, scale=scale, trained=True)

    def forecast_next(self, opens, highs, lows, closes) -> float:
        c = np.asarray(closes, dtype=np.float64)
        r = np.diff(np.log(c))
        if not self.trained:
            return close_ewma_next(c)
        rv = self._aligned_rv(highs, lows, c) * self.scale
        m = min(len(r), len(rv))
        r, rv = r[-m:][-_RECURSION_CAP:], rv[-m:][-_RECURSION_CAP:]
        omega = (1.0 - self.alpha - self.beta) * self.uncond_var
        var = self.uncond_var
        for j in range(len(rv)):
            var = omega + self.alpha * rv[j] + self.beta * var
        return math.sqrt(max(var, SIGMA_FLOOR ** 2))


# --------------------------------------------------------------------------- #
# HAR-RV — learned log-variance regression (Corsi 2009)                        #
# --------------------------------------------------------------------------- #
@dataclass
class HARVol:
    """Heterogeneous AutoRegressive model on **log** realised variance.

    Features at bar *t*: the mean log realised variance over the last 1, 5 and
    22 bars (the "daily / weekly / monthly" cascade); target: log var of bar
    *t+1*. Fit by OLS on log-variance (≈ homoskedastic, robust). The realised
    variance proxy is range-based when the OHLC is real, else squared returns.
    """

    coef: np.ndarray = field(default_factory=lambda: np.zeros(4))
    lags: tuple[int, int, int] = (1, 5, 22)
    trained: bool = False

    @staticmethod
    def _logvar_proxy(opens, highs, lows, closes) -> np.ndarray:
        c = np.asarray(closes, dtype=np.float64)
        if real_ohlc(opens, highs, lows, c):
            v = parkinson_var(np.asarray(highs, dtype=np.float64),
                              np.asarray(lows, dtype=np.float64))
        else:
            r = np.diff(np.log(c))
            v = np.concatenate([[float(np.var(r[:5])) if len(r) >= 5 else 1e-4], r * r])
        return np.log(np.maximum(v, 1e-12))

    def _features(self, lv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        l1, l5, l22 = self.lags
        rows, idx = [], []
        for t in range(max(self.lags), len(lv) - 1):
            rows.append([
                1.0,
                lv[t],                       # last bar
                float(np.mean(lv[t - l5 + 1:t + 1])),    # last week
                float(np.mean(lv[t - l22 + 1:t + 1])),   # last month
            ])
            idx.append(t)
        return np.asarray(rows, dtype=np.float64), np.asarray(idx, dtype=np.int64)

    @classmethod
    def fit(cls, opens, highs, lows, closes) -> "HARVol":
        m = cls()
        lv = m._logvar_proxy(opens, highs, lows, closes)
        if len(lv) < max(m.lags) + 30:
            return m
        X, idx = m._features(lv)
        y = lv[idx + 1]
        # Ridge-stabilised OLS (tiny λ) — guards a near-singular design.
        XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
        m.coef = np.linalg.solve(XtX, X.T @ y)
        m.trained = True
        return m

    def forecast_next(self, opens, highs, lows, closes) -> float:
        """Next-bar σ from the HAR cascade evaluated at the last observed bar."""
        c = np.asarray(closes, dtype=np.float64)
        if not self.trained:
            return close_ewma_next(c)
        lv = self._logvar_proxy(opens, highs, lows, c)
        j = len(lv) - 1
        if j < max(self.lags):
            return close_ewma_next(c)
        l5, l22 = self.lags[1], self.lags[2]
        feat = np.array([
            1.0, lv[j],
            float(np.mean(lv[j - l5 + 1:j + 1])),
            float(np.mean(lv[j - l22 + 1:j + 1])),
        ])
        return max(math.exp(0.5 * float(feat @ self.coef)), SIGMA_FLOOR)


# --------------------------------------------------------------------------- #
# Unified forecaster used by the live pipeline                                 #
# --------------------------------------------------------------------------- #
_METHODS = ("rough", "rgarch", "garch", "range_ewma", "close_ewma", "har", "flat")

# Default method: the **rough-vol** RFSV forecaster (ml.rough_vol) now wins the
# walk-forward shoot-out outright — best NLL *and* QLIKE on BTC/ETH/SOL 1h,
# beating the previous champion Realized-GARCH by another ~1-2% NLL / ~5-8%
# QLIKE. This is the payoff of the roughness finding (H≈0.07, see
# tests/data_checks.py): the continuous power-law memory kernel beats HAR's
# 3-bucket cascade and GARCH's geometric β-decay. Re-measure with
# tests/measure_model.py before changing.
_DEFAULT_METHOD = "rough"


@dataclass
class VolForecaster:
    """One-step-ahead σ for the scenario engine, with a robust fallback chain.

    Two entry points:

    * :meth:`sigma` — stateless one-shot (fits fresh each call); used by tests
      and offline tools.
    * :meth:`sigma_for` — **cached** for the live loop: the only expensive part
      (the GARCH grid-MLE) is refit at most every ``refit_every`` bars and the
      fitted parameters are reused; each tick still re-applies the cheap O(n)
      variance recursion to the latest bar. This keeps the TUI responsive (the
      advisor's "refit every N bars, don't refit per tick").

    Either way σ is always finite and floored: a method that lacks the data it
    needs (too few bars, synthetic OHLC) degrades to the close-EWMA — the
    previous best — and finally to the floor.
    """

    method: str = _DEFAULT_METHOD
    lam: float = _LAM
    estimator: str = "parkinson"
    refit_every: int = 24
    # --- live cache (sigma_for) ---
    _model: object = field(default=None, repr=False)
    _fit_step: int = field(default=-10 ** 9, repr=False)
    _fit_key: tuple = field(default=(), repr=False)

    # -- stateless one-shot ---------------------------------------------- #
    def sigma(self, closes, opens=None, highs=None, lows=None) -> float:
        c = np.asarray(closes, dtype=np.float64)
        if len(c) < 6:
            return SIGMA_FLOOR
        try:
            return max(self._fit_and_apply(self.method, opens, highs, lows, c), SIGMA_FLOOR)
        except Exception:
            return max(close_ewma_next(c, self.lam), SIGMA_FLOOR)

    # -- cached, for the live engine ------------------------------------- #
    def sigma_for(self, state) -> float:
        """Cached one-step σ from a :class:`~core.market.MarketState`."""
        c = np.asarray(state.history, dtype=np.float64)
        if len(c) < 8:
            return max(close_ewma_next(c, self.lam), SIGMA_FLOOR)
        step = int(getattr(state, "step", len(c) - 1))
        end = step + 1
        o = state.opens[:end] if getattr(state, "opens", None) is not None else None
        h = state.highs[:end] if getattr(state, "highs", None) is not None else None
        low = state.lows[:end] if getattr(state, "lows", None) is not None else None
        key = (getattr(state, "symbol", ""), getattr(state, "source", ""))
        stale = (
            self._model is None or key != self._fit_key
            or step < self._fit_step or step - self._fit_step >= self.refit_every
        )
        try:
            if stale:
                self._model = self._fit(self.method, o, h, low, c)
                self._fit_step, self._fit_key = step, key
            return max(self._apply(self.method, self._model, o, h, low, c), SIGMA_FLOOR)
        except Exception:
            return max(close_ewma_next(c, self.lam), SIGMA_FLOOR)

    # -- method dispatch ------------------------------------------------- #
    @staticmethod
    def _fit(method, o, h, low, c):
        if method == "rough":
            from ml.rough_vol import RoughVol
            return RoughVol.fit(o, h, low, c)
        if method == "rgarch":
            return RealizedGarch.fit(o, h, low, c)
        if method == "garch":
            return GarchModel.fit(np.diff(np.log(c)))
        if method == "har":
            return HARVol.fit(o, h, low, c)
        return None  # the EWMA/flat methods carry no fitted state

    def _apply(self, method, model, o, h, low, c) -> float:
        if method == "rough" and model is not None:
            return model.forecast_next(o, h, low, c)
        if method == "rgarch" and model is not None:
            return model.forecast_next(o, h, low, c)
        if method == "garch" and model is not None:
            return model.forecast_next(np.diff(np.log(c)))
        if method == "har" and model is not None:
            return model.forecast_next(o, h, low, c)
        if method == "range_ewma" and real_ohlc(o, h, low, c):
            return range_ewma_next(o, h, low, c, self.lam, self.estimator)
        if method == "flat":
            return flat_next(c)
        return close_ewma_next(c, self.lam)

    def _fit_and_apply(self, method, o, h, low, c) -> float:
        return self._apply(method, self._fit(method, o, h, low, c), o, h, low, c)
