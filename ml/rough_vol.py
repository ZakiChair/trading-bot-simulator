"""Rough volatility forecaster — RFSV (Gatheral-Jaisson-Rosenbaum 2018).

The data check (``tests/data_checks.py``) showed log realized-variance on this
crypto data is **decisively rough**: the Hurst exponent estimated from the
structure function ζ_q = q·H is H ≈ 0.06–0.08 (BTC/ETH/SOL 1h) — even rougher
than equities (the GJR paper's H ≈ 0.1). That is a strong, specific fact, and it
says something the existing vol stack only *approximates*: HAR-RV (Corsi 2009) is
a 3-bucket (1/5/22-bar) cascade that crudely mimics a power-law memory kernel.
A genuine rough-vol model uses the *continuous* power-law kernel directly.

Model (Rough Fractional Stochastic Volatility): ``log σ²_t`` is a fractional
Brownian motion with small Hurst H. The optimal forecast of future log-variance
under fBm is the Nuzman-Poor / GJR prediction formula (their eq. 5.1):

    E[log v_{t+Δ} | F_t] = (cos(Hπ)/π) · Δ^{H+½} · ∫₀^∞ log v_{t−u}
                                                   / ((u+Δ)·u^{H+½}) du

The kernel ``u^{-(H+½)}`` diverges as the lag u→0 (so the *most recent* bar
dominates — the signature of roughness) yet stays integrable because H+½ < 1.
We discretise it over past bars, normalise to a proper weighted mean (log-var is
stationary/mean-reverting here, so a normalised power-law mean is the robust
form), and exponentiate with a lognormal correction ``exp(m + ½ŝ²)`` — ŝ² being
the in-sample conditional variance of the one-step log-var residual — so the
output is a genuine *variance* forecast, not its median.

Two outputs the rest of the bot needs:
* :meth:`forecast_next` — next-bar σ, scored head-to-head against
  RealizedGARCH/HAR in the walk-forward shoot-out (``tests/measure_model.py``).
  It only earns its place if it wins there.
* :meth:`forecast_path_vars` — the *term structure* of per-bar variance over the
  whole horizon, which lets the Monte-Carlo cone widen along the rough-vol
  forecast instead of a single flat σ (the upgrade the scenario engine consumes).

No leak by construction: every forecast at bar t uses only ``log v`` of bars ≤ t.
numpy-only, no scipy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ml.vol_model import SIGMA_FLOOR, parkinson_var, real_ohlc

# Bounds on the estimated Hurst — crypto log-RV sits ~0.05-0.15; clamp so a noisy
# short window can't produce H≥½ (no roughness) or H≤0 (kernel blows up).
_H_LO, _H_HI = 0.02, 0.45
# How many past bars enter the power-law kernel. The kernel decays as
# u^{-(H+½)}·(u+Δ)^{-1} ≈ u^{-(H+3/2)} in the tail, so weight beyond a few
# hundred bars is negligible; cap keeps the live hot-path O(1) in history.
_KERNEL_LEN = 300


def estimate_hurst(logrv: np.ndarray, qs=(0.5, 1.0, 1.5, 2.0),
                   lags=range(1, 40)) -> float:
    """Hurst/roughness of a log realized-variance series via the GJR structure
    function. For a fractional process the q-th structure function scales as
    ``m(q,Δ) = E[|Δ_Δ log v|^q] ∝ Δ^{qH}``; pooling the per-q slopes ζ_q against
    q gives H. Clamped to ``[_H_LO, _H_HI]``."""
    x = np.asarray(logrv, dtype=np.float64)
    x = x[np.isfinite(x)]
    lags = list(lags)
    if len(x) < max(lags) + 10:
        return 0.1
    zeta = []
    for q in qs:
        m = [np.mean(np.abs(x[d:] - x[:-d]) ** q) for d in lags]
        m = np.asarray(m)
        good = m > 0
        if good.sum() < 3:
            return 0.1
        zeta.append(np.polyfit(np.log(np.asarray(lags)[good]), np.log(m[good]), 1)[0])
    H = float(np.polyfit(np.asarray(qs), np.asarray(zeta), 1)[0])
    if not np.isfinite(H):
        return 0.1
    return float(min(_H_HI, max(_H_LO, H)))


def _logrv_proxy(opens, highs, lows, closes) -> np.ndarray:
    """Per-bar log realized variance: range-based (Parkinson) when the OHLC is
    real, else squared close-returns. The series the roughness acts on."""
    c = np.asarray(closes, dtype=np.float64)
    if real_ohlc(opens, highs, lows, c):
        v = parkinson_var(np.asarray(highs, dtype=np.float64),
                          np.asarray(lows, dtype=np.float64))
    else:
        r = np.diff(np.log(c))
        seed = float(np.var(r[:20])) if len(r) >= 20 else (float(np.var(r)) if len(r) else 1e-4)
        v = np.concatenate([[seed], r * r])
    return np.log(np.maximum(v, 1e-12))


@dataclass
class RoughVol:
    """RFSV next-step (and term-structure) variance forecaster.

    ``fit`` estimates H from the structure function and the lognormal correction
    ŝ² from the one-step log-var residuals; the power-law kernel weights are then
    a pure function of (H, Δ). Designed to be *cheap* to refit (only H and a
    residual variance are estimated) and O(_KERNEL_LEN) to apply per tick.
    """

    hurst: float = 0.1
    s2: float = 0.0          # conditional variance of one-step log-var residual
    uncond_logvar: float = math.log(1e-4)
    trained: bool = False
    _kernel_len: int = field(default=_KERNEL_LEN, repr=False)
    # Variance résiduelle du log-var MESURÉE par horizon (k → ŝ²_k), estimée en
    # walk-forward in-sample au fit. L'ancienne extrapolation ŝ²·k^{2H} faisait
    # croître avec l'horizon la part *bruit de mesure du proxy RV* de ŝ² (qui est
    # constante en k) en plus de la part fBm (qui, elle, croît en k^{2H}) —
    # le cône 12 barres sur-couvrait (~×1.5 sur la variance réalisée).
    _s2_ks: np.ndarray | None = field(default=None, repr=False)
    _s2_by_h: np.ndarray | None = field(default=None, repr=False)

    # -- kernel ---------------------------------------------------------- #
    @staticmethod
    def _weights(H: float, delta: float, length: int) -> np.ndarray:
        """Normalised power-law forecast weights for horizon ``delta`` bars.

        ``w_j ∝ 1 / ((delta + u_j) · u_j^{H+½})`` with ``u_j = j − ½`` the lag of
        the j-th most recent bar (midpoint avoids the u=0 singularity). Index 0 =
        most recent bar. Normalised to a weighted mean (Σw=1)."""
        u = np.arange(length, dtype=np.float64) + 0.5          # 0.5, 1.5, ...
        w = 1.0 / ((delta + u) * np.power(u, H + 0.5))
        s = w.sum()
        return w / s if s > 0 else w

    # -- fit ------------------------------------------------------------- #
    @classmethod
    def fit(cls, opens, highs, lows, closes) -> "RoughVol":
        lv = _logrv_proxy(opens, highs, lows, closes)
        lv = lv[np.isfinite(lv)]
        if len(lv) < 60:
            uv = float(np.mean(lv)) if len(lv) else math.log(1e-4)
            return cls(uncond_logvar=uv)
        H = estimate_hurst(lv)
        length = min(cls._kernel_len, len(lv) - 1)
        # Per-horizon in-sample log-var residuals: forecast lv[t+k] from the
        # window ending at t, measure the residual variance per k → the
        # lognormal correction table ŝ²_k. Measuring each horizon directly
        # (instead of scaling ŝ²_1 by k^{2H}) keeps the constant-in-horizon
        # RV-proxy measurement noise out of the growth law.
        ks = np.array([1, 2, 3, 4, 6, 8, 12, 16, 20, 24], dtype=np.float64)
        ks = ks[ks < max(len(lv) - length, 2)]
        s2_by_h = []
        s2 = 0.0
        for k in ks:
            k = int(k)
            w = cls._weights(H, float(k), length)
            preds, actual = [], []
            idxs = range(length, len(lv) - k)
            stride = max(1, (len(lv) - k - length) // 400)
            for t in list(idxs)[::stride]:
                window = lv[t - length + 1:t + 1][::-1]        # most-recent first
                preds.append(float(window @ w))
                actual.append(float(lv[t + k]))
            resid = np.asarray(actual) - np.asarray(preds)
            v = float(np.var(resid)) if len(resid) > 5 else 0.0
            s2_by_h.append(v)
            if k == 1:
                s2 = v
        if not s2_by_h:
            return cls(uncond_logvar=float(np.mean(lv)))
        return cls(hurst=H, s2=s2, uncond_logvar=float(np.mean(lv)), trained=True,
                   _s2_ks=ks[:len(s2_by_h)], _s2_by_h=np.asarray(s2_by_h))

    # -- forecast -------------------------------------------------------- #
    def _forecast_logvar(self, lv: np.ndarray, delta: float) -> float:
        """E[log v_{t+delta} | F_t] from the normalised power-law kernel."""
        length = min(self._kernel_len, len(lv))
        if length < 2:
            return self.uncond_logvar
        w = self._weights(self.hurst, delta, length)
        window = lv[-length:][::-1]                            # most-recent first
        return float(window @ w)

    def forecast_next(self, opens, highs, lows, closes) -> float:
        """Next-bar σ (delta=1), lognormal-corrected to a true variance."""
        if not self.trained:
            from ml.vol_model import close_ewma_next
            return close_ewma_next(np.asarray(closes, dtype=np.float64))
        lv = _logrv_proxy(opens, highs, lows, closes)
        m = self._forecast_logvar(lv, 1.0)
        var = math.exp(m + 0.5 * self.s2)
        return math.sqrt(max(var, SIGMA_FLOOR ** 2))

    def forecast_path_vars(self, opens, highs, lows, closes, horizon: int) -> np.ndarray:
        """Term structure of per-bar *variance* forecasts for k=1..horizon.

        Each future bar k gets E[v_{t+k}] = exp(E[log v_{t+k}] + ½ŝ²_k), with the
        lognormal correction grown by the fBm prediction-variance scaling ŝ²_k ≈
        ŝ² · k^{2H} (uncertainty of the forecast widens with horizon). Lets the
        cone follow the rough-vol forecast instead of a single flat σ."""
        lv = _logrv_proxy(opens, highs, lows, closes)
        if not self.trained or len(lv) < 2:
            from ml.vol_model import close_ewma_next
            s = close_ewma_next(np.asarray(closes, dtype=np.float64))
            return np.full(horizon, s * s, dtype=np.float64)
        out = np.empty(horizon, dtype=np.float64)
        for k in range(1, horizon + 1):
            m = self._forecast_logvar(lv, float(k))
            out[k - 1] = max(math.exp(m + 0.5 * self._s2_at(k)), SIGMA_FLOOR ** 2)
        return out

    def _s2_at(self, k: int) -> float:
        """ŝ²_k : variance résiduelle du log-var à l'horizon k — interpolée de la
        table mesurée au fit (extrapolation plate au-delà). Repli sur la loi
        k^{2H} si la table n'existe pas (modèle non entraîné/hérité)."""
        if self._s2_ks is not None and self._s2_by_h is not None and len(self._s2_ks):
            return float(np.interp(float(k), self._s2_ks, self._s2_by_h))
        return self.s2 * (k ** (2.0 * self.hurst))
