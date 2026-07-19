"""Volatility-adaptive direction threshold.

A single hardcoded ``|return| < 0.003 → flat`` band is wrong across timeframes:
on 1m bars almost every candle falls inside it (so a model scores ~100% by
always predicting "flat"), while on 1d bars it's far too tight. The honest band
scales with the data's own volatility, so "flat" means *small relative to how
this series actually moves* rather than an absolute number.

**Why a quantile and not ``k·σ``.** A fixed ``k·σ`` band does *not* deliver the
cross-regime robustness it promises: the flat *fraction* it captures depends on
the return distribution's kurtosis, which varies by asset/timeframe/regime.
Crypto bar-returns are sharply peaked (a fat spike near zero), so ``k=0.5·σ``
swallows ~60 % of bars into "flat" instead of the ~38 % a Gaussian would give —
the up/flat/down labels collapse to a flat-dominated plurality and the learned
model converges to predicting NEUTRE almost always. Defining the band as an
empirical *quantile* of ``|log-return|`` fixes the flat share by construction
(``flat`` = the smallest-magnitude ``FLAT_FRACTION`` of moves), so the three
classes stay balanced on every series regardless of tail shape. This is a
*relabelling* of what counts as a move — not class reweighting — so it is
orthogonal to (and does not contradict) the trainer's deliberate
``balance_classes=False`` calibration choice.
"""
from __future__ import annotations

import numpy as np

# Target share of bars labelled "flat": the smallest-|move| ``FLAT_FRACTION`` of
# the recent window. 0.34 ⇒ roughly symmetric thirds (up/flat/down), so neutral
# never dominates purely because returns are peaked. Holds across timeframes by
# construction (unlike a fixed k·σ, whose flat share drifts with kurtosis).
_FLAT_FRACTION = 0.34
_K_FALLBACK = 0.5  # k·σ band used only when |r| has no positive spread (degenerate)
_FLOOR = 3e-4   # never call everything flat, even on ultra-calm series
_CEIL = 5e-2    # never swallow real moves on very noisy series
_DEFAULT = 3e-3  # legacy value — used only when there isn't enough history


def adaptive_threshold(
    prices: np.ndarray | None,
    *,
    window: int = 200,
    flat_fraction: float = _FLAT_FRACTION,
) -> float:
    """Quantile flat band: the ``flat_fraction`` quantile of recent ``|log-return|``,
    clamped to ``[floor, ceil]``.

    A bar is "flat" when its move is in the smallest-magnitude ``flat_fraction``
    of the most recent ``window`` bars, so the flat class holds a fixed share of
    the data on every series (peaked or not). Falls back to ``k·σ`` if the
    returns are degenerate (no positive spread) and to the legacy ``0.003`` when
    there isn't enough history.
    """
    if prices is None:
        return _DEFAULT
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 5:
        return _DEFAULT
    r = np.diff(np.log(p[-(window + 1):]))
    if len(r) < 4:
        return _DEFAULT
    # Dispersion around the *median*, not around zero: a band measured on ``|r|``
    # would conflate drift with volatility — in a steep trend ``|r|`` is large
    # because of the drift, so the flat band would balloon and label genuine
    # directional bars as flat. Centring on the median strips the drift, so the
    # band tracks true fluctuation (≈ ``|r|`` on real crypto where drift≈0).
    a = np.abs(r - np.median(r))
    thr = float(np.quantile(a, flat_fraction))
    if not np.isfinite(thr) or thr <= 0.0:
        # Degenerate window (many exactly-zero returns): fall back to k·σ.
        sigma = float(np.std(r))
        if not np.isfinite(sigma) or sigma <= 0.0:
            return _DEFAULT
        thr = _K_FALLBACK * sigma
    return float(min(_CEIL, max(_FLOOR, thr)))


def classify_idx(ret: float, threshold: float) -> int:
    """Direction class index for a return: 0=up, 1=flat, 2=down."""
    if ret > threshold:
        return 0
    if ret < -threshold:
        return 2
    return 1


def classify_dir(ret: float, threshold: float) -> str:
    """Direction label for a return: 'up' | 'flat' | 'down'."""
    if ret > threshold:
        return "up"
    if ret < -threshold:
        return "down"
    return "flat"
