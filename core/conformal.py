"""Conformal prediction — distribution-free, self-calibrating return intervals.

The cone (``core/cone.py``) gives a *model-based* predictive distribution; its
intervals are only as calibrated as the model. Conformal prediction adds a layer
with a **finite-sample, distribution-free coverage guarantee**: wrap any point/
scale forecast, and the conformal interval covers the truth at the nominal rate
*whatever* the model's misspecification — the guarantee comes from the exchange-
ability of the calibration residuals, not from the model being right. That is
exactly the honesty the project already chases with temperature scaling and
Brier, taken to its logical end: a stated "90 % interval" really contains the
next return ~90 % of the time.

Two ideas, both implemented here:

* **Split conformal (Vovk; Lei et al. 2018).** Nonconformity score
  ``s_t = |r_t − μ̂_t| / σ̂_t`` on a rolling calibration window; the interval is
  ``μ̂ ± Q_{1−α}(scores)·σ̂``. The ``(n+1)(1−α)/n`` quantile gives the finite-
  sample guarantee. Scaling by σ̂ (the rough-vol forecast) makes the score
  *normalised*, so the interval **breathes with volatility** instead of being a
  constant width — locally adaptive conformal (Lei et al.).

* **Adaptive Conformal Inference (Gibbs & Candès 2021).** Markets are not
  exchangeable (vol clusters, regimes drift), so static split-conformal coverage
  drifts. ACI tracks the realised miscoverage online and nudges the working level
  ``α_t ← α_t + γ·(α − errₜ)`` each bar, recovering the target coverage *over
  time* under distribution shift — the right tool for a live loop.

The layer is model-agnostic: feed it ``(μ̂, σ̂)`` from the cone / rough-vol head
and the realised return as each bar settles. Validated on empirical-vs-nominal
coverage in ``tests/measure_model.py`` (the only thing that proves a conformal
layer works). numpy-only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformalCalibrator:
    """Rolling split-conformal + ACI for symmetric return intervals.

    ``target_alpha`` is the desired miscoverage (0.1 → 90 % interval). ``window``
    is the calibration memory. ``gamma`` is the ACI learning rate (0 disables ACI
    → pure split conformal). The working level ``alpha_t`` is what's actually used
    each bar; ACI moves it to defend the *long-run* target coverage."""

    target_alpha: float = 0.1
    window: int = 300
    gamma: float = 0.02
    alpha_t: float = 0.1
    _scores: deque = field(default_factory=lambda: deque(maxlen=300), repr=False)
    n_seen: int = 0
    n_covered: int = 0

    def __post_init__(self) -> None:
        self.alpha_t = self.target_alpha
        self._scores = deque(maxlen=self.window)

    # -- the conformal quantile multiplier ------------------------------- #
    def _q(self, alpha: float) -> float:
        """Finite-sample conformal quantile of the normalised scores at level
        ``1−alpha``. With n calibration scores the rank is ⌈(n+1)(1−α)⌉; if that
        exceeds n (too little data / very high coverage) the guarantee needs an
        infinite band → fall back to a wide Gaussian multiplier."""
        n = len(self._scores)
        a = min(max(alpha, 1e-4), 0.999)
        if n < 20:
            # Not enough calibration yet: Gaussian z as a sane prior multiplier.
            from math import sqrt
            return float(sqrt(2.0)) * _erfinv(1.0 - a)
        rank = int(np.ceil((n + 1) * (1.0 - a)))
        if rank > n:
            return float(np.max(self._scores)) * 1.5  # widen past observed worst
        return float(np.sort(self._scores)[rank - 1])

    # -- public API ------------------------------------------------------ #
    def interval(self, mu: float, sigma: float, *, alpha: float | None = None) -> tuple[float, float]:
        """Conformal interval ``[lo, hi]`` for the next return. ``alpha=None``
        uses the ACI working level (the live default); pass an explicit alpha for
        an ad-hoc level (e.g. a 99 % band) off the same calibration scores."""
        a = self.alpha_t if alpha is None else alpha
        q = self._q(a)
        half = q * max(sigma, 1e-9)
        return mu - half, mu + half

    def update(self, realized: float, mu: float, sigma: float,
               interval: tuple[float, float] | None = None) -> dict:
        """Settle one bar: record the normalised nonconformity score and run the
        ACI update against the working level. Returns small telemetry.

        ``interval`` : l'intervalle réellement ÉMIS au moment de la prédiction.
        Le fournir garantit que la couverture juge la bande annoncée à l'usager
        — pas une bande recalculée au settle, qui divergerait dès qu'un second
        consommateur (ou un reset) mute l'état du calibrateur entre-temps."""
        s = abs(realized - mu) / max(sigma, 1e-9)
        if interval is not None:
            covered = interval[0] <= realized <= interval[1]
        else:
            # ACI: did the *previous* working interval cover? (pre-update q)
            q = self._q(self.alpha_t)
            covered = abs(realized - mu) <= q * max(sigma, 1e-9)
        err = 0.0 if covered else 1.0
        self.alpha_t = float(min(0.999, max(1e-3, self.alpha_t + self.gamma * (self.target_alpha - err))))
        self._scores.append(s)
        self.n_seen += 1
        self.n_covered += int(covered)
        return {"score": s, "alpha_t": self.alpha_t, "covered": covered,
                "running_coverage": self.n_covered / max(self.n_seen, 1)}

    def empirical_coverage(self) -> float:
        return self.n_covered / max(self.n_seen, 1)


def _erfinv(y: float) -> float:
    """Inverse error function (Winitzki approximation) — no scipy. Used only as
    the cold-start interval multiplier before enough calibration scores exist."""
    import math
    y = min(max(y, -0.999999), 0.999999)
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)
