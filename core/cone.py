"""Heavy-tailed, rough-volatility Monte-Carlo cone (path generation).

The legacy cone (``core/scenarios.py``) draws **Gaussian** shocks around a single
**flat** σ with uniform jitter. Two things are wrong with that for crypto:

* **Tails.** Crypto returns are strongly leptokurtic; a Gaussian cone
  systematically under-covers the 95/99 % tails (it thinks a −8 % hour is a
  10-sigma impossibility). Student-t innovations fix the shape.
* **Width term-structure.** A single flat σ ignores that volatility is *rough*
  and **mean-reverting at every scale** (H≈0.07, see ``tests/data_checks.py``):
  the forecastable per-bar variance has a term structure. ``ml.rough_vol`` gives
  it; the cone should widen along that curve, not a flat line.

This module builds the cone as a **parameterised ladder** so each ingredient can
be A/B-validated *before* it is wired into the engine:

    Gaussian-flat  →  Gaussian-rough  →  +Student-t  →  +leverage  →  +Hawkes

Invariants kept from the validated honesty work (``ANALYSE_CRITIQUE_MODELE.md``
§1-2, §8.4):

* **Centre = martingale (drift 0).** The 1-bar direction is ≈50/50 on real data;
  a drifted cone was measured to hurt both density and direction. We change the
  cone's *shape and width*, never re-introduce a directional centre.
* The Student-t innovation is **standardised to unit variance** (raw t_ν has
  variance ν/(ν−2)), so the t rung differs from the Gaussian rung *only* in tail
  shape at matched variance — otherwise "Student-t helps" would just mean "a
  wider cone helps".

numpy-only. The fit (ν, leverage) is cheap and meant to be cached at the engine's
``refit_every`` cadence next to the RoughVol fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ml.rough_vol import RoughVol
from ml.vol_model import SIGMA_FLOOR, close_ewma_next

# ν search grid for the Student-t MLE. ``inf`` = Gaussian limit. Floor at 2.5 so
# the variance ν/(ν−2) (hence the unit-variance rescale) stays finite & sane.
_NU_GRID = np.array([3, 3.5, 4, 5, 6, 8, 10, 15, 25, 1e9])
_NU_FLOOR = 3.0
# Leverage γ is clamped: drops raise vol, but a runaway multiplier would blow up
# the cone. Cap keeps exp(γ·cumret) bounded over a 12-bar horizon.
_LEV_MAX = 8.0


@dataclass
class ConeParams:
    """Which rung of the ladder. ``nu=inf`` ⇒ Gaussian; ``leverage=0`` ⇒ none;
    ``rough=False`` ⇒ flat σ (legacy width); ``hawkes`` ⇒ add clustered jumps."""

    nu: float = float("inf")
    leverage: float = 0.0
    rough: bool = True
    hawkes: bool = False


# --------------------------------------------------------------------------- #
# Standardised innovation draws                                               #
# --------------------------------------------------------------------------- #
def standardized_t(rng: np.random.Generator, nu: float, size) -> np.ndarray:
    """Unit-variance innovations. Gaussian when ``nu`` is ~inf, else a Student-t
    rescaled by ``√((ν−2)/ν)`` so Var=1 — tail shape changes, scale does not."""
    if not np.isfinite(nu) or nu >= 1e6:
        return rng.standard_normal(size)
    nu = max(nu, _NU_FLOOR)
    return rng.standard_t(nu, size=size) * math.sqrt((nu - 2.0) / nu)


def _causal_ewma_sigma(r: np.ndarray, lam: float = 0.97) -> np.ndarray:
    """Per-bar causal EWMA σ: σ̂_t forecasts bar t from returns < t (no leak).
    Smooth (high λ) on purpose — used only to standardise residuals for the ν
    MLE, where σ-noise would otherwise be mistaken for tail fatness."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    out = np.empty(n, dtype=np.float64)
    var = float(r[0] ** 2) if n else 1e-8
    for t in range(n):
        out[t] = math.sqrt(max(var, 1e-12))
        var = lam * var + (1.0 - lam) * float(r[t]) ** 2
    return out


def _student_t_nll(z: np.ndarray, nu: float) -> float:
    """Mean negative log-likelihood of unit-variance data ``z`` under a
    unit-variance Student-t(ν). Gaussian in the ν→∞ limit."""
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return float("inf")
    if not np.isfinite(nu) or nu >= 1e6:
        return float(0.5 * np.mean(z * z) + 0.5 * math.log(2 * math.pi))
    s2 = (nu - 2.0) / nu  # the unit-variance t has scale √s2 around a raw t_ν
    # f(z) = Γ((ν+1)/2)/(Γ(ν/2)√(νπ s2)) (1 + z²/(ν s2))^{-(ν+1)/2}
    c = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
         - 0.5 * math.log(nu * math.pi * s2))
    ll = c - 0.5 * (nu + 1) * np.log1p((z * z) / (nu * s2))
    return float(-np.mean(ll))


# --------------------------------------------------------------------------- #
# Cone model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class ConeModel:
    """Fits the tail (ν) and leverage (γ) of the innovation distribution and
    generates rough-vol-sized Monte-Carlo paths.

    ``fit`` standardises returns by the one-step rough-vol forecast and (a) grid-
    MLEs the Student-t ν on those residuals, (b) regresses next-bar log-variance
    on the current return to get the leverage slope. Both are data estimates with
    honest fallbacks; whether they *help the predictive cone* is decided by the
    A/B in ``tests/measure_model.py`` (CRPS / coverage / directional Brier), not
    asserted here."""

    nu: float = float("inf")
    leverage: float = 0.0
    _rough: RoughVol | None = field(default=None, repr=False)
    _hawkes: object = field(default=None, repr=False)
    _last_returns: np.ndarray | None = field(default=None, repr=False)
    _last_sigmas: np.ndarray | None = field(default=None, repr=False)
    trained: bool = False

    # -- fit ------------------------------------------------------------- #
    @classmethod
    def fit(cls, opens, highs, lows, closes) -> "ConeModel":
        c = np.asarray(closes, dtype=np.float64)
        r = np.diff(np.log(c))
        if len(r) < 80:
            return cls()
        rough = RoughVol.fit(opens, highs, lows, c)

        # Standardise returns for the ν MLE by a **smooth** causal EWMA σ, NOT the
        # sharp rough one-step forecast. The rough σ is deliberately jumpy (it is
        # tuned to track clustering for the cone width); dividing returns by a
        # jumpy σ̂ injects estimation noise that masquerades as fat tails (a
        # constant-vol Gaussian series would then look leptokurtic). A smoother σ
        # for *this* step decouples tail-shape estimation from σ-forecast noise,
        # so ν reflects genuine conditional fat tails. Last ≤1500 bars (ν is
        # stable; keeps the walk-forward refit bounded).
        r_win = r[-1500:]
        sig = _causal_ewma_sigma(r_win, lam=0.97)
        z = r_win / np.maximum(sig, 1e-12)
        z = z[np.isfinite(z)]
        if len(z) < 40:
            return cls(_rough=rough, trained=True)
        # Re-center & re-scale to unit variance before the ν MLE so the dof
        # captures *tail shape*, not a residual scale mismatch.
        z = (z - np.mean(z)) / (np.std(z) + 1e-12)
        nu = float(min(_NU_GRID, key=lambda nn: _student_t_nll(z, nn)))

        # Leverage: OLS slope of next-bar log realized variance on the current
        # return. Negative slope = drops precede higher vol → γ = −slope > 0.
        lv = np.log(np.maximum(r * r, 1e-12))
        m = min(len(r) - 1, len(lv) - 1)
        x = r[:m]
        y = lv[1:m + 1]
        if len(x) > 30 and np.std(x) > 1e-9:
            b = float(np.polyfit(x, y, 1)[0])
            lev = float(np.clip(-b, 0.0, _LEV_MAX))
        else:
            lev = 0.0

        # Hawkes jump process on vol-scaled returns. Detection σ is the smooth
        # causal EWMA (same as the ν step — a stable regime-relative scale).
        from core.hawkes import HawkesJumps
        sig_series = _causal_ewma_sigma(r, lam=0.94)
        hawkes = HawkesJumps.fit(r, sig_series, k=4.0)
        return cls(nu=nu, leverage=lev, _rough=rough, _hawkes=hawkes,
                   _last_returns=r[-500:].copy(),
                   _last_sigmas=sig_series[-500:].copy(), trained=True)

    # -- variance term structure ---------------------------------------- #
    def var_path(self, opens, highs, lows, closes, horizon: int, *, rough: bool) -> np.ndarray:
        """Per-bar variance forecast over the horizon. Rough term structure when
        ``rough`` (and a fitted RoughVol exists), else a flat one-step σ²."""
        c = np.asarray(closes, dtype=np.float64)
        if rough and self._rough is not None and self._rough.trained:
            vp = self._rough.forecast_path_vars(opens, highs, lows, c, horizon)
            return np.maximum(vp, SIGMA_FLOOR ** 2)
        sig = close_ewma_next(c) if self._rough is None else self._rough.forecast_next(opens, highs, lows, c)
        return np.full(horizon, max(sig, SIGMA_FLOOR) ** 2, dtype=np.float64)

    # -- path sampling --------------------------------------------------- #
    def sample_logpaths(
        self, rng: np.random.Generator, var_path: np.ndarray, n: int,
        params: ConeParams, jumps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cumulative log-return paths, shape ``(n, h)``. Martingale centre.

        Per bar k: ``r_k = z_k·√(var_path_k)·lev_mult_k`` with ``z_k`` a
        standardised innovation (Gaussian or unit-variance t). Leverage modulates
        each step's vol by ``exp(γ·(−C_{k-1}))`` where ``C`` is the path's prior
        cumulative return (a vectorised one-pass discretisation of the leverage
        effect: a drawdown so far inflates the next step's vol). Optional
        pre-sampled ``jumps`` (n,h) are added on top (Hawkes layer)."""
        h = len(var_path)
        sd = np.sqrt(var_path)[None, :]                       # (1,h)
        z = standardized_t(rng, params.nu, (n, h))
        base = z * sd                                         # (n,h) pre-leverage
        if params.leverage > 0:
            # Prior cumulative return before step k (shifted cumsum of base).
            cum_prior = np.concatenate([np.zeros((n, 1)), np.cumsum(base, axis=1)[:, :-1]], axis=1)
            mult = np.exp(np.clip(-params.leverage * cum_prior, -3.0, 3.0))
            steps = z * sd * mult
        else:
            steps = base
        if jumps is None and params.hawkes and self._hawkes is not None and getattr(self._hawkes, "trained", False):
            exc = self._hawkes.excitation_from(self._last_returns, self._last_sigmas) \
                if self._last_returns is not None else 0.0
            jumps = self._hawkes.simulate(rng, n, h, base_excitation=exc,
                                          sd_path=np.sqrt(var_path))
        if jumps is not None:
            steps = steps + jumps
        return np.cumsum(steps, axis=1)
