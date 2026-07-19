"""Gaussian Hidden Markov Model for market regimes (Baum-Welch, numpy-only).

The legacy regime detector (``core.market._detect_regimes``) is a hand-set
threshold rule: ``vol > 0.025 → volatile``, ``mom > 0.03 → bull`` … with
hand-written transition matrices (``markov_model._DEFAULT_REGIME_TM``). The data
check (``tests/data_checks.py``) showed it whipsaws — its regime label persists
only ~0.63 of the time, while a genuine volatility state persists ~0.93. A
hardcoded box-classifier can't capture that a regime is a *latent, persistent*
state you only observe noisily through returns.

A Gaussian HMM is the textbook fix: K latent states, each emitting a Gaussian
over an observation vector (here ``[return, log realized-variance]`` — so states
separate along *both* drift sign and volatility level, recovering the
bull/bear/range/volatile structure), with a learned K×K transition matrix that
encodes persistence directly. We fit it by Baum-Welch (the EM algorithm for
HMMs), in **log space** (numerically stable forward-backward), numpy-only.

Causality for live use: :meth:`filter_state` runs the forward pass only, so the
regime probability at bar t uses bars ≤ t (no look-ahead) — the honest input for
a live forecast. :meth:`viterbi` (full-path decode) is for *display/analysis* of
already-closed history.

Whether the learned regimes are actually better than the heuristic — more
persistent AND more predictive of next-bar volatility — is decided by the
walk-forward A/B in ``tests/measure_model.py``, not asserted here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)
_VAR_FLOOR = 1e-8

# Regime label order — must match core.markov_model.REGIME_STATES /
# core.market.Regime so the learned transition matrix drops into the existing
# Markov regime sampler without remapping.
REGIME_ORDER = ("bull", "bear", "range", "volatile")
# Volatility-regime labels (the honest taxonomy: direction is 50/50 noise, the
# *volatility level* is the persistent, separable latent state — see
# tests/data_checks.py). Ordered low→high vol.
VOL_REGIME_ORDER = ("calm", "normal", "turbulent")


def _logsumexp(a: np.ndarray, axis=None):
    if axis is None:
        a = np.asarray(a, dtype=np.float64).ravel()
        m = np.max(a)
        if not np.isfinite(m):
            m = 0.0
        return float(m + np.log(np.sum(np.exp(a - m))))
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass
class GaussianHMM:
    """Diagonal-covariance Gaussian HMM fitted by Baum-Welch.

    State i emits ``x ~ N(means[i], diag(covars[i]))``. Diagonal covariance keeps
    the M-step closed-form and robust on the few thousand bars we have (a full
    covariance would over-parameterise). ``startprob``/``transmat`` are the usual
    HMM initial and transition distributions.
    """

    n_states: int
    n_dim: int
    startprob: np.ndarray = field(default=None)       # (K,)
    transmat: np.ndarray = field(default=None)        # (K,K)
    means: np.ndarray = field(default=None)           # (K,D)
    covars: np.ndarray = field(default=None)          # (K,D) diagonal variances
    trained: bool = False
    loglik: float = float("-inf")

    # -- emission log-prob ------------------------------------------------ #
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """(T,K) log N(x_t; μ_k, Σ_k) for every bar and state."""
        v = np.maximum(self.covars, _VAR_FLOOR)                     # (K,D)
        # −½[ D·ln2π + Σ_d ln v_kd + Σ_d (x_d−μ_kd)²/v_kd ]
        const = -0.5 * (self.n_dim * _LOG_2PI + np.sum(np.log(v), axis=1))  # (K,)
        # (T,K): squared-mahalanobis term
        diff = X[:, None, :] - self.means[None, :, :]              # (T,K,D)
        maha = np.sum((diff * diff) / v[None, :, :], axis=2)       # (T,K)
        return const[None, :] - 0.5 * maha

    # -- forward-backward in log space ----------------------------------- #
    def _forward_backward(self, log_e: np.ndarray):
        T, K = log_e.shape
        log_t = np.log(self.transmat + 1e-300)
        log_a = np.full((T, K), -np.inf)
        log_a[0] = np.log(self.startprob + 1e-300) + log_e[0]
        for t in range(1, T):
            # α_t(j) = e_t(j) · Σ_i α_{t-1}(i) T_ij
            log_a[t] = log_e[t] + _logsumexp(log_a[t - 1][:, None] + log_t, axis=0)
        log_b = np.full((T, K), -np.inf)
        log_b[-1] = 0.0
        for t in range(T - 2, -1, -1):
            log_b[t] = _logsumexp(log_t + (log_e[t + 1] + log_b[t + 1])[None, :], axis=1)
        ll = _logsumexp(log_a[-1])
        log_gamma = log_a + log_b - ll
        return log_a, log_b, log_gamma, log_t, ll

    # -- fit (Baum-Welch / EM) ------------------------------------------- #
    @classmethod
    def fit(cls, X: np.ndarray, n_states: int = 4, *, n_iter: int = 30,
            tol: float = 1e-3, seed: int = 0, n_init: int = 1) -> "GaussianHMM":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        T, D = X.shape
        best = None
        for init in range(max(1, n_init)):
            m = cls._single_fit(X, n_states, n_iter, tol, seed + init * 101)
            if best is None or m.loglik > best.loglik:
                best = m
        return best

    @classmethod
    def _single_fit(cls, X, K, n_iter, tol, seed) -> "GaussianHMM":
        T, D = X.shape
        rng = np.random.default_rng(seed)
        # k-means++-ish init: pick K spread-out points as initial means.
        idx = [int(rng.integers(T))]
        for _ in range(K - 1):
            d2 = np.min([np.sum((X - X[j]) ** 2, axis=1) for j in idx], axis=0)
            probs = d2 / max(d2.sum(), 1e-12)
            idx.append(int(rng.choice(T, p=probs)))
        means = X[idx].copy()
        covars = np.tile(np.var(X, axis=0) + _VAR_FLOOR, (K, 1))
        startprob = np.full(K, 1.0 / K)
        transmat = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(transmat, 0.9)

        self = cls(K, D, startprob, transmat, means, covars)
        prev_ll = -np.inf
        for _ in range(n_iter):
            log_e = self._log_emission(X)
            log_a, log_b, log_gamma, log_t, ll = self._forward_backward(log_e)
            gamma = np.exp(log_gamma)                              # (T,K)
            # ξ_t(i,j) ∝ α_t(i) T_ij e_{t+1}(j) β_{t+1}(j)
            log_xi = (log_a[:-1, :, None] + log_t[None, :, :]
                      + (log_e[1:] + log_b[1:])[:, None, :] - ll)
            xi = np.exp(log_xi).sum(axis=0)                        # (K,K)
            # M-step
            startprob = gamma[0] + 1e-8
            startprob /= startprob.sum()
            transmat = xi + 1e-8
            transmat /= transmat.sum(axis=1, keepdims=True)
            gsum = gamma.sum(axis=0) + 1e-8                        # (K,)
            means = (gamma.T @ X) / gsum[:, None]
            for k in range(K):
                diff = X - means[k]
                covars[k] = (gamma[:, k] @ (diff * diff)) / gsum[k]
            covars = np.maximum(covars, _VAR_FLOOR)
            self = cls(K, D, startprob, transmat, means, covars, True, float(ll))
            if ll - prev_ll < tol and prev_ll > -np.inf:
                break
            prev_ll = ll
        return self

    # -- inference ------------------------------------------------------- #
    def smoothed_states(self, X: np.ndarray) -> np.ndarray:
        """(T,) Viterbi-free smoothed most-likely state per bar (argmax γ).
        Uses the full series (past+future) — for analysis of closed history."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.n_dim:
            X = X.T
        _, _, log_gamma, _, _ = self._forward_backward(self._log_emission(X))
        return np.argmax(log_gamma, axis=1)

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """(T,) most-likely *path* (joint), the standard regime decode."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.n_dim:
            X = X.T
        log_e = self._log_emission(X)
        T, K = log_e.shape
        log_t = np.log(self.transmat + 1e-300)
        delta = np.log(self.startprob + 1e-300) + log_e[0]
        back = np.zeros((T, K), dtype=np.int64)
        for t in range(1, T):
            m = delta[:, None] + log_t                            # (K,K)
            back[t] = np.argmax(m, axis=0)
            delta = log_e[t] + np.max(m, axis=0)
        path = np.zeros(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        return path

    def filter_state(self, X: np.ndarray) -> np.ndarray:
        """(K,) causal filtered state distribution at the LAST bar — forward pass
        only, so it uses bars ≤ t (no look-ahead). The live regime input."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.n_dim:
            X = X.T
        log_e = self._log_emission(X)
        T, K = log_e.shape
        log_t = np.log(self.transmat + 1e-300)
        log_a = np.log(self.startprob + 1e-300) + log_e[0]
        for t in range(1, T):
            log_a = log_e[t] + _logsumexp(log_a[:, None] + log_t, axis=0)
        log_a -= _logsumexp(log_a)
        return np.exp(log_a)


# --------------------------------------------------------------------------- #
# Market-regime wrapper                                                        #
# --------------------------------------------------------------------------- #
def _causal_ewma(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EWMA smoother (past-only, no look-ahead)."""
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = a * x[i] + (1 - a) * acc
        out[i] = acc
    return out


def _vol_observations(closes, highs=None, lows=None, *, span: int = 6,
                      opens=None) -> np.ndarray:
    """1-D observation = causally-smoothed log realized variance. The volatility
    level alone — the persistent, separable latent state the data check
    green-lit. No direction dimension (1-bar direction is noise, and a return
    axis is exactly what made the 2-D HMM whipsaw).

    ``opens`` doit être fourni pour que ``real_ohlc`` puisse authentifier le
    range : l'ancien appel ``real_ohlc(None, …)`` retournait toujours False et
    le HMM tombait silencieusement sur le proxy r² — jamais Parkinson.
    """
    from ml.vol_model import parkinson_var, real_ohlc

    c = np.asarray(closes, dtype=np.float64)
    r = np.diff(np.log(c))
    if highs is not None and lows is not None and real_ohlc(opens, highs, lows, c):
        rv = parkinson_var(np.asarray(highs, dtype=np.float64),
                           np.asarray(lows, dtype=np.float64))[1:]
    else:
        rv = r * r
    lrv = np.log(np.maximum(rv, 1e-12))
    if len(lrv) >= 2:
        lrv = _causal_ewma(lrv, span)
    s = np.std(lrv)
    return ((lrv - np.mean(lrv)) / (s if s > 1e-12 else 1.0))[:, None]


def _regime_observations(closes, highs=None, lows=None, *, span: int = 6,
                         opens=None) -> np.ndarray:
    """Observation matrix ``[smoothed trend, smoothed log-RV]`` per bar.

    Crucial design choice learned from the walk-forward A/B
    (tests/measure_model.py::_regime_eval): a *raw 1-bar return* dimension makes
    the HMM whipsaw — the sign of a single bar is ≈50/50 noise (the project's
    core thesis), so a return-driven state flips every bar. We instead feed a
    **causally EWMA-smoothed return** (a persistent *trend* proxy) and a smoothed
    **log realized variance**. Both are past-only (no leak), z-scored so neither
    dominates the diagonal-Gaussian fit. This keeps the part of "regime" that is
    actually persistent and predictable — the volatility level and the slow trend
    — and drops the part that isn't. Range-RV (Parkinson) when real OHLC exists,
    else squared returns."""
    from ml.vol_model import parkinson_var, real_ohlc

    c = np.asarray(closes, dtype=np.float64)
    r = np.diff(np.log(c))
    if highs is not None and lows is not None and real_ohlc(opens, highs, lows, c):
        rv = parkinson_var(np.asarray(highs, dtype=np.float64),
                           np.asarray(lows, dtype=np.float64))[1:]
    else:
        rv = r * r
    lrv = np.log(np.maximum(rv, 1e-12))
    m = min(len(r), len(lrv))
    r, lrv = r[-m:], lrv[-m:]
    if m >= 2:
        r = _causal_ewma(r, span)        # trend proxy (persistent), not 1-bar noise
        lrv = _causal_ewma(lrv, span)    # smoothed vol level
    def z(x):
        s = np.std(x)
        return (x - np.mean(x)) / (s if s > 1e-12 else 1.0)
    return np.column_stack([z(r), z(lrv)])


@dataclass
class RegimeHMM:
    """Gaussian-HMM regime model mapped onto the bot's 4 named regimes.

    Fits a 4-state HMM on ``[return, log-RV]`` and assigns each latent state a
    name by its emission means: the highest-variance state is ``volatile``; of
    the remaining three, the highest / lowest / middle mean-return states are
    ``bull`` / ``bear`` / ``range``. Exposes the learned transition matrix in
    REGIME_ORDER (drop-in for the Markov regime sampler) and a causal live
    regime read."""

    hmm: GaussianHMM
    state_to_regime: dict          # latent index -> regime name
    regime_to_state: dict
    kind: str = "vol3"             # "vol3" (volatility regimes) | "named4"
    trained: bool = True

    @classmethod
    def fit(cls, opens=None, highs=None, lows=None, closes=None, *,
            kind: str = "vol3", seed: int = 0, max_bars: int = 2500,
            n_init: int = 2, n_iter: int = 30) -> "RegimeHMM | None":
        """Fit the regime HMM. ``kind='vol3'`` (default, the validated taxonomy)
        learns 3 **volatility** states (calm/normal/turbulent) on smoothed log-RV
        — direction is 50/50 noise, so the persistent latent state is the vol
        level. ``kind='named4'`` keeps the legacy bull/bear/range/volatile shape
        on [trend, vol] for callers that still need those labels. ``max_bars``
        caps the fit window (the regime is stable; bounds the live refit cost)."""
        if kind == "named4":
            return cls._fit_named4(opens, highs, lows, closes, seed=seed)
        X = _vol_observations(closes, highs, lows, opens=opens)
        if len(X) < 120:
            return None
        X = X[-max_bars:]
        hmm = GaussianHMM.fit(X, n_states=3, seed=seed, n_init=n_init, n_iter=n_iter)
        if not hmm.trained:
            return None
        order = list(np.argsort(hmm.means[:, 0]))      # ascending vol level
        s2r = {int(order[0]): "calm", int(order[1]): "normal", int(order[2]): "turbulent"}
        return cls(hmm=hmm, state_to_regime=s2r,
                   regime_to_state={v: k for k, v in s2r.items()}, kind="vol3")

    @classmethod
    def _fit_named4(cls, opens, highs, lows, closes, *, seed=0) -> "RegimeHMM | None":
        X = _regime_observations(closes, highs, lows, opens=opens)
        if len(X) < 120:
            return None
        hmm = GaussianHMM.fit(X, n_states=4, seed=seed)
        if not hmm.trained:
            return None
        ret_mean, vol_mean = hmm.means[:, 0], hmm.means[:, 1]
        volatile_state = int(np.argsort(vol_mean)[-1])
        rest = sorted((s for s in range(4) if s != volatile_state), key=lambda s: ret_mean[s])
        s2r = {rest[0]: "bear", rest[1]: "range", rest[2]: "bull", volatile_state: "volatile"}
        return cls(hmm=hmm, state_to_regime=s2r,
                   regime_to_state={v: k for k, v in s2r.items()}, kind="named4")

    def _obs(self, closes, highs, lows, opens=None) -> np.ndarray:
        return (_vol_observations(closes, highs, lows, opens=opens)
                if self.kind == "vol3"
                else _regime_observations(closes, highs, lows, opens=opens))

    @property
    def order(self) -> tuple:
        return VOL_REGIME_ORDER if self.kind == "vol3" else REGIME_ORDER

    def transmat_in_regime_order(self) -> np.ndarray:
        """(K,K) learned transition matrix with rows/cols in ``self.order``."""
        order = self.order
        K = len(order)
        T = np.zeros((K, K))
        for i, ri in enumerate(order):
            for j, rj in enumerate(order):
                T[i, j] = self.hmm.transmat[self.regime_to_state[ri], self.regime_to_state[rj]]
        T /= T.sum(axis=1, keepdims=True)
        return T

    def regime_at(self, closes, highs=None, lows=None, opens=None) -> str:
        """Causal live regime: argmax of the filtered state at the last bar."""
        default = self.order[len(self.order) // 2]
        X = self._obs(closes, highs, lows, opens=opens)
        if len(X) < 2:
            return default
        k = int(np.argmax(self.hmm.filter_state(X)))
        return self.state_to_regime.get(k, default)

    def regime_probs(self, closes, highs=None, lows=None, opens=None) -> dict[str, float]:
        """Distribution causale filtrée sur les régimes nommés au dernier bar.

        La probabilité CONTINUE (pas l'argmax) est ce que la politique de risque
        consomme : une porte binaire sur l'argmax saute de ±40 % d'exposition à
        chaque bascule de régime au voisinage de la frontière — churn mesuré au
        harnais — alors que P(turbulent) évolue continûment."""
        uniform = {r: 1.0 / len(self.order) for r in self.order}
        X = self._obs(closes, highs, lows, opens=opens)
        if len(X) < 2:
            return uniform
        probs = self.hmm.filter_state(X)
        return {
            self.state_to_regime.get(k, self.order[0]): float(probs[k])
            for k in range(len(probs))
        }

    def regime_path(self, closes, highs=None, lows=None, opens=None) -> list[str]:
        """Viterbi regime path over the (closed) history, as regime names. The
        path is aligned to returns (length len(closes)-1); callers that need a
        per-price label can left-pad with the first regime."""
        default = self.order[len(self.order) // 2]
        X = self._obs(closes, highs, lows, opens=opens)
        if len(X) < 2:
            return [default] * max(len(np.asarray(closes)) - 1, 1)
        return [self.state_to_regime.get(int(s), default) for s in self.hmm.viterbi(X)]

