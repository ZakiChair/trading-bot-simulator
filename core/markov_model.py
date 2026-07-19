"""Markov chain models for regime and direction transitions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.market import MarketState, Regime
from core.thresholds import adaptive_threshold, classify_idx

DIRECTION_STATES = ("up", "flat", "down")
REGIME_STATES = tuple(r.value for r in Regime)
REGIME_IDX = {name: i for i, name in enumerate(REGIME_STATES)}

_SMOOTH = 0.05

_DEFAULT_DIR_TM = np.array(
    [
        [0.45, 0.30, 0.25],
        [0.30, 0.40, 0.30],
        [0.25, 0.30, 0.45],
    ]
)

_DEFAULT_REGIME_TM = {
    "bull": np.array([0.70, 0.15, 0.10, 0.05]),
    "bear": np.array([0.05, 0.10, 0.15, 0.70]),
    "range": np.array([0.15, 0.15, 0.55, 0.15]),
    "volatile": np.array([0.20, 0.20, 0.20, 0.40]),
}

# Lookup tables for vectorized regime sampling
REGIME_DRIFTS = np.array([0.0004, -0.00035, 0.0, 0.0], dtype=np.float64)
REGIME_VOLS = np.array([0.008, 0.009, 0.006, 0.018], dtype=np.float64)
DIR_DRIFT_SCALE = np.array([0.6, 0.0, -0.6], dtype=np.float64)


def _classify_return(ret: float, threshold: float) -> int:
    """Direction class for a log return (0=up, 1=flat, 2=down).

    The flat band is now the volatility-adaptive ``k·σ`` (``core.thresholds``)
    instead of a hardcoded 0.003 — so the Markov states use the *same*
    hausse/neutre/baisse definition as the learned model and the live scorer.
    """
    return classify_idx(ret, threshold)


@dataclass
class MarkovSnapshot:
    """Frozen Markov model state for UI display."""

    direction_formula: str = "P(S_{t+1}=j | S_t=i) = T_{ij}  (chaîne de Markov)"
    regime_formula: str = "P(R_{t+1}=k | R_t=r) = Π_{rk}  (Markov de régime)"
    direction_matrix: np.ndarray | None = None
    regime_matrix: np.ndarray | None = None
    start_direction: str = "flat"
    start_regime: str = "range"
    markov_weight: float = 0.35
    combined_formula: str = (
        "logit_i = log L_i + α·log P_markov_i + b_i  →  P_i = softmax(logit)"
    )

    def direction_matrix_text(self, width: int = 6) -> list[str]:
        if self.direction_matrix is None:
            return ["[dim]—[/]"]
        labels = ["↑", "●", "↓"]
        header = "     " + "  ".join(f"{l:>{width}}" for l in labels)
        lines = [header]
        for i, row_label in enumerate(labels):
            cells = "  ".join(f"{v:>{width}.0%}" for v in self.direction_matrix[i])
            lines.append(f" {row_label} │ {cells}")
        return lines

    def regime_matrix_text(self) -> list[str]:
        if self.regime_matrix is None:
            return ["[dim]—[/]"]
        labels = ["bull", "bear", "range", "vol"]
        lines = []
        for i, src in enumerate(REGIME_STATES):
            row = self.regime_matrix[i]
            cells = " ".join(f"{labels[j]}:{row[j]:.0%}" for j in range(4))
            lines.append(f" {src[:4]:5s} → {cells}")
        return lines


class MarkovChainModel:
    """Estimate and sample from direction + regime Markov chains."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.direction_tm = _DEFAULT_DIR_TM.copy()
        self._log_dir_tm = np.log(self.direction_tm + 1e-12)
        self.regime_tm = {k: v.copy() for k, v in _DEFAULT_REGIME_TM.items()}
        self._regime_rows = np.array([self.regime_tm[r] for r in REGIME_STATES])
        self._log_regime_rows = np.log(self._regime_rows + 1e-12)
        self.start_direction_idx = 1
        self.start_regime = "range"
        self.start_regime_idx = REGIME_IDX["range"]
        self.dir_threshold = 0.003  # volatility-adaptive once fit() has run

    def fit(self, state: MarketState) -> MarkovSnapshot:
        # 500 bars, pas 60 : 59 transitions sur 9 cases (~6 comptes/case) rendaient
        # la matrice statistiquement indiscernable du bruit (χ² significatif dans
        # 16 % des fenêtres seulement) et elle sautait de ±3,5 pts toutes les 5
        # barres. Une fenêtre longue estime les fréquences conditionnelles réelles
        # (faibles — V de Cramér ≈ 0,09 — mais au moins mesurées, pas du bruit).
        returns = state.returns(window=500)
        self.start_regime = state.regime.value
        self.start_regime_idx = REGIME_IDX.get(self.start_regime, 2)
        # Same volatility-adaptive flat band the learned model / live scorer use,
        # estimated from the full visible history (not just the 60-bar window).
        self.dir_threshold = adaptive_threshold(state.history)

        if len(returns) >= 10:
            self.direction_tm = self._estimate_direction_tm(returns, self.dir_threshold)
            self._log_dir_tm = np.log(self.direction_tm + 1e-12)
            self.start_direction_idx = _classify_return(float(returns[-1]), self.dir_threshold)
        else:
            self.direction_tm = _DEFAULT_DIR_TM.copy()
            self._log_dir_tm = np.log(self.direction_tm + 1e-12)
            self.start_direction_idx = 1

        self.regime_tm = self._estimate_regime_tm(state)
        self._regime_rows = np.array([self.regime_tm[r] for r in REGIME_STATES])
        self._log_regime_rows = np.log(self._regime_rows + 1e-12)

        return MarkovSnapshot(
            direction_matrix=self.direction_tm.copy(),
            regime_matrix=self._regime_rows.copy(),
            start_direction=DIRECTION_STATES[self.start_direction_idx],
            start_regime=self.start_regime,
        )

    def _estimate_direction_tm(self, returns: np.ndarray, threshold: float) -> np.ndarray:
        n = len(DIRECTION_STATES)
        counts = np.full((n, n), _SMOOTH)
        states = [_classify_return(float(r), threshold) for r in returns]
        for i in range(len(states) - 1):
            counts[states[i], states[i + 1]] += 1.0
        return counts / counts.sum(axis=1, keepdims=True)

    def _estimate_regime_tm(self, state: MarketState) -> dict[str, np.ndarray]:
        if len(state.returns(window=40)) < 15:
            return {k: v.copy() for k, v in _DEFAULT_REGIME_TM.items()}

        n = len(REGIME_STATES)
        counts = np.full((n, n), _SMOOTH)
        regimes: list[int] = []
        prices = state.history
        for i in range(20, len(prices)):
            window_ret = prices[i] / prices[i - 20] - 1.0
            vol = float(np.std(np.diff(np.log(prices[max(0, i - 20) : i + 1]))))
            if vol > 0.025:
                regimes.append(3)
            elif window_ret > 0.03:
                regimes.append(0)
            elif window_ret < -0.03:
                regimes.append(1)
            else:
                regimes.append(2)

        for i in range(len(regimes) - 1):
            counts[regimes[i], regimes[i + 1]] += 1.0

        return {
            name: counts[j] / counts[j].sum()
            for j, name in enumerate(REGIME_STATES)
        }

    def _vectorized_markov_sample(
        self,
        n_paths: int,
        horizon: int,
        tm: np.ndarray,
        log_tm: np.ndarray,
        start_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch-sample Markov paths — O(horizon) not O(n_paths × horizon)."""
        current = np.full(n_paths, start_idx, dtype=np.int8)
        paths = np.zeros((n_paths, horizon), dtype=np.int8)
        log_probs = np.zeros(n_paths, dtype=np.float64)

        for t in range(horizon):
            probs = tm[current]
            r = self._rng.random(n_paths)
            cum = np.cumsum(probs, axis=1)
            nxt = (r[:, None] >= cum).sum(axis=1)
            nxt = np.clip(nxt, 0, probs.shape[1] - 1).astype(np.int8)
            log_probs += log_tm[current, nxt]
            paths[:, t] = nxt
            current = nxt

        return paths, log_probs

    def sample_direction_paths_batch(
        self, n_paths: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._vectorized_markov_sample(
            n_paths, horizon, self.direction_tm, self._log_dir_tm, self.start_direction_idx
        )

    def sample_regime_paths_batch(
        self, n_paths: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        paths, log_p = self._vectorized_markov_sample(
            n_paths, horizon, self._regime_rows, self._log_regime_rows, self.start_regime_idx
        )
        return paths, log_p * 0.5  # lower weight for regime path

    def sample_direction_path(
        self, horizon: int
    ) -> tuple[np.ndarray, float, list[str]]:
        paths, log_ps = self.sample_direction_paths_batch(1, horizon)
        labels = [DIRECTION_STATES[i] for i in paths[0]]
        return paths[0], float(log_ps[0]), labels

    def sample_regime_path(self, horizon: int) -> tuple[list[str], float]:
        paths, log_ps = self.sample_regime_paths_batch(1, horizon)
        return [REGIME_STATES[i] for i in paths[0]], float(log_ps[0])