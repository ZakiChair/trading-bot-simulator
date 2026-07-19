"""Learned next-candle direction model — softmax classifier trained by
gradient descent on historical prices.

The bot's *prediction* of the next candle used to be purely statistical
(Markov + GBM). This module adds a model that **learns** from history: it maps a
small, engine-free feature vector (momentum, volatility, skew, range position…)
to a probability over the next candle's direction — hausse / neutre / baisse.

Two phases mirror the user's mental model:

* **Entraînement** — :func:`train_candle_model` walks the loaded history, builds
  a look-ahead-free ``(X, y)`` dataset and runs plain mini-/full-batch gradient
  descent for several epochs, recording a loss/accuracy curve and persisting the
  weights to ``models/<symbol>_<timeframe>.npz``.
* **Live (prédiction)** — :meth:`CandleModel.predict_proba` loads those frozen
  weights and outputs the direction probabilities that drive the bubble chart.

The features are deliberately *bundle-free* (computed from a price slice only),
so building the dataset over ~1500 bars is a handful of vectorised ops and a
full training run finishes in well under a second — fast enough for a button
press in the TUI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.thresholds import adaptive_threshold, classify_idx

# Direction classes — index order matches core.next_candle._classify_idx
# (0=up, 1=flat, 2=down) so the learned output drops straight into the existing
# forecast pipeline.
DIRECTIONS: tuple[str, str, str] = ("up", "flat", "down")
DIR_THRESHOLD = 0.003  # legacy fallback; real labelling now uses adaptive_threshold

# Bumped whenever ``candle_features`` changes shape/meaning. A persisted model
# whose version (or weight-row count) doesn't match the running code is rejected
# on load — otherwise ``predict_proba`` would broadcast a new-length feature
# vector against stale ``feat_mean``/``weights`` and raise every tick. A reject
# returns None → the engine falls back to the statistical forecast until the
# user retrains (key ``g``). v2 = volatility-normalised stationary features.
FEATURE_VERSION = 2
FEATURE_NAMES: tuple[str, ...] = (
    "r1_z", "mom5_z", "mom10_z", "mom20_z",
    "vol_ratio", "vol_level", "rsi14", "autocorr1",
    "skew20", "up_frac14", "accel_z", "bb_z", "range_pos20",
)
N_FEATURES = len(FEATURE_NAMES)

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR = _DEFAULT_MODELS_DIR


def models_dir() -> Path:
    """Répertoire des poids persistés.

    Honore ``BOT_MODELS_DIR`` (env) pour que les tests/outillages qui
    entraînent puissent rediriger la persistance vers un répertoire jetable —
    la suite de tests a déjà écrasé deux fois le vrai ``models/BTC_USDT_1h.npz``
    avec un modèle-jouet entraîné sur un faux feed (val « 100 % » sur 66
    échantillons synthétiques).

    Précédence : un ``cm.MODELS_DIR`` monkeypatché (le pattern
    ``_isolated_models()`` des tests) gagne sur l'env — sinon un
    ``BOT_MODELS_DIR`` global ferait FUIR les modèles entre tests d'une même
    exécution, exactement ce que l'isolation par test empêche.
    """
    import os

    if MODELS_DIR != _DEFAULT_MODELS_DIR:
        return MODELS_DIR  # isolation par test (monkeypatch) — prioritaire
    override = os.environ.get("BOT_MODELS_DIR")
    return Path(override) if override else MODELS_DIR


def classify_return(log_ret: float, threshold: float = DIR_THRESHOLD) -> int:
    """Map a log return to a direction class index (0=up, 1=flat, 2=down)."""
    return classify_idx(log_ret, threshold)


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) else 0.0


def candle_features(prices: np.ndarray) -> np.ndarray:
    """Engine-free, **volatility-normalised** feature vector for one bar.

    Uses only past data (``prices[-1]`` is the current close), so the same
    function serves both dataset construction (``prices[: i + 1]``) and live
    inference (``state.history``) — train/inference parity, no look-ahead.

    Every return/momentum feature is divided by the recent per-bar σ (a z-score),
    so the vector is **stationary**: a +1 % move in a calm market and a +1 % move
    in a turbulent one map to very different inputs. The old raw features made
    the linear weights regime-specific (they fit one volatility regime and broke
    on the next), which is the over-fit the baseline showed (train ≫ val). The
    set adds the structure that *is* predictable on real data — volatility
    clustering (``vol_ratio``), mean-reversion (``rsi14``, ``bb_z``) and the
    momentum/mean-reversion regime itself (``autocorr1``).
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 5:
        return np.zeros(N_FEATURES, dtype=np.float64)

    logret = np.diff(np.log(p))
    n = len(logret)

    # Recent per-bar volatility — the normaliser. Floored so it never divides by
    # zero on a flat patch.
    sig = max(_safe_std(logret[-20:]), 1e-9)

    def mom_z(k: int) -> float:
        """k-bar log return in σ units (√k random-walk scaling), clipped."""
        if n < k:
            return 0.0
        r = float(np.log(p[-1] / p[-k - 1]))
        return float(np.clip(r / (sig * math.sqrt(k)), -6.0, 6.0))

    r1_z = float(np.clip(logret[-1] / sig, -6.0, 6.0))
    m5, m10, m20 = mom_z(5), mom_z(10), mom_z(20)
    accel_z = float(np.clip(m5 - m10, -6.0, 6.0))

    # Volatility clustering: short vs long realised vol (log-ratio is stationary).
    sig_s = max(_safe_std(logret[-10:]), 1e-9)
    sig_l = max(_safe_std(logret[-50:]) if n >= 50 else _safe_std(logret), 1e-9)
    vol_ratio = float(np.clip(math.log(sig_s / sig_l), -3.0, 3.0))
    vol_level = float(np.clip(math.log(sig / sig_l), -3.0, 3.0))

    # RSI-style oscillator on the last 14 returns, centred to [-0.5, 0.5].
    r14 = logret[-14:]
    gains = float(np.sum(r14[r14 > 0]))
    losses = float(-np.sum(r14[r14 < 0]))
    rsi = (gains / (gains + losses) - 0.5) if (gains + losses) > 1e-12 else 0.0

    # Lag-1 autocorrelation of recent returns: >0 trending, <0 mean-reverting.
    rr = logret[-30:]
    rc = rr - rr.mean()
    denom = float(np.sum(rc * rc))
    autocorr1 = float(np.clip(np.sum(rc[1:] * rc[:-1]) / denom, -1.0, 1.0)) if denom > 1e-12 else 0.0

    last = logret[-20:]
    sd = _safe_std(last)
    skew = float(np.mean(((last - last.mean()) / sd) ** 3)) if sd > 1e-9 and len(last) > 2 else 0.0
    skew = float(np.clip(skew, -5.0, 5.0))

    up_frac = float(np.mean(r14 > 0) - 0.5) if len(r14) else 0.0

    # Bollinger z: price distance from its 20-bar mean in price-σ units.
    window20 = p[-20:]
    ma20 = float(np.mean(window20))
    sd20 = float(np.std(window20))
    bb_z = float(np.clip((p[-1] - ma20) / sd20, -4.0, 4.0)) if sd20 > 1e-9 else 0.0

    rng = float(window20.max() - window20.min())
    range_pos = float((p[-1] - window20.min()) / rng - 0.5) if rng > 1e-9 else 0.0

    return np.array(
        [r1_z, m5, m10, m20, vol_ratio, vol_level, rsi, autocorr1,
         skew, up_frac, accel_z, bb_z, range_pos],
        dtype=np.float64,
    )


def build_dataset(
    prices: np.ndarray,
    warmup: int = 35,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` over the price series, look-ahead-free.

    For each bar ``i`` (from ``warmup`` to the second-to-last), the features use
    ``prices[: i + 1]`` and the label is the direction of the realised next-bar
    log return ``log(prices[i+1] / prices[i])``.

    The flat/up/down band is **local to each bar** when ``threshold is None``:
    bar ``i`` is labelled with ``adaptive_threshold(prices[: i + 1])`` — the band
    appropriate to *that bar's own recent volatility*, which is exactly the band
    live inference recomputes per tick. A single global band (the old behaviour)
    is wrong here: it's estimated from the most recent window and then stamped
    onto bars from calmer/wilder past regimes, so the flat class swells to a
    plurality and the trained model learns a flat-dominated prior that predicts
    NEUTRE almost always — even when the realised moves are balanced. Pass an
    explicit ``threshold`` to force a fixed band (e.g. for ablations).
    """
    p = np.asarray(prices, dtype=np.float64)
    n = len(p)
    if n < warmup + 2:
        return np.zeros((0, N_FEATURES)), np.zeros(0, dtype=np.int64)

    X: list[np.ndarray] = []
    y: list[int] = []
    for i in range(warmup, n - 1):
        X.append(candle_features(p[: i + 1]))
        thr_i = adaptive_threshold(p[: i + 1]) if threshold is None else threshold
        y.append(classify_return(float(np.log(p[i + 1] / p[i])), thr_i))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class TrainReport:
    """Outcome of a training run — drives the learning curve in the UI."""

    epochs: int
    n_train: int
    n_val: int
    loss_history: list[float] = field(default_factory=list)
    train_acc_history: list[float] = field(default_factory=list)
    val_acc_history: list[float] = field(default_factory=list)
    final_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    val_majority: float = 0.0  # honest baseline: always-predict-majority-class accuracy on val
    val_brier: float = 0.0     # multiclass Brier on val (calibration; lower = better)
    temperature: float = 1.0   # post-hoc calibration scalar fit on val
    class_counts: tuple[int, int, int] = (0, 0, 0)


@dataclass
class CandleModel:
    """Softmax (multinomial logistic) next-candle direction classifier."""

    symbol: str = ""
    timeframe: str = "1h"
    weights: np.ndarray = field(default_factory=lambda: np.zeros((N_FEATURES, 3)))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    feat_mean: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))
    feat_std: np.ndarray = field(default_factory=lambda: np.ones(N_FEATURES))
    trained: bool = False
    n_samples: int = 0
    epochs_trained: int = 0
    learning_rate: float = 0.0
    val_accuracy: float = 0.0
    train_accuracy: float = 0.0
    dir_threshold: float = DIR_THRESHOLD  # flat band used for labels (adaptive)
    # Temperature scaling: a single scalar T fit on the validation split that
    # divides the logits before softmax. T>1 softens over-confident probabilities
    # so a stated "70 %" is right ~70 % of the time (lower Brier). 1.0 = identity.
    temperature: float = 1.0
    # Bar index (into the loaded price series) of the last bar whose label was
    # used for *training*. The live walk-forward must start strictly after this
    # so its "fiabilité" is measured on genuinely unseen candles. -1 = unknown.
    train_end_step: int = -1
    loss_history: list[float] = field(default_factory=list)
    val_acc_history: list[float] = field(default_factory=list)

    # -- online (Live) self-reinforcement --------------------------------- #
    # Continual learning during Live mode: each *settled* candle becomes one SGD
    # step (see online_update). This tracks REGIME drift and keeps the live
    # probabilities calibrated — it does NOT manufacture a 1-bar directional edge
    # (~50/50 by construction on this substrate; see ANALYSE_CRITIQUE_MODELE.md).
    # The update is anchored to the batch-trained weights so it can't wander, and
    # it lives in MEMORY only (never persisted): persisting online-adapted weights
    # would let a later Live walk start on bars the model already saw, resurrecting
    # the non-OOS look-ahead bug (red-flag #3 in trading-bot-pertinence-redflags).
    anchor_weights: np.ndarray | None = field(default=None, repr=False)
    anchor_bias: np.ndarray | None = field(default=None, repr=False)
    n_online: int = 0
    # lr0=0.003 chosen by the OOS A/B sweep (tests/measure_model.py online) across
    # BTC/ETH/SOL 1h: it is the only rate that improves (or is neutral on) Brier
    # for ALL three assets (mean ΔBrier −0.0046, ~0.7 % better calibration) while
    # keeping drift gentle (0.15–0.31, far under the trust region). Larger rates
    # chase per-candle noise and *hurt* calibration. The win is small and is a
    # regime/calibration effect — NOT a directional edge (acc stays ≈50/50).
    online_lr0: float = 0.003     # base step; decays as lr0/(1+n/τ)
    online_tau: float = 200.0     # lr-decay timescale (in settled bars)
    online_anchor: float = 0.02   # ridge pull-back strength toward the batch weights
    online_max_drift: float = 0.75  # hard L2 trust region around the anchor (relative)
    online_loss_history: list[float] = field(default_factory=list, repr=False)

    # -- inference -------------------------------------------------------- #
    def _standardize(self, feats: np.ndarray) -> np.ndarray:
        return (feats - self.feat_mean) / self.feat_std

    def predict_proba(self, prices: np.ndarray) -> np.ndarray:
        """Calibrated direction probabilities ``[p_up, p_flat, p_down]``."""
        feats = candle_features(prices)
        t = self.temperature if self.temperature and self.temperature > 1e-6 else 1.0
        z = (self._standardize(feats) @ self.weights + self.bias) / t
        return _softmax(z.reshape(1, -1))[0]

    def predict_direction(self, prices: np.ndarray) -> str:
        return DIRECTIONS[int(np.argmax(self.predict_proba(prices)))]

    # -- online self-reinforcement (Live) -------------------------------- #
    def features_asof(self, prices: np.ndarray) -> np.ndarray | None:
        """Standardised feature vector for the *current* bar — the exact input
        :meth:`predict_proba` used. Captured at prediction time and replayed
        (unchanged) as the training input once the candle settles, so the online
        step learns on precisely what it predicted with — no slice/off-by-one
        drift between forecast and update. ``None`` when it can't be computed."""
        p = np.asarray(prices, dtype=np.float64)
        if not self.trained or len(p) < 5:
            return None
        return self._standardize(candle_features(p))

    def ensure_anchor(self) -> None:
        """Pin the current (batch-trained / freshly loaded) weights as the ridge
        anchor that online updates are pulled back toward. Idempotent — the first
        call wins, so the anchor always stays the *clean* solution."""
        if self.anchor_weights is None:
            self.anchor_weights = self.weights.copy()
            self.anchor_bias = self.bias.copy()

    def online_update(self, x_std: np.ndarray | None, target_idx: int) -> dict:
        """One online SGD step from a settled candle (self-reinforcement).

        Same softmax cross-entropy objective as the batch trainer, applied to a
        single sample on the RAW logits (temperature is reporting-only
        calibration). An anchor term ``online_anchor·(W − W₀)`` keeps the weights
        from drifting away from the validated batch solution; the step decays as
        ``online_lr0/(1 + n/τ)``. Standardisation stays fixed (batch stats).

        Honest scope: tracks non-stationarity and keeps the live probabilities
        calibrated; it cannot create a 1-bar directional edge. Returns a small
        telemetry dict (``{}`` when it can't / shouldn't update)."""
        if not self.trained or x_std is None or not (0 <= int(target_idx) < 3):
            return {}
        target_idx = int(target_idx)
        self.ensure_anchor()
        lr = self.online_lr0 / (1.0 + self.n_online / max(self.online_tau, 1e-9))
        logits = x_std @ self.weights + self.bias  # (3,) raw, no temperature
        z = logits - logits.max()
        e = np.exp(z)
        probs = e / e.sum()
        err = probs.copy()
        err[target_idx] -= 1.0  # probs − onehot(target)
        gW = np.outer(x_std, err) + self.online_anchor * (self.weights - self.anchor_weights)
        gb = err + self.online_anchor * (self.bias - self.anchor_bias)
        # Rebind (not in-place) so a read-only loaded array is never mutated.
        self.weights = self.weights - lr * gW
        self.bias = self.bias - lr * gb
        # Trust region: never let the live weights stray more than
        # ``online_max_drift`` (relative L2) from the validated batch anchor — a
        # hard guarantee that the online model stays a neighbour of the model we
        # actually validated, even under an adverse one-sided stream.
        if self.online_max_drift > 0:
            dev = self.weights - self.anchor_weights
            base = float(np.linalg.norm(self.anchor_weights)) + 1e-9
            rel = float(np.linalg.norm(dev)) / base
            if rel > self.online_max_drift:
                self.weights = self.anchor_weights + dev * (self.online_max_drift / rel)
        self.n_online += 1
        nll = float(-math.log(max(float(probs[target_idx]), 1e-12)))
        self.online_loss_history.append(nll)
        if len(self.online_loss_history) > 200:
            self.online_loss_history = self.online_loss_history[-200:]
        return {
            "lr": float(lr),
            "nll": nll,
            "n_online": self.n_online,
            "pred": int(np.argmax(probs)),
            "target": target_idx,
        }

    def reset_online(self) -> None:
        """Discard online adaptation — restore the clean batch-trained weights."""
        if self.anchor_weights is not None:
            self.weights = self.anchor_weights.copy()
            self.bias = self.anchor_bias.copy()
        self.n_online = 0
        self.online_loss_history = []

    def online_drift(self) -> float:
        """Relative L2 distance of the online weights from the batch anchor — a
        0..∞ 'how far it has adapted' gauge for the UI (0 = unchanged)."""
        if self.anchor_weights is None or not self.n_online:
            return 0.0
        base = float(np.linalg.norm(self.anchor_weights)) + 1e-9
        return float(np.linalg.norm(self.weights - self.anchor_weights) / base)

    def online_recent_nll(self, k: int = 30) -> float:
        """Mean online NLL over the last ``k`` settled candles (lower = better
        live fit). 0.0 until the first online update."""
        if not self.online_loss_history:
            return 0.0
        tail = self.online_loss_history[-k:]
        return float(sum(tail) / len(tail))

    # -- persistence ------------------------------------------------------ #
    def save(self, path: Path | None = None) -> Path:
        path = path or model_path(self.symbol, self.timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            bias=self.bias,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
            meta=np.array(
                [
                    self.symbol,
                    self.timeframe,
                    str(self.n_samples),
                    str(self.epochs_trained),
                    f"{self.learning_rate:.6f}",
                    f"{self.val_accuracy:.6f}",
                    f"{self.train_accuracy:.6f}",
                    f"{self.dir_threshold:.8f}",
                    str(self.train_end_step),
                    str(FEATURE_VERSION),
                    f"{self.temperature:.6f}",
                ]
            ),
            loss_history=np.asarray(self.loss_history, dtype=np.float64),
            val_acc_history=np.asarray(self.val_acc_history, dtype=np.float64),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> CandleModel:
        data = np.load(path, allow_pickle=False)
        meta = data["meta"]
        # Back-compat: older models were saved without the threshold / train-end.
        dir_thr = float(meta[7]) if len(meta) > 7 else DIR_THRESHOLD
        train_end = int(meta[8]) if len(meta) > 8 else -1
        # Reject a model trained against a different feature schema. Older models
        # carry no version field (→ treated as v1); a row-count mismatch is the
        # hard guard in case the version was somehow not bumped.
        feat_version = int(meta[9]) if len(meta) > 9 else 1
        weights = data["weights"]
        if feat_version != FEATURE_VERSION or weights.shape[0] != N_FEATURES:
            raise ValueError(
                f"stale feature schema (v{feat_version}/{weights.shape[0]} feats "
                f"vs v{FEATURE_VERSION}/{N_FEATURES}) — retrain required"
            )
        temperature = float(meta[10]) if len(meta) > 10 else 1.0
        return cls(
            symbol=str(meta[0]),
            timeframe=str(meta[1]),
            weights=data["weights"],
            bias=data["bias"],
            feat_mean=data["feat_mean"],
            feat_std=data["feat_std"],
            trained=True,
            n_samples=int(meta[2]),
            epochs_trained=int(meta[3]),
            learning_rate=float(meta[4]),
            val_accuracy=float(meta[5]),
            train_accuracy=float(meta[6]),
            dir_threshold=dir_thr,
            temperature=temperature,
            train_end_step=train_end,
            loss_history=list(data["loss_history"]),
            val_acc_history=list(data["val_acc_history"]),
        )


def model_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return models_dir() / f"{safe}_{timeframe}.npz"


def load_model(symbol: str, timeframe: str) -> CandleModel | None:
    """Load a persisted model for ``(symbol, timeframe)`` if one exists."""
    model, _note = load_model_with_note(symbol, timeframe)
    return model


def load_model_with_note(symbol: str, timeframe: str) -> tuple[CandleModel | None, str]:
    """Comme :func:`load_model`, mais dit POURQUOI quand rien n'est chargé.

    Un ``.npz`` au schéma de features périmé (ou corrompu) était rejeté en
    silence : l'utilisateur voyait « non entraîné (g) » sans jamais savoir
    qu'un modèle existait et avait été écarté. La note remonte au journal.
    """
    path = model_path(symbol, timeframe)
    if not path.exists():
        return None, ""
    try:
        return CandleModel.load(path), ""
    except Exception as exc:
        return None, (
            f"⚠ Modèle {path.name} ignoré ({exc}) — "
            f"ré-entraîne avec la touche g (ou --train)."
        )


def _accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.argmax(probs, axis=1) == y))


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score — mean squared error between the predicted
    probability vector and the one-hot truth (0 = perfect; lower = better
    calibrated). The honest "is a stated 70 % really 70 %?" metric."""
    if len(y) == 0:
        return 0.0
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _fit_temperature(val_logits: np.ndarray, yva: np.ndarray) -> float:
    """Temperature scaling: the scalar ``T`` that minimises validation NLL when
    logits are divided by ``T`` before softmax. Coarse-to-fine 1-D search (the
    objective is smooth and unimodal in T). Needs a non-trivial val split."""
    if len(yva) < 20:
        return 1.0
    Y = np.eye(3)[yva]

    def nll(T: float) -> float:
        return float(-np.mean(np.sum(Y * np.log(_softmax(val_logits / T) + 1e-12), axis=1)))

    grid = np.linspace(0.5, 5.0, 46)
    best_T = min(grid, key=nll)
    fine = np.linspace(max(0.3, best_T - 0.1), best_T + 0.1, 21)
    best_T = min(fine, key=nll)
    return float(best_T)


def train_candle_model(
    prices: np.ndarray,
    *,
    symbol: str = "",
    timeframe: str = "1h",
    epochs: int = 400,
    lr: float = 0.3,
    l2: float = 2e-3,
    val_frac: float = 0.2,
    warmup: int = 35,
    oos_tail_frac: float = 0.15,
    balance_classes: bool = False,
    progress=None,
) -> tuple[CandleModel, TrainReport]:
    """Train a :class:`CandleModel` by gradient descent on historical prices.

    Chronological train/val split (no shuffle — time series), feature
    standardisation fit on the train split only, early stopping on the
    validation loss. Returns the model plus a :class:`TrainReport` whose
    histories feed the learning curve.

    ``oos_tail_frac`` : fraction FINALE de l'historique jamais vue pendant
    l'entraînement NI la sélection. La val (early stopping + température) fait
    partie de la sélection de modèle — démarrer la marche Live/Paper « OOS »
    sur ces mêmes barres, comme avant, sur-estimait la fiabilité (audit
    2026-07). ``train_end_step`` pointe désormais la fin de la zone touchée
    (train+val) ; la marche Live démarre strictement après, sur la queue vierge.

    ``balance_classes`` defaults to **False**: 1-bar direction has no real edge
    (balanced accuracy ≈ random on real data), so up-weighting the minority
    up/down classes only *worsens* calibration and pushes raw accuracy below the
    majority baseline without buying any genuine skill. Training on the natural
    class distribution gives honest, better-calibrated probabilities (lower
    Brier), and the bubble chart already shows the full probability *mass* per
    direction — so predictions stay varied without forcing a skewed objective.

    ``progress`` is an optional ``callable(epoch, loss, val_acc)`` used by the UI
    to stream training progress.
    """
    p_full = np.asarray(prices, dtype=np.float64)
    # Queue vierge : ni entraînement ni sélection ne voient les derniers
    # ``oos_tail_frac`` de l'historique — c'est la zone d'évaluation honnête.
    cut = len(p_full)
    if oos_tail_frac > 0 and len(p_full) > (warmup + 60) / (1 - oos_tail_frac):
        cut = int(len(p_full) * (1.0 - oos_tail_frac))
    X, y = build_dataset(p_full[:cut], warmup=warmup)
    n = len(y)
    if n < 30:
        # Not enough history to learn anything meaningful — return an untrained
        # model so callers can fall back to the statistical forecast.
        return CandleModel(symbol=symbol, timeframe=timeframe), TrainReport(0, 0, 0)

    thr = adaptive_threshold(p_full[:cut])
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:], y[n_train:]
    # TOUTES les barres < cut sont consommées par train+val (la val sert à
    # l'early stopping et à la température = sélection de modèle). La marche
    # Live/Paper doit commencer strictement après ``cut - 1``.
    train_end_step = cut - 1

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-8] = 1.0
    Xtr_s = (Xtr - mean) / std
    Xva_s = (Xva - mean) / std

    Ytr = np.eye(3)[ytr]
    counts = np.bincount(ytr, minlength=3).astype(np.float64)
    if balance_classes:
        inv = np.where(counts > 0, ytr.size / (3.0 * counts), 0.0)
        sample_w = inv[ytr]
        sample_w *= len(sample_w) / sample_w.sum()  # keep average weight ~1
    else:
        sample_w = np.ones(len(ytr))
    sw = sample_w.reshape(-1, 1)

    rng = np.random.default_rng(0)
    W = rng.normal(0.0, 0.01, (N_FEATURES, 3))
    b = np.zeros(3)

    report = TrainReport(epochs=epochs, n_train=n_train, n_val=n_val)
    report.class_counts = (int(counts[0]), int(counts[1]), int(counts[2]))
    report.val_majority = float(np.bincount(yva, minlength=3).max() / max(len(yva), 1))

    Yva = np.eye(3)[yva]
    eps = 1e-12
    # Early stopping: track the *validation* loss and keep the weights that
    # minimised it, instead of training a fixed 400 epochs and shipping the
    # over-fit tail. Checks happen every 5 epochs; patience is in those checks.
    best_val_loss = np.inf
    best_W, best_b = W.copy(), b.copy()
    best_epoch = 0
    since_improve = 0
    patience = 12  # ~60 epochs without val-loss improvement → stop

    for epoch in range(epochs):
        logits = Xtr_s @ W + b
        probs = _softmax(logits)
        # Weighted cross-entropy (mean over weighted samples) + L2.
        ce = -np.sum(sw * Ytr * np.log(probs + eps)) / len(ytr)
        loss = ce + l2 * float(np.sum(W * W))

        dlogits = (probs - Ytr) * sw / len(ytr)
        dW = Xtr_s.T @ dlogits + 2.0 * l2 * W
        db = dlogits.sum(axis=0)
        W -= lr * dW
        b -= lr * db

        if epoch % 5 == 0 or epoch == epochs - 1:
            vprobs = _softmax(Xva_s @ W + b)
            val_loss = float(-np.mean(np.sum(Yva * np.log(vprobs + eps), axis=1)))
            tr_acc = _accuracy(probs, ytr)
            va_acc = _accuracy(vprobs, yva)
            report.loss_history.append(float(loss))
            report.train_acc_history.append(tr_acc)
            report.val_acc_history.append(va_acc)
            if progress is not None:
                progress(epoch, float(loss), va_acc)
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_W, best_b = W.copy(), b.copy()
                best_epoch = epoch
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= patience and n_val >= 20:
                    break

    # Restore the best (lowest-val-loss) weights before scoring / persisting.
    W, b = best_W, best_b
    epochs_run = best_epoch + 1
    val_logits = Xva_s @ W + b
    # Post-hoc probability calibration: fit a temperature on the held-out split,
    # then score the *calibrated* probabilities with Brier (honest "is 70 % really
    # 70 %?"). Temperature only rescales confidence, so accuracy is unchanged.
    temperature = _fit_temperature(val_logits, yva)
    val_probs = _softmax(val_logits / temperature)
    train_acc = _accuracy(_softmax(Xtr_s @ W + b), ytr)
    val_acc = _accuracy(val_probs, yva)
    report.final_loss = report.loss_history[-1] if report.loss_history else 0.0
    report.train_accuracy = train_acc
    report.val_accuracy = val_acc
    report.val_brier = brier_score(val_probs, yva)

    model = CandleModel(
        symbol=symbol,
        timeframe=timeframe,
        weights=W,
        bias=b,
        feat_mean=mean,
        feat_std=std,
        trained=True,
        n_samples=n,
        epochs_trained=epochs_run,
        learning_rate=lr,
        val_accuracy=val_acc,
        train_accuracy=train_acc,
        temperature=temperature,
        dir_threshold=thr,
        train_end_step=train_end_step,
        loss_history=list(report.loss_history),
        val_acc_history=list(report.val_acc_history),
    )
    return model, report
