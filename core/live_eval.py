"""Live-mode self-evaluation: walk the candles forward, score each prediction
against what actually happened, and grade the bot's reliability.

In **Live mode** the bot no longer trades — its only job is to *predict the next
candle*. This module turns that into an honest feedback loop:

1. When the bot predicts candle ``t+1`` it **opens** a :class:`PendingPrediction`
   (direction, expected return, target bar + timestamp, wall-clock time).
2. When that candle **closes**, :meth:`LiveEvaluator.settle` measures the delta
   between prediction and reality (right/wrong direction, return error) and folds
   it into a running tally.
3. At the end of a live session :meth:`LiveEvaluator.finalize` emits a
   **note de fiabilité** (0–100) and a verdict on whether the model needs more
   training — its directional accuracy vs the always-predict-majority baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core.conformal import ConformalCalibrator
from core.timeutil import format_ts, now_hms

_DIR_THRESHOLD = 0.003  # |return| below this is "flat" — matches the forecast
_DIR_FR = {"up": "HAUSSE", "flat": "NEUTRE", "down": "BAISSE"}
_DIR_STYLE = {"up": "green", "flat": "yellow", "down": "red"}

# Minimum candles before a verdict is statistically meaningful. Below this the
# error bars on accuracy are wider than any plausible edge, so we refuse to call
# the bot "fiable" no matter how good the point estimate looks.
_MIN_EVAL = 25


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion.

    The honest "at least this accurate" figure at ~95% confidence — far less
    optimistic than the raw hit rate on small samples, so a lucky 7/10 doesn't
    masquerade as skill.
    """
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def classify(ret: float, threshold: float = _DIR_THRESHOLD) -> str:
    if ret > threshold:
        return "up"
    if ret < -threshold:
        return "down"
    return "flat"


@dataclass
class PendingPrediction:
    """A forecast awaiting the close of its target candle."""

    target_step: int
    target_ts: int | None
    base_price: float
    direction: str
    return_pct: float
    probability: float
    model_driven: bool
    made_at: str  # wall-clock HH:MM:SS when the prediction was emitted
    timeframe: str = "1h"
    # Full direction mass [P_up, P_flat, P_down] at prediction time — lets the
    # scorer measure *calibration* (Brier), not just whether argmax was right.
    probs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Standardised feature vector the model predicted with — stashed at predict
    # time so the online (self-reinforcement) update learns on exactly what it
    # forecast, with no slice/off-by-one drift. None when no model drove it.
    features: Any = None
    # Next-bar σ̂ (rough-vol) and the conformal return interval emitted at predict
    # time — settled (scored + folded into the calibrator) when the candle closes.
    pred_sigma: float = 0.0
    conf_lo: float = 0.0
    conf_hi: float = 0.0
    conf_level: float = 0.9


@dataclass
class SettledPrediction:
    """A prediction whose candle has closed — prediction vs reality."""

    target_step: int
    target_ts: int | None
    predicted_dir: str
    realized_dir: str
    predicted_ret: float
    realized_ret: float
    correct: bool


@dataclass
class LiveEvaluator:
    """Running scorecard for a live prediction session."""

    started: bool = False
    pending: PendingPrediction | None = None
    n_eval: int = 0
    n_correct: int = 0
    abs_err_sum: float = 0.0
    brier_sum: float = 0.0  # Σ multiclass Brier over settled predictions
    dir_threshold: float = _DIR_THRESHOLD  # flat band for scoring (adaptive)
    realized_counts: dict[str, int] = field(
        default_factory=lambda: {"up": 0, "flat": 0, "down": 0}
    )
    history: list[SettledPrediction] = field(default_factory=list)
    finalized: bool = False
    # Distribution-free conformal calibrator for the next-bar return interval.
    # Fed predict-before-update (interval emitted at open, scored+settled at
    # close → no leak); reset per session so it never carries one asset's scores
    # into another (mirrors the engine's per-symbol cone cache key).
    conformal: ConformalCalibrator = field(
        default_factory=lambda: ConformalCalibrator(target_alpha=0.1, window=300, gamma=0.01)
    )

    # -- lifecycle -------------------------------------------------------- #
    def reset(self) -> None:
        self.started = True
        self.pending = None
        self.n_eval = 0
        self.n_correct = 0
        self.abs_err_sum = 0.0
        self.brier_sum = 0.0
        self.realized_counts = {"up": 0, "flat": 0, "down": 0}
        self.history = []
        self.finalized = False
        self.conformal = ConformalCalibrator(target_alpha=0.1, window=300, gamma=0.01)

    # -- opening a prediction -------------------------------------------- #
    def open_from_forecast(self, forecast) -> PendingPrediction:
        # La prédiction SCORÉE = la source CALIBRÉE (argmax des masses
        # hausse/neutre/baisse) — la même que le bandeau « Consensus » et que le
        # Brier calculé sur ``probs``. L'ancienne version scorait la bulle ★
        # (pick pondéré vraisemblance × risque) : l'écran pouvait afficher
        # « NEUTRE probable » pendant que la fiabilité comptait un pari HAUSSE —
        # deux « plus probable » contradictoires, et une accuracy mesurée sur un
        # objet que le Brier ne voyait pas.
        probs = (
            float(getattr(forecast, "prob_up", 0.0)),
            float(getattr(forecast, "prob_flat", 0.0)),
            float(getattr(forecast, "prob_down", 0.0)),
        )
        if any(probs):
            idx = max(range(3), key=lambda i: probs[i])
            direction = ("up", "flat", "down")[idx]
            probability = probs[idx]
            return_pct = float(getattr(forecast, "expected_return", 0.0))
        else:  # repli (forecast sans masses — ne devrait pas arriver)
            mp = forecast.most_probable
            direction, probability, return_pct = mp.direction, mp.probability, mp.return_pct
        # Conformal next-bar return interval: martingale centre (μ=0), scale =
        # the rough-vol one-step σ̂ carried on the forecast. The band is what the
        # UI shows as "with X% confidence the next bar lands in […]"; it is scored
        # and folded into the calibrator only when the candle settles.
        sigma = float(getattr(forecast, "gbm_vol", 0.0)) or 0.0
        level = 1.0 - self.conformal.target_alpha
        lo, hi = self.conformal.interval(0.0, sigma) if sigma > 0 else (0.0, 0.0)
        self.pending = PendingPrediction(
            target_step=forecast.target_step,
            target_ts=forecast.target_ts,
            base_price=forecast.current_price,
            direction=direction,
            return_pct=return_pct,
            probability=probability,
            model_driven=forecast.model_driven,
            made_at=now_hms(),
            timeframe=forecast.timeframe,
            probs=probs,
            pred_sigma=sigma,
            conf_lo=lo,
            conf_hi=hi,
            conf_level=level,
        )
        return self.pending

    def pending_log_line(self) -> str:
        p = self.pending
        if p is None:
            return ""
        tag = "🧠" if p.model_driven else "📊"
        return (
            f"[dim]\\[{p.made_at}][/] {tag} prédiction bougie #{p.target_step} "
            f"({format_ts(p.target_ts)} UTC) → "
            f"[{_DIR_STYLE[p.direction]}]{_DIR_FR[p.direction]}[/] "
            f"{p.return_pct:+.2%} (P={p.probability:.0%})"
        )

    # -- settling against reality ---------------------------------------- #
    def settle(self, realized_price: float) -> SettledPrediction | None:
        """Score the pending prediction against the realised close price."""
        p = self.pending
        if p is None:
            return None
        # Log return — the same unit as the model's training labels and the
        # adaptive flat band (both computed on log returns). The old arithmetic
        # return was a small but avoidable train/score inconsistency.
        realized_ret = (
            math.log(realized_price / p.base_price)
            if p.base_price and realized_price > 0 else 0.0
        )
        realized_dir = classify(realized_ret, self.dir_threshold)
        correct = realized_dir == p.direction

        self.n_eval += 1
        self.n_correct += int(correct)
        self.abs_err_sum += abs(realized_ret - p.return_pct)
        self.realized_counts[realized_dir] += 1
        # Multiclass Brier of the predicted direction mass vs the realised one-hot
        # ([P_up,P_flat,P_down] order). Only meaningful when the forecast carried
        # real probabilities (model/Markov driven); a zero vector contributes the
        # one-hot's own norm (1.0), so skip the all-zero fallback.
        if any(p.probs):
            idx = {"up": 0, "flat": 1, "down": 2}[realized_dir]
            self.brier_sum += sum(
                (pr - (1.0 if i == idx else 0.0)) ** 2 for i, pr in enumerate(p.probs)
            )

        # Settle the conformal interval: the band was emitted at predict time, so
        # scoring + folding the residual in now is leak-free (predict→update).
        # Coverage is judged against the STORED emitted band (conf_lo/hi), not a
        # re-derived one — what the user saw is what gets scored.
        if p.pred_sigma > 0:
            self.conformal.update(realized_ret, 0.0, p.pred_sigma,
                                  interval=(p.conf_lo, p.conf_hi))

        settled = SettledPrediction(
            target_step=p.target_step,
            target_ts=p.target_ts,
            predicted_dir=p.direction,
            realized_dir=realized_dir,
            predicted_ret=p.return_pct,
            realized_ret=realized_ret,
            correct=correct,
        )
        self.history.append(settled)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        self.pending = None
        return settled

    def settle_log_line(self, s: SettledPrediction) -> str:
        mark = "[green]✓ juste[/]" if s.correct else "[red]✗ raté[/]"
        delta = s.realized_ret - s.predicted_ret
        return (
            f"[dim]\\[{now_hms()}][/] 🔎 bougie #{s.target_step} clôturée "
            f"({format_ts(s.target_ts)} UTC) — {mark} · "
            f"prévu [{_DIR_STYLE[s.predicted_dir]}]{_DIR_FR[s.predicted_dir]}[/] "
            f"{s.predicted_ret:+.2%} vs réel "
            f"[{_DIR_STYLE[s.realized_dir]}]{_DIR_FR[s.realized_dir]}[/] "
            f"{s.realized_ret:+.2%} (Δ={delta:+.2%}) · "
            f"fiabilité {self.accuracy:.0%} sur {self.n_eval}"
        )

    # -- metrics ---------------------------------------------------------- #
    @property
    def accuracy(self) -> float:
        """Directional hit rate — the share of candles called correctly."""
        return self.n_correct / self.n_eval if self.n_eval else 0.0

    @property
    def baseline(self) -> float:
        """Always-predict-the-most-common-direction accuracy (honest floor)."""
        if self.n_eval == 0:
            return 0.0
        return max(self.realized_counts.values()) / self.n_eval

    @property
    def edge(self) -> float:
        """Accuracy above the majority-class baseline (the real skill signal)."""
        return self.accuracy - self.baseline

    @property
    def acc_lower_bound(self) -> float:
        """Wilson 95% lower bound on directional accuracy."""
        return wilson_lower_bound(self.n_correct, self.n_eval)

    @property
    def significant_edge(self) -> bool:
        """True only when the *lower bound* of accuracy beats the baseline.

        This is the honest "the edge is real, not luck" test: even the
        pessimistic end of the confidence interval must clear the always-predict-
        majority floor, on a large-enough sample.
        """
        return self.n_eval >= _MIN_EVAL and self.acc_lower_bound > self.baseline

    @property
    def mean_abs_error(self) -> float:
        """Mean |return error| — magnitude calibration, not just direction."""
        return self.abs_err_sum / self.n_eval if self.n_eval else 0.0

    @property
    def mean_brier(self) -> float:
        """Mean multiclass Brier of the live direction probabilities (0=perfect,
        lower=better calibrated). 0 until a probabilistic forecast settles."""
        return self.brier_sum / self.n_eval if self.n_eval else 0.0

    @property
    def brier_climatology(self) -> float:
        """Brier d'un forecast constant = fréquences empiriques réalisées — le
        plancher qu'un modèle calibré SANS edge peut atteindre (1 − Σp²)."""
        if self.n_eval == 0:
            return 2.0 / 3.0
        ps = [c / self.n_eval for c in self.realized_counts.values()]
        return 1.0 - sum(p * p for p in ps)

    @property
    def reliability_note(self) -> float:
        """Note de fiabilité 0–100 — sur des cibles ATTEIGNABLES.

        L'ancienne note (50 % accuracy + 50 % edge Wilson) était plafonnée
        ~18-20/100 par construction sur un substrat sans edge directionnel :
        l'utilisateur voyait toujours « faible, ré-entraîne » quel que soit le
        modèle — une boucle sans issue (audit UX 2026-07). La nouvelle note
        mesure ce que le modèle PRÉTEND savoir faire :

        * **couverture** (45 %) — l'intervalle conforme 90 % couvre-t-il ~90 % ?
        * **calibration** (35 %) — Brier vs la climatologie (un modèle calibré
          sans edge fait ≈ la climatologie → 0.5 ; mieux → plus ; pire → 0) ;
        * **edge directionnel** (20 %) — le bonus Wilson, presque toujours 0
          ici, et c'est normal.

        Un modèle honnête et bien calibré tourne autour de 55-75 ; un modèle
        mal calibré chute sous 45 — et LÀ, ré-entraîner aide vraiment.
        """
        if self.n_eval == 0:
            return 0.0
        # Couverture de l'intervalle conforme (neutre 0.5 tant que < 20 réglés).
        if self.conformal.n_seen >= 20:
            cov_err = abs(self.conformal.empirical_coverage()
                          - (1.0 - self.conformal.target_alpha))
            cov_term = max(0.0, 1.0 - cov_err / 0.10)
        else:
            cov_term = 0.5
        # Calibration : skill de Brier vs climatologie (0 skill → 0.5).
        if self.brier_sum > 0:
            skill = 1.0 - self.mean_brier / max(self.brier_climatology, 1e-9)
            cal_term = max(0.0, min(1.0, 0.5 + 2.5 * skill))
        else:
            cal_term = 0.5
        sig_edge = max(0.0, self.acc_lower_bound - self.baseline)
        edge_term = max(0.0, min(1.0, sig_edge / 0.15))
        return round(100.0 * (0.45 * cov_term + 0.35 * cal_term + 0.20 * edge_term), 1)

    def grade(self) -> tuple[str, str, bool]:
        """``(emoji+label, style, needs_more_training)``.

        ``needs_more_training`` n'est True que quand ré-entraîner peut VRAIMENT
        aider (calibration dégradée) — pas quand l'edge directionnel est absent,
        ce qui est structurel sur ce substrat (l'ancien conseil « ré-entraîne »
        en boucle était inatteignable par construction)."""
        if self.n_eval < _MIN_EVAL:
            return f"⏳ échantillon insuffisant ({self.n_eval}/{_MIN_EVAL})", "yellow", True
        lb_edge = self.acc_lower_bound - self.baseline
        if lb_edge >= 0.05:
            return "🟢 fiable (edge directionnel significatif)", "green", False
        note = self.reliability_note
        if note >= 65:
            return "🟢 calibré (probabilités & couverture justes)", "green", False
        if note >= 45:
            return "🟡 calibré, sans edge directionnel (attendu)", "yellow", False
        if self.edge < -0.05:
            return "🔴 mal calibré (sous la baseline) — ré-entraîner (g)", "red", True
        return "🟠 calibration dégradée — ré-entraîner peut aider (g)", "yellow", True

    def reliability_line(self) -> str:
        """Compact running-reliability string for the status strip / panels."""
        if self.n_eval == 0:
            label = "en attente de la 1ʳᵉ clôture"
            return f"fiabilité — · {label}"
        label, _style, _ = self.grade()
        brier = f" · Brier {self.mean_brier:.2f}" if self.brier_sum else ""
        return (
            f"fiabilité {self.reliability_note:.0f}/100 ({self.accuracy:.0%}) "
            f"{label} · {self.n_eval} bougies{brier}"
        )

    def conformal_line(self) -> str:
        """Calibrated next-bar interval + its realised coverage (the honest 'is a
        stated 90% really 90%?' read). Empty until a prediction is pending."""
        p = self.pending
        if p is None or p.pred_sigma <= 0:
            return ""
        cov = self.conformal.empirical_coverage()
        cov_txt = (f" · couverture réelle {cov:.0%} sur {self.conformal.n_seen}"
                   if self.conformal.n_seen >= 20 else " · (calibration en cours)")
        # Le niveau affiché est la CIBLE (la garantie qu'ACI défend à long
        # terme) ; le niveau de travail α_t, que l'ACI module pour y parvenir,
        # est montré à part pour ne pas faire passer l'un pour l'autre.
        aci = (f" · α_t {self.conformal.alpha_t:.2f}"
               if abs(self.conformal.alpha_t - self.conformal.target_alpha) > 0.005 else "")
        return (
            f"intervalle conforme {p.conf_level:.0%} bougie #{p.target_step}: "
            f"[{p.conf_lo:+.2%}, {p.conf_hi:+.2%}]{cov_txt}{aci}"
        )

    # -- end of session --------------------------------------------------- #
    def finalize(self) -> list[str]:
        """Emit the closing reliability report. Idempotent within a session."""
        if self.finalized or self.n_eval == 0:
            self.finalized = True
            return []
        self.finalized = True
        label, style, needs = self.grade()
        rc = self.realized_counts
        advice = (
            "→ la calibration s'est dégradée : ré-entraîner (touche G) peut aider."
            if needs
            else "→ calibration/couverture au rendez-vous — l'absence d'edge "
                 "directionnel est structurelle, pas un défaut d'entraînement."
        )
        return [
            "═══ Bilan Live — note de fiabilité ═══",
            (
                f"[{style}]Note {self.reliability_note:.0f}/100 — {label}[/] · "
                f"{self.n_eval} bougies évaluées"
            ),
            (
                f"Direction juste {self.accuracy:.1%} (IC95 ≥ {self.acc_lower_bound:.1%}) "
                f"vs baseline {self.baseline:.1%} (edge {self.edge:+.1%}) · "
                f"erreur moy. {self.mean_abs_error:.2%} · Brier {self.mean_brier:.3f}"
            ),
            (
                f"Réel observé ↑{rc['up']} ●{rc['flat']} ↓{rc['down']}  {advice}"
            ),
        ]

