"""Couche de décision — politique de budget de risque (remplace ScenarioPolicy).

L'ancienne couche (``ml/scenario_selector.py``) était un empilement d'heuristiques
codées en dur autour d'un « apprentissage » structurellement vide : un biais appris
PAR INDEX de scénario Monte-Carlo (les IDs sont des tirages arbitraires régénérés
à chaque barre — un biais par ID est du bruit pur), des logits d'action dominés
par des constantes (E[r]×10, P↑×4…), une récompense = rendement 1 barre (le
signal SANS edge, cf. ``ANALYSE_CRITIQUE_MODELE.md`` §1-2), et une gate de marge
qui, sous le forecast martingale honnête (P↑≈P↓), bloquait presque tout : le bot
ne tradait pratiquement jamais, et quand il le faisait c'était l'ε-exploration.

La refonte inverse la logique : au lieu de demander au bot de deviner le signe
(impossible, mesuré), on lui donne le seul travail que les têtes VALIDÉES du
projet savent faire — **tenir un budget de risque** :

    exposition* = clip(σ_cible / σ̂, 0, e_max) × porte_régime × (1 + tilt)

* ``σ̂``  — la vol next-bar rough-vol (RFSV), la tête gagnante du shoot-out
  walk-forward NLL/QLIKE (§8.3) ;
* ``porte_régime`` — dé-risquage en régime HMM « turbulent » (le régime de vol
  est le seul état latent persistant et séparable mesuré, cf. RAPPORT_REFONTE §2) ;
* ``tilt`` — inclinaison directionnelle **fermée par défaut** : elle ne s'ouvre
  que si l'évaluateur Wilson (``core/live_eval.py``) atteste d'un edge
  directionnel *significatif* (borne inférieure IC95 > baseline). Sur ce
  substrat c'est ≈ jamais — c'est le point : l'abstention directionnelle est
  codée, pas espérée.

Le rebalancement est **conscient des coûts** : une bande de non-trade évite de
payer frais+slippage pour des micro-ajustements. Chaque décision expose sa trace
complète (``DecisionTrace``) pour que l'UI montre le *pourquoi*, pas juste le quoi.

Ce que cette couche NE fait PAS : fabriquer un edge directionnel. Son succès se
mesure sur des cibles atteignables et vérifiables (vol réalisée ≈ cible,
turnover borné, drawdown adouci en régime turbulent), validées en walk-forward
net de coûts dans ``tests/measure_model.py::_risk_overlay_eval`` — jamais
affirmées par décret.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Barres par an, par timeframe — pour annualiser la vol per-bar.
_BARS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040,
    "30m": 17_520, "1h": 8_760, "4h": 2_190, "1d": 365,
}

# Dé-risquage de régime : « turbulent » ne se contente pas d'un σ̂ plus grand
# (déjà dans le ratio σ_cible/σ̂) — la persistance du régime (≈0.8-0.9 mesurée)
# signifie que la turbulence d'aujourd'hui prédit celle de demain au-delà du
# niveau de σ̂. La porte est CONTINUE : mult = 1 − k·P(turbulent), avec la
# probabilité filtrée du HMM — une porte binaire sur l'argmax sautait de ±40 %
# d'exposition à chaque bascule près de la frontière (churn mesuré au harnais).
# L'effet net est mesuré dans _risk_overlay_eval, pas affirmé.
_TURBULENT_DERISK = 0.5           # mult = 1 − 0.5·P(turbulent) ∈ [0.5, 1]
_REGIME_MULT = {"calm": 1.0, "normal": 1.0, "turbulent": 0.5}  # repli sans probs


def bars_per_year(timeframe: str) -> int:
    return _BARS_PER_YEAR.get(timeframe, 8_760)


def annualize_sigma(sigma_bar: float, timeframe: str) -> float:
    """σ per-bar → σ annualisé (échelle √t)."""
    return float(sigma_bar * math.sqrt(bars_per_year(timeframe)))


@dataclass
class DecisionTrace:
    """Trace complète d'une décision — l'UI affiche le raisonnement, pas
    seulement l'action. Tous les termes de la formule d'exposition."""

    sigma_bar: float = 0.0          # σ̂ utilisé pour le sizing (lissé), per-bar
    sigma_raw: float = 0.0          # σ̂ rough-vol brut du tick (affichage)
    sigma_ann: float = 0.0          # σ̂ (sizing) annualisé
    sigma_target_ann: float = 0.0   # cible de vol annualisée
    raw_exposure: float = 0.0       # σ_cible/σ̂ avant clip/portes
    regime: str = "normal"          # régime HMM (calm/normal/turbulent)
    regime_mult: float = 1.0        # porte de régime appliquée
    dir_tilt: float = 0.0           # inclinaison directionnelle (0 sauf edge Wilson)
    edge_significant: bool = False  # la porte Wilson est-elle ouverte ?
    target_exposure: float = 0.0    # e* final ∈ [0, e_max]
    current_exposure: float = 0.0   # e avant rebalancement
    exec_exposure: float = 0.0      # cible d'exécution (bord de bande)
    gap: float = 0.0                # e* − e
    band: float = 0.0               # bande de non-trade
    rebalanced: bool = False        # a-t-on tradé cette barre ?
    est_cost: float = 0.0           # coût estimé du rebalancement (fraction d'equity)
    reason: str = ""                # phrase courte pour le journal / l'UI

    def formula_line(self) -> str:
        """Formule instanciée — reproduit EXACTEMENT le calcul de e* terme à
        terme (l'ancienne version affichait clip(ratio)×portes alors que le
        code clippait le produit : en vol basse le lecteur qui recalculait
        tombait sur un autre chiffre que e*)."""
        capped = min(self.raw_exposure, 1.0)
        ratio_txt = f"{self.sigma_target_ann:.0%}/{self.sigma_ann:.0%}"
        if self.raw_exposure > 1.0:
            ratio_txt = f"min({ratio_txt}, 1)"
        return (
            f"e* = {ratio_txt} [{capped:.0%}]"
            f" × {self.regime_mult:.2f} ({self.regime})"
            f" × (1{self.dir_tilt:+.2f}) = {self.target_exposure:.0%}"
        )


@dataclass
class RiskPolicy:
    """Politique déterministe de budget de risque (vol-targeting + régime +
    porte Wilson + bande de coûts). Aucun paramètre « appris » sur le bruit :
    chaque terme vient d'une tête validée ou d'un coût observable."""

    sigma_target_ann: float = 0.35   # budget de vol annualisé (plafond de risque)
    e_max: float = 1.0               # long-only, pas de levier
    band: float = 0.10               # bande de non-trade sur |e*−e|
    tilt_scale: float = 1.0          # ampleur du tilt quand la porte Wilson s'ouvre
    tilt_cap: float = 0.5            # |tilt| max
    # Lissage EWMA du σ̂ pour le SIZING. Le σ̂ rough-vol est volontairement
    # nerveux (optimisé pour la densité 1 barre) ; l'utiliser brut pour dimen-
    # sionner fait osciller e* à travers la bande et le portefeuille churne
    # (mesuré : 90 trades / 400 barres, ~3.4 % d'equity partis en coûts, tout
    # l'écart vs le benchmark statique). Le sizing utilise un σ̂ lissé ; le σ̂
    # brut reste celui du cône/des intervalles (la densité ne change pas).
    sigma_smooth: float = 0.85       # λ EWMA (≈ demi-vie 4 barres) ; 0 = brut
    _sigma_state: float | None = field(default=None, repr=False)
    # Même lissage pour P(turbulent) : la probabilité filtrée du HMM sature vite
    # (quasi binaire, |ΔP| ≈ 0.10/barre mesuré) — brute, elle re-churnait la
    # porte de régime que la continuité venait d'adoucir.
    _pturb_state: float | None = field(default=None, repr=False)
    last_trace: DecisionTrace | None = field(default=None, repr=False)

    def reset_state(self) -> None:
        self._sigma_state = None
        self._pturb_state = None
        self.last_trace = None

    def target_exposure(
        self,
        *,
        sigma_bar: float,
        timeframe: str,
        vol_regime: str = "normal",
        p_turbulent: float | None = None,
        prob_up: float | None = None,
        prob_down: float | None = None,
        edge_significant: bool = False,
    ) -> DecisionTrace:
        """Calcule e* et sa trace. ``p_turbulent`` = probabilité filtrée HMM du
        régime turbulent (porte continue) ; ``prob_up/down`` = masses calibrées
        du modèle (utilisées seulement si ``edge_significant`` — porte Wilson)."""
        raw_sigma = float(max(sigma_bar, 1e-6))
        if self.sigma_smooth > 0 and self._sigma_state is not None:
            sigma_bar = self.sigma_smooth * self._sigma_state + (1 - self.sigma_smooth) * raw_sigma
        else:
            sigma_bar = raw_sigma
        self._sigma_state = sigma_bar
        sig_ann = annualize_sigma(sigma_bar, timeframe)
        raw = self.sigma_target_ann / max(sig_ann, 1e-9)
        # Plafonner le RATIO avant les portes (la formule documentée/affichée :
        # e* = clip(σ_cible/σ̂, 0, e_max) × porte_régime × (1+tilt)). L'ancien
        # code clippait le PRODUIT : dès que σ̂ était assez bas pour que
        # ratio × porte > 1, le dé-risquage de régime était silencieusement
        # neutralisé — un P(turbulent)=1 pouvait laisser e*=100 %. La porte de
        # turbulence doit toujours mordre sur l'exposition effectivement prise.
        raw_capped = min(raw, self.e_max)
        if p_turbulent is not None:
            pt = float(np.clip(p_turbulent, 0.0, 1.0))
            if self.sigma_smooth > 0 and self._pturb_state is not None:
                pt = self.sigma_smooth * self._pturb_state + (1 - self.sigma_smooth) * pt
            self._pturb_state = pt
            mult = 1.0 - _TURBULENT_DERISK * pt
        else:
            mult = _REGIME_MULT.get(vol_regime, 1.0)

        tilt = 0.0
        if edge_significant and prob_up is not None and prob_down is not None:
            # Inclinaison ∝ marge directionnelle calibrée — plafonnée. Long-only :
            # un tilt négatif réduit l'exposition (il ne shorte pas).
            tilt = float(np.clip(self.tilt_scale * (prob_up - prob_down),
                                 -self.tilt_cap, self.tilt_cap))

        e_star = float(np.clip(raw_capped * mult * (1.0 + tilt), 0.0, self.e_max))
        return DecisionTrace(
            sigma_bar=sigma_bar,
            sigma_raw=raw_sigma,
            sigma_ann=sig_ann,
            sigma_target_ann=self.sigma_target_ann,
            raw_exposure=raw,
            regime=vol_regime,
            regime_mult=mult,
            dir_tilt=tilt,
            edge_significant=bool(edge_significant),
            target_exposure=e_star,
        )

    def decide(
        self,
        trace: DecisionTrace,
        current_exposure: float,
        *,
        cost_per_side: float = 0.001,
    ) -> DecisionTrace:
        """Applique la bande de non-trade : rebalancer seulement si l'écart
        |e*−e| dépasse ``band`` — et alors seulement JUSQU'AU BORD de la bande
        (règle du trading minimal) : on ne paie que le trajet nécessaire pour
        rentrer dans la zone de tolérance, pas le trajet complet vers e* qu'un
        prochain tick de σ̂ annulerait. Complète la trace (gap, coût, raison)."""
        e = float(np.clip(current_exposure, 0.0, self.e_max))
        gap = trace.target_exposure - e
        trace.current_exposure = e
        trace.gap = gap
        trace.band = self.band
        if abs(gap) > self.band:
            # Cible d'exécution = demi-bande à l'intérieur de la zone de
            # tolérance. Le bord exact laissait l'exposition pile à la frontière
            # et chaque micro-mouvement de σ̂ re-déclenchait un trade (mesuré :
            # 109 jambes / 400 barres) ; la demi-bande laisse un tampon.
            exec_target = trace.target_exposure - math.copysign(self.band * 0.5, gap)
            exec_target = float(np.clip(exec_target, 0.0, self.e_max))
            trace.exec_exposure = exec_target
            trace.est_cost = abs(exec_target - e) * cost_per_side
            trace.rebalanced = True
            sens = "hausse" if gap > 0 else "baisse"
            trace.reason = (
                f"rebalance {e:.0%}→{exec_target:.0%} (bord de bande, {sens}, "
                f"σ̂ {trace.sigma_ann:.0%} vs budget {trace.sigma_target_ann:.0%}, "
                f"régime {trace.regime})"
            )
        else:
            trace.exec_exposure = e
            trace.est_cost = 0.0
            trace.rebalanced = False
            trace.reason = (
                f"hold — écart {gap:+.0%} dans la bande ±{self.band:.0%} "
                f"(coût de {abs(gap) * cost_per_side * 1e4:.0f} bps évité)"
            )
        self.last_trace = trace
        return trace
