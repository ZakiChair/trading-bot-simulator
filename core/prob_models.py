"""Probabilistic model metadata exposed to the UI."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProbModelSnapshot:
    """Parameters and formulas used for the current scenario batch."""

    # --- Modèle 1 : cône Monte-Carlo (rough-vol + Student-t, martingale) ---
    estimated_vol: float = 0.0
    estimated_momentum: float = 0.0
    regime: str = "range"
    horizon: int = 12
    n_scenarios: int = 100
    mean_path_vol: float = 0.0

    # --- Têtes validées en walk-forward (refonte) ---
    # Volatility regime from a 3-state Gaussian HMM (Baum-Welch) on smoothed
    # log-RV — calm / normal / turbulent. The honest taxonomy: 1-bar direction is
    # noise, the *volatility level* is the persistent, separable latent state. It
    # separates next-bar |return| better than the old hardcoded heuristic on
    # BTC/ETH/SOL (it is descriptive/generative, not a directional forecast).
    vol_regime: str = "normal"
    # Distribution causale filtrée sur les 3 régimes (calm/normal/turbulent) —
    # la politique de risque consomme P(turbulent) en continu (pas l'argmax).
    vol_regime_probs: dict = None
    # Student-t innovation tail (ν) fit per asset for the Monte-Carlo cone; the
    # cone is sized by the rough-vol term structure. Roughness H≈0.07.
    cone_nu: float = 0.0
    rough_hurst: float = 0.0
    # Rough-vol formula shown in the UI (the σ head).
    rough_formula: str = (
        "log σ²_{t+Δ} = (cos Hπ/π)·Δ^{H+½}·∫ log σ²_{t-u}/((u+Δ)u^{H+½})du   (RFSV, H≈0.07)"
    )
    cone_formula: str = (
        "r_k = z_k·√v_k,  z_k ~ Student-t(ν) standardisée,  v_k = structure par terme rough-vol"
    )

    # --- Vraisemblance (pick de risque, centre martingale) ---
    # Centrée à 0 (martingale) — le drift momentum a été retiré (§8.4) ; cette
    # vraisemblance ne sert qu'au *pick d'affichage* du scénario (pénalité de
    # drawdown), jamais au forecast directionnel.
    likelihood_formula: str = (
        "log L_i = -½(r_i²/σ²) - 0.3·(DD_i²/σ²)   (centre martingale)"
    )
    prob_formula: str = "P_i = softmax(log L_i)   (pondération d'affichage)"

    # --- Décision (politique de budget de risque, core/decision.py) ---
    decision_formula: str = (
        "e* = clip(σ_cible/σ̂, 0, 1) × porte_régime(HMM) × (1 + tilt·1[edge Wilson])"
    )

    most_probable_id: int = 0
    most_probable_prob: float = 0.0
