"""Tests — politique de budget de risque (core/decision.py) + portefeuille v2.

Run: PYTHONPATH=. .venv/bin/python tests/test_decision.py
"""
from __future__ import annotations

import math

import numpy as np

from core.decision import RiskPolicy, annualize_sigma, bars_per_year
from core.portfolio import Action, Portfolio


def test_vol_targeting_scales_inverse_to_sigma():
    """σ̂ double → exposition cible ≈ moitié (avant clip). ``sigma_smooth=0``
    pour tester la formule pure (le lissage anti-churn partage un état EWMA
    entre appels successifs — testé séparément)."""
    pol = RiskPolicy(sigma_target_ann=0.35, sigma_smooth=0.0)
    lo = pol.target_exposure(sigma_bar=0.008, timeframe="1h")   # σ_ann ≈ 75 %
    hi = pol.target_exposure(sigma_bar=0.016, timeframe="1h")   # σ_ann ≈ 150 %
    assert lo.target_exposure > hi.target_exposure > 0
    assert abs(lo.raw_exposure / hi.raw_exposure - 2.0) < 1e-9
    # Marché calme → le clip long-only plafonne à 1 (pas de levier).
    calm = pol.target_exposure(sigma_bar=0.001, timeframe="1h")
    assert calm.target_exposure == 1.0
    print("✓ vol-targeting: e* ∝ 1/σ̂, plafonné à 1 (pas de levier)")


def test_sigma_smoothing_dampens_churn():
    """Le σ̂ de sizing est lissé : un spike isolé de σ̂ brut ne fait pas sauter
    l'exposition cible d'un coup (anti-churn mesuré au harnais)."""
    pol = RiskPolicy(sigma_target_ann=0.35, sigma_smooth=0.85)
    for _ in range(20):
        pol.target_exposure(sigma_bar=0.006, timeframe="1h")
    before = pol.target_exposure(sigma_bar=0.006, timeframe="1h").target_exposure
    spiked = pol.target_exposure(sigma_bar=0.018, timeframe="1h")  # spike ×3
    assert spiked.sigma_raw == 0.018
    assert spiked.sigma_bar < 0.009, "le sizing doit amortir le spike"
    drop = before - spiked.target_exposure
    assert drop < 0.35, f"chute d'exposition trop brutale: {drop:.0%}"
    print("✓ lissage σ̂: un spike isolé n'emporte pas l'exposition")


def test_regime_gate_derisks_turbulent():
    pol = RiskPolicy(sigma_target_ann=0.35)
    normal = pol.target_exposure(sigma_bar=0.006, timeframe="1h", vol_regime="normal")
    turb = pol.target_exposure(sigma_bar=0.006, timeframe="1h", vol_regime="turbulent")
    assert turb.target_exposure < normal.target_exposure
    assert turb.regime_mult == 0.5
    print("✓ porte de régime: turbulent → exposition réduite")


def test_directional_tilt_gated_by_wilson():
    """Sans edge Wilson significatif, les probs directionnelles sont IGNORÉES —
    l'abstention directionnelle est structurelle, pas un espoir."""
    pol = RiskPolicy(sigma_target_ann=0.35)
    no_edge = pol.target_exposure(
        sigma_bar=0.010, timeframe="1h", prob_up=0.9, prob_down=0.05,
        edge_significant=False,
    )
    assert no_edge.dir_tilt == 0.0, "tilt must stay closed without a Wilson edge"
    with_edge = pol.target_exposure(
        sigma_bar=0.010, timeframe="1h", prob_up=0.9, prob_down=0.05,
        edge_significant=True,
    )
    assert with_edge.dir_tilt > 0.0
    assert with_edge.dir_tilt <= pol.tilt_cap
    assert with_edge.target_exposure >= no_edge.target_exposure
    print("✓ tilt directionnel fermé par défaut, ouvert uniquement sur edge Wilson")


def test_no_trade_band_avoids_micro_rebalancing():
    pol = RiskPolicy(band=0.10, sigma_smooth=0.0)
    trace = pol.target_exposure(sigma_bar=0.008, timeframe="1h")
    e_star = trace.target_exposure
    inside = pol.decide(trace, current_exposure=e_star - 0.05)
    assert not inside.rebalanced, "un écart de 5 % dans la bande ±10 % ne doit pas trader"
    trace2 = pol.target_exposure(sigma_bar=0.008, timeframe="1h")
    outside = pol.decide(trace2, current_exposure=max(0.0, e_star - 0.3))
    assert outside.rebalanced
    assert outside.reason
    # Exécution au milieu de la bande (tampon anti-re-déclenchement), pas à e*.
    assert abs(outside.exec_exposure - (e_star - 0.05)) < 1e-9
    print("✓ bande de non-trade: pas de micro-rebalancement payant des frais")


def test_annualization_scaling():
    s = annualize_sigma(0.01, "1h")
    assert abs(s - 0.01 * math.sqrt(bars_per_year("1h"))) < 1e-12
    assert bars_per_year("1d") == 365
    print("✓ annualisation √t par timeframe")


def test_portfolio_rebalance_reaches_target_and_counts_costs():
    p = Portfolio(initial_cash=10_000.0, cash=10_000.0)
    t1 = p.rebalance(0.6, price=100.0, step=1, sigma_bar=0.01)
    assert t1 is not None and t1.action == Action.BUY
    assert abs(p.exposure(100.0) - 0.6) < 0.01, p.exposure(100.0)
    assert p.fees_paid > 0 and p.slippage_paid > 0
    # Redescendre à 0.2 : vente partielle, position conservée > 0.
    t2 = p.rebalance(0.2, price=100.0, step=2, sigma_bar=0.01)
    assert t2 is not None and t2.action == Action.SELL
    assert p.position > 0
    assert abs(p.exposure(100.0) - 0.2) < 0.01
    # Liquidation totale.
    t3 = p.rebalance(0.0, price=100.0, step=3)
    assert t3 is not None and p.position == 0.0 and p.entry_price == 0.0
    print("✓ portefeuille: rebalancement fractionnaire atteint la cible, coûts comptés")


def test_portfolio_slippage_scales_with_vol():
    p = Portfolio()
    assert p.slippage_frac(0.02) > p.slippage_frac(0.002) > 0
    assert p.cost_per_side(0.0) >= p.fee_rate
    print("✓ slippage ∝ σ̂ (exécution honnête)")


def test_legacy_execute_still_all_in_all_out():
    p = Portfolio(initial_cash=10_000.0, cash=10_000.0)
    tr = p.execute(Action.BUY, price=50.0, step=1)
    assert tr is not None and p.position > 0
    assert abs(p.exposure(50.0) - 0.95) < 0.01
    tr2 = p.execute(Action.SELL, price=55.0, step=2)
    assert tr2 is not None and p.position == 0.0
    assert tr2.pnl > 0 and tr2.pnl_pct > 0
    # Long-only: SELL sans position → HOLD (aucun trade).
    assert p.execute(Action.SELL, price=55.0, step=3) is None
    print("✓ chemin hérité execute() (tout-ou-rien) intact")


def test_equity_is_mark_to_market():
    p = Portfolio(initial_cash=10_000.0, cash=10_000.0)
    p.rebalance(1.0, price=100.0, step=1)
    eq_up = p.mark_to_market(110.0)
    eq_dn = p.mark_to_market(90.0)
    assert eq_up > p.initial_cash > eq_dn, (eq_up, eq_dn)
    print("✓ equity = mark-to-market (PnL latent visible)")


if __name__ == "__main__":
    test_vol_targeting_scales_inverse_to_sigma()
    test_sigma_smoothing_dampens_churn()
    test_regime_gate_derisks_turbulent()
    test_directional_tilt_gated_by_wilson()
    test_no_trade_band_avoids_micro_rebalancing()
    test_annualization_scaling()
    test_portfolio_rebalance_reaches_target_and_counts_costs()
    test_portfolio_slippage_scales_with_vol()
    test_legacy_execute_still_all_in_all_out()
    test_equity_is_mark_to_market()
    print("✓ decision + portfolio v2 tests OK")
