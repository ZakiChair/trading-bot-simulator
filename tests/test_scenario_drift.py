"""Tests — the Monte-Carlo cone is a heavy-tailed martingale.

Pins two findings:

* **Martingale centre** (tests/measure_model.py::_scenario_engine_eval,
  walk-forward, no leak): a *drifted* cone made BOTH the terminal-return density
  and the directional Brier strictly worse than a zero-drift martingale (the
  §1-2 result — 1-bar direction ≈ 50/50 — surfacing in the engine). The cone is
  therefore centred at the current price by construction; the old
  ``_DRIFT_SHRINK`` drift machinery was removed (it only ever multiplied to 0).
* **Heavy tails** (tests/measure_model.py::_cone_ladder_eval): the cone draws
  Student-t innovations (ν fit per asset) sized by the rough-vol term structure,
  which fixed the 95/99 % tail under-coverage of the old Gaussian flat-σ cone
  without hurting CRPS or directional Brier. Leverage was measured to regress
  direction (§8.4) and is left OFF.
"""
from __future__ import annotations

import numpy as np

import core.scenarios as scen
from core.market import MarketState, Regime
from core.scenarios import ScenarioEngine


def _trending_state(per_bar: float = 0.012, n: int = 220) -> MarketState:
    """A steep, near-monotonic uptrend → large momentum.

    Old code drove ``base_drift = mom·0.3`` into every path, so the 12-bar cone
    would be centred far above the current price. The martingale cone must not.
    """
    rng = np.random.default_rng(7)
    log_p = np.cumsum(rng.normal(per_bar, per_bar * 0.25, n))
    prices = 100.0 * np.exp(log_p)
    return MarketState(
        prices=prices,
        volumes=np.full(n, 1000.0),
        step=n - 1,
        regime=Regime.BULL,
        symbol="TEST/USDT",
        source="synthetic",
    )


def test_drift_shrink_is_zero_by_default():
    assert scen._DRIFT_SHRINK == 0.0


def test_cone_is_martingale_under_strong_trend():
    state = _trending_state()
    assert state.momentum() > 0.05, "fixture should have strong positive momentum"

    engine = ScenarioEngine(n_scenarios=6000, horizon=12, seed=3)
    bundle = engine.generate(state)
    terms = np.array([s.terminal_return for s in bundle.scenarios])
    dirs = np.array([s.direction for s in bundle.scenarios])
    sigma = float(terms.std())

    # Cone centred at the current price: both mean and median terminal return
    # ≪ cone width, despite +momentum the old drift would have driven into the
    # centre. (We assert the *generative* density — the equal-weight MC draws —
    # which is what _scenario_engine_eval validated. The softmax-weighted masses
    # carry a separate, smaller drawdown-penalty tilt; see ANALYSE §8.4.)
    assert abs(terms.mean()) < 0.15 * sigma, (
        f"cone not centred: mean={terms.mean():.4f} vs σ={sigma:.4f}"
    )
    assert abs(float(np.median(terms))) < 0.10 * sigma, np.median(terms)
    # Equal-weight directional frequencies are ~symmetric (no momentum tilt).
    up_eq = float((dirs == "up").mean())
    down_eq = float((dirs == "down").mean())
    assert abs(up_eq - down_eq) < 0.08, f"up={up_eq:.3f} down={down_eq:.3f}"


def test_cone_innovations_unit_variance_and_fat_tailed():
    """The Student-t innovation is standardised to unit variance (raw t_ν has
    variance ν/(ν−2)), so the only thing differing from the Gaussian rung is tail
    shape at matched variance — the guard that makes the tail-coverage win real
    and not just 'a wider cone'. ν→∞ recovers the Gaussian limit."""
    from core.cone import standardized_t

    rng = np.random.default_rng(0)
    z = standardized_t(rng, 4.0, 200_000)
    assert abs(float(z.std()) - 1.0) < 0.05, f"t not unit-variance: σ={z.std():.3f}"
    kurt = float(np.mean((z - z.mean()) ** 4) / z.var() ** 2)
    assert kurt > 4.0, f"t_4 should be fat-tailed (kurtosis>4), got {kurt:.2f}"

    g = standardized_t(rng, float("inf"), 200_000)
    assert abs(float(g.std()) - 1.0) < 0.05
    gk = float(np.mean((g - g.mean()) ** 4) / g.var() ** 2)
    assert gk < 3.3, f"ν→∞ should be Gaussian (kurtosis≈3), got {gk:.2f}"


def test_cone_fit_recovers_fat_tails():
    """ConeModel.fit must recover a low ν from genuinely fat-tailed returns (and
    not flag fat tails on Gaussian returns) — proving the estimator is live."""
    from core.cone import ConeModel, standardized_t

    rng = np.random.default_rng(1)
    r_t = standardized_t(rng, 4.0, 4000) * 0.01
    p_t = 100.0 * np.exp(np.cumsum(r_t))
    m_t = ConeModel.fit(None, None, None, p_t)
    assert m_t.nu < 8.0, f"failed to recover fat tails: ν={m_t.nu}"

    r_g = rng.standard_normal(4000) * 0.01
    p_g = 100.0 * np.exp(np.cumsum(r_g))
    m_g = ConeModel.fit(None, None, None, p_g)
    assert m_g.nu > 8.0, f"flagged fat tails on Gaussian data: ν={m_g.nu}"


def test_forecast_is_equal_weight_not_drawdown_tilted():
    """prob_up/prob_down/expected_return must report the *equal-weight* generative
    density (honest forecast), not the softmax-weighted masses (which carry the
    drawdown penalty's upward tilt — §8.4, weighted dir-Brier 0.67 > equal 0.50).
    """
    assert scen._FORECAST_EQUAL_WEIGHT is True
    state = _trending_state()
    engine = ScenarioEngine(n_scenarios=6000, horizon=12, seed=5)
    bundle = engine.generate(state)
    dirs = np.array([s.direction for s in bundle.scenarios])

    # Reported masses == equal-weight path frequencies (forecast).
    assert abs(bundle.prob_up - float((dirs == "up").mean())) < 1e-9
    assert abs(bundle.prob_down - float((dirs == "down").mean())) < 1e-9
    # ...and symmetric (martingale cone), unlike the up-tilted weighted masses.
    assert abs(bundle.prob_up - bundle.prob_down) < 0.08, (
        bundle.prob_up, bundle.prob_down
    )


def test_selection_still_uses_drawdown_weighted_probability():
    """The drawdown penalty must remain in the per-scenario ``probability`` (risk
    preference for *selection*), and that weighted view stays upward-tilted —
    proving the forecast/selection split is real, not a global rewrite."""
    state = _trending_state()
    engine = ScenarioEngine(n_scenarios=6000, horizon=12, seed=5)
    bundle = engine.generate(state)
    dirs = np.array([s.direction for s in bundle.scenarios])
    probs = np.array([s.probability for s in bundle.scenarios])
    probs = probs / probs.sum()
    w_up = float(probs[dirs == "up"].sum())
    w_down = float(probs[dirs == "down"].sum())
    # Selection masses still carry the drawdown tilt (up > down) the forecast drops.
    assert w_up > w_down + 0.05, (w_up, w_down)
    assert abs(probs.sum() - 1.0) < 1e-9


if __name__ == "__main__":
    test_drift_shrink_is_zero_by_default()
    test_cone_is_martingale_under_strong_trend()
    test_cone_innovations_unit_variance_and_fat_tailed()
    test_cone_fit_recovers_fat_tails()
    test_forecast_is_equal_weight_not_drawdown_tilted()
    test_selection_still_uses_drawdown_weighted_probability()
    print("✓ scenario drift (heavy-tailed martingale cone) tests OK")
