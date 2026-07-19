"""Tests — exécution par exposition (politique de risque) et forecast next-candle."""
from __future__ import annotations

from core.engine import SimulationSession
from core.portfolio import Action, Portfolio
from core.run_mode import RunMode
from core.scenarios import ScenarioEngine


def test_resolve_action_long_only():
    assert Portfolio.resolve_action(Action.BUY, False) == Action.BUY
    assert Portfolio.resolve_action(Action.SELL, False) == Action.HOLD
    assert Portfolio.resolve_action(Action.BUY, True) == Action.HOLD
    assert Portfolio.resolve_action(Action.SELL, True) == Action.SELL
    assert Portfolio.resolve_action(Action.HOLD, True) == Action.HOLD


def test_risk_policy_trades_and_stays_long_only():
    """Le bot rebalance (il fait quelque chose !) mais avec un turnover borné
    par la bande de non-trade, et l'exposition reste dans [0, 1]."""
    s = SimulationSession.create(n_scenarios=100)
    s.set_run_mode(RunMode.TRAIN)
    for _ in range(200):
        s.tick()
        e = s.bot.portfolio.exposure(s.bot.market.price)
        assert -1e-9 <= e <= 1.0 + 1e-9, f"exposition hors [0,1]: {e}"
    b = s.bot
    assert len(b.portfolio.trades) >= 1, "la politique de risque doit rebalancer au moins une fois"
    # Bande de non-trade : le turnover doit rester très en dessous de 1/barre.
    assert b.metrics.turnover < 0.5, f"turnover excessif: {b.metrics.turnover:.2f}"
    # La position ne devient jamais négative (long-only).
    assert b.portfolio.position >= 0.0


def test_next_candle_forecast():
    s = SimulationSession.create(asset="BTC/USDT")
    s.tick()
    bundle = s.bot.current_bundle
    assert bundle is not None
    fc = bundle.next_candle
    assert fc is not None
    assert fc.most_probable.probability > 0
    assert abs(fc.prob_up + fc.prob_flat + fc.prob_down - 1.0) < 0.05
    assert "Bougie" in fc.summary_line()
    assert fc.backtest_hit_rate >= 0


def test_next_candle_timeframe_agnostic():
    session = SimulationSession.create()
    state = session.bot.market
    for tf in ("1m", "1h", "1d"):
        engine = ScenarioEngine(n_scenarios=100, timeframe=tf)
        bundle = engine.generate(state)
        assert bundle.timeframe == tf
        assert bundle.next_candle is not None
        assert bundle.next_candle.timeframe == tf


def test_forecast_log_on_step():
    s = SimulationSession.create()
    before = len(s.bot.log)
    s.tick()
    after = len(s.bot.log)
    assert after > before
    # Either a training-step forecast or a Live prediction line names the candle.
    assert any("bougie" in m.lower() for m in s.bot.log)


if __name__ == "__main__":
    test_resolve_action_long_only()
    test_risk_policy_trades_and_stays_long_only()
    test_next_candle_forecast()
    test_next_candle_timeframe_agnostic()
    test_forecast_log_on_step()
    print("✓ position + forecast tests OK")
