"""Regression tests for the UI polish + honesty panel (refonte 2026-07).

Run:  PYTHONPATH=. .venv/bin/python tests/test_honesty_and_polish.py

Covers:
  - Position cell is a readable quantity + $ value, not a raw 6-dp float.
  - Trades table fits the narrow sidebar (no verbose reason spill).
  - The decision panel shows the full risk-budget reasoning.
  - format_honesty renders the structural truth in every model state, and the
    measured edge/calibration when a train report is present.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from io import StringIO
from pathlib import Path

from rich.console import Console

# Redirect persistence so pressing keys / save() can't clobber the real config.
os.environ.setdefault("BOT_UI_CONFIG", "/tmp/_bot_ui_test_settings.json")

import ml.candle_model as _cm  # noqa: E402


@contextlib.contextmanager
def _isolated_models():
    """Le test d'entraînement ne doit pas écraser models/*.npz de l'utilisateur."""
    orig = _cm.MODELS_DIR
    _cm.MODELS_DIR = Path(tempfile.mkdtemp())
    try:
        yield
    finally:
        _cm.MODELS_DIR = orig

from core.run_mode import RunMode  # noqa: E402
from exchange.live_client import ExchangeMode  # noqa: E402
from ui.app import BotSimulatorApp  # noqa: E402
from ui.colors import ThemeStyles  # noqa: E402
from ui.panels import (  # noqa: E402
    _format_position,
    format_decision,
    format_honesty,
    format_probabilistic_models,
    format_trades,
)


def _render(renderable, width: int) -> str:
    buf = StringIO()
    Console(file=buf, width=width, force_terminal=False).print(renderable)
    return buf.getvalue()


def _test_position_is_readable() -> None:
    class _P:
        position = 5.247799
    cell = _format_position(_P(), price=1669.0)
    assert "≈" in cell and "$" in cell, cell
    assert "5.247799" not in cell, f"raw float leaked: {cell}"
    # quantity rounded, value compacted to k for big notionals
    assert cell.startswith("5.25"), cell
    assert "k" in cell, cell
    print("✓ position cell is a readable quantity + $ value")


async def _test_trades_fit_sidebar() -> None:
    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        # Force training mode so the bot actually trades (a persisted model can
        # auto-switch the boot to LIVE, where it only predicts — no trades).
        app.session.set_run_mode(RunMode.TRAIN)
        for _ in range(120):
            app.session.tick()
        bot = app.session.bot
        assert bot.portfolio.trades, "the risk policy should have rebalanced by now"
        out = _render(format_trades(bot, styles=ThemeStyles(app)), width=31)
        # The verbose rebalance reason must never leak into the table cells.
        assert "σ̂" not in out and "rebalance " not in out, f"verbose reason leaked:\n{out}"
        # The exposure column shows where each leg landed.
        assert "%" in out, out
    print("✓ trades table fits the sidebar (exposure legs, no verbose spill)")


async def _test_decision_panel_shows_reasoning() -> None:
    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        app.session.set_run_mode(RunMode.TRAIN)
        for _ in range(30):
            app.session.tick()
        out = _render(
            format_decision(app.session.bot, ThemeStyles(app), history=app.session.history),
            width=92,
        )
        low = out.lower()
        assert "rough-vol" in low, out
        assert "régime" in low, out
        assert "wilson" in low, out
        assert "exposition" in low, out
        # La porte directionnelle doit être annoncée fermée (pas d'edge en replay).
        assert "fermée" in low, out
    print("✓ decision panel shows the full risk-budget reasoning (Wilson gate closed)")


async def _test_honesty_structural_truth_always() -> None:
    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        for _ in range(30):
            app.session.tick()
        bot = app.session.bot
        st = ThemeStyles(app)
        # Untrained: structural truth + verdict present, no measured row.
        out = _render(format_honesty(bot, None, st), width=92)
        assert "50/50" in out, out
        assert "rough-vol" in out.lower(), out
        assert "Verdict" in out, out
        print("✓ honesty panel shows structural truth when untrained")

        # Trained with a report: measured edge + calibration appear.
        with _isolated_models():
            rep = app.session.train_model(epochs=120, lr=0.4)
        out2 = _render(format_honesty(bot, rep, st), width=92)
        assert "holdout" in out2.lower(), out2
        assert "edge" in out2.lower(), out2
        assert "Brier" in out2 or "calibr" in out2.lower(), out2
        assert "Calibré, pas rentable" in out2, out2
        print("✓ honesty panel shows measured edge + calibration when trained")


async def _test_honesty_panel_mounted_in_tab() -> None:
    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        app._refresh_ui()
        await pilot.press("5")  # Apprentissage tab
        await pilot.pause()
        hp = app.query_one("#honesty-panel")
        lc = app.query_one("#learning-chart")
        # Both share the tab and are visible (the chart on top, panel beneath).
        assert hp.region.height > 0, "honesty panel not visible in tab"
        assert lc.region.height > 0, "learning chart not visible in tab"
        assert lc.region.y < hp.region.y, "honesty panel should sit below the curve"
        assert str(hp.border_title), "honesty panel has no border title"
    print("✓ honesty panel mounted below the learning curve on tab 5")


def _test_model_panel_matches_reality() -> None:
    """The displayed pipeline must match the code after the refonte: the cone is a
    rough-vol Student-t **martingale** feeding a risk-budget decision — and the
    stale GBM formula `S(t+1)=S(t)·exp(μΔt+σε)` must be GONE."""
    from core.bot import TradingBot
    from core.market import load_market_simulator

    sim = load_market_simulator("BTC/USDT", "1h", limit=1500)
    bot = TradingBot(sim).new_episode(episode=1)
    bot.current_bundle = bot.scenario_engine.generate(sim.state_at(1400))
    txt = _render(format_probabilistic_models(bot), 110).lower()
    assert "rough" in txt or "rfsv" in txt, "model panel must show the rough-vol head"
    assert "martingale" in txt, "model panel must state the cone is a martingale"
    assert "s(t+1) = s(t)" not in txt, "stale GBM formula still rendered — refonte contradiction"
    assert "décision" in txt, "model panel must show the decision layer"
    print("✓ model panel matches reality (rough/martingale/décision, stale GBM gone)")


async def _main() -> None:
    _test_position_is_readable()
    await _test_trades_fit_sidebar()
    await _test_decision_panel_shows_reasoning()
    await _test_honesty_structural_truth_always()
    await _test_honesty_panel_mounted_in_tab()
    _test_model_panel_matches_reality()
    print("✓ honesty + polish tests OK")


if __name__ == "__main__":
    asyncio.run(_main())
