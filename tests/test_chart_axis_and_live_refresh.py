"""Regression tests for the price-chart real-date axis + Live-mode data refresh.

Run:  PYTHONPATH=. .venv/bin/python tests/test_chart_axis_and_live_refresh.py

Covers:
  - PriceChart X-axis carries REAL date/time labels (from market timestamps)
    for both candle and line styles, on data that has timestamps.
  - It falls back to bar indices on synthetic data (no timestamps).
  - Entering Live mode (m) on a LiveMarketFeed force-refreshes the feed so the
    bot predicts the candle forming on the freshest bars.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("BOT_UI_CONFIG", "/tmp/_bot_chart_axis_test.json")
# Isolation des poids persistés : _test_entering_live_force_refreshes entraîne
# un modèle sur un FAUX feed de 120 barres — sans cette redirection il écrasait
# le vrai models/BTC_USDT_1h.npz avec un jouet « val 100 % » (bug reproduit
# 2 fois, audit 2026-07-02).
os.environ.setdefault("BOT_MODELS_DIR", tempfile.mkdtemp(prefix="bot-models-test-"))

from core.live_feed import LiveMarketFeed  # noqa: E402
from core.market import load_market_simulator  # noqa: E402
from core.timeutil import format_axis_ts, timeframe_ms  # noqa: E402
from exchange.live_client import ExchangeMode  # noqa: E402
from ui.app import BotSimulatorApp  # noqa: E402
from ui.charts import PriceChart  # noqa: E402

# Match a real axis label like "13/06 09h" or "13/06/26".
_REAL_DATE = re.compile(r"\d{2}/\d{2}(/\d{2}|\s+\d{1,2}h)")


def _axis_line(pc: PriceChart) -> str:
    out = pc.plt.build()
    clean = [re.sub(r"\x1b\[[0-9;]*m", "", ln) for ln in out.splitlines()]
    return clean[-1] if clean else ""


async def _test_real_dates_on_candles_and_line() -> None:
    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.pause()
        # Freeze the background tick so it can't rewind the market step we pin.
        app.session.auto_train = False
        app.session.bot.paused = True
        sim = load_market_simulator("BTC/USDT", "1h")
        if sim.timestamps is None:
            print("• skip real-date test (no cached data with timestamps)")
            return
        app.session.bot_engine.simulator = sim
        app.session.bot.market = sim.state_at(sim.n_bars - 1)
        app.session.bot.scenario_engine.timeframe = "1h"
        pc = app.query_one(PriceChart)

        pc.set_style("candles")
        pc.update_data(app.session.bot)
        cand_axis = _axis_line(pc)
        assert _REAL_DATE.search(cand_axis), f"candle axis has no real date: {cand_axis!r}"
        # Last label must name the most recent bar (it's the newest close time).
        newest = format_axis_ts(int(sim.timestamps[-1]), "1h")
        assert newest.split()[0] in cand_axis, f"newest {newest!r} not on axis {cand_axis!r}"

        pc.set_style("line")
        pc.update_data(app.session.bot)
        line_axis = _axis_line(pc)
        assert _REAL_DATE.search(line_axis), f"line axis has no real date: {line_axis!r}"
        assert newest.split()[0] in line_axis, f"newest {newest!r} not on line axis {line_axis!r}"
    print("✓ price chart X-axis shows real dates (candles + line)")


async def _test_synthetic_falls_back_to_indices() -> None:
    from core.market import MarketSimulator

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.pause()
        app.session.auto_train = False
        app.session.bot.paused = True
        # A genuinely synthetic simulator carries no timestamps (SIMULATION mode
        # actually loads cached REAL data, so build one explicitly here).
        synth = MarketSimulator(n_bars=400, seed=7, symbol="SYNTH/USDT")
        assert synth.timestamps is None, "synthetic simulator should have no timestamps"
        app.session.bot_engine.simulator = synth
        app.session.bot.market = synth.state_at(synth.n_bars - 1)
        pc = app.query_one(PriceChart)
        pc.set_style("line")
        pc.update_data(app.session.bot)
        axis = _axis_line(pc)
        assert not _REAL_DATE.search(axis), f"synthetic line should use indices, got {axis!r}"
    print("✓ synthetic data falls back to bar-index axis (no fake dates)")


# ---- Live-mode force refresh ---------------------------------------------- #
class _FakeClient:
    exchange_id = "binance"

    def __init__(self, n: int, tf: str = "1h") -> None:
        self.tf = tf
        self.tf_ms = timeframe_ms(tf)
        self._now = 1_700_000_000_000
        self._last_open = self._now - (self._now % self.tf_ms)
        self._n = n
        self.fetch_calls = 0

    def now_ms(self) -> int:
        return self._now

    def ping(self):
        return {"status": "ok"}

    def _frame(self, n: int) -> pd.DataFrame:
        opens = [self._last_open - (n - i) * self.tf_ms for i in range(n)]
        opens.append(self._last_open)
        ts = np.array(opens, dtype=np.int64)
        closes = 100.0 + np.arange(len(ts), dtype=float)
        return pd.DataFrame(
            {
                "open": closes, "high": closes + 1, "low": closes - 1,
                "close": closes, "volume": np.ones(len(ts)),
            },
            index=pd.to_datetime(ts, unit="ms", utc=True),
        )

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=500):
        self.fetch_calls += 1
        return self._frame(self._n)

    def advance_one_candle(self) -> None:
        self._now += self.tf_ms
        self._last_open += self.tf_ms
        self._n += 1


def _test_force_refresh_pulls_latest() -> None:
    client = _FakeClient(n=60)
    feed = LiveMarketFeed(symbol="BTC/USDT", timeframe="1h", client=client)
    n0 = feed.n_bars
    # A candle closes; force_refresh must append it even within the interval
    # (maybe_refresh would skip it because _last_refresh is fresh).
    client.advance_one_candle()
    assert feed.maybe_refresh() is False, "interval should suppress maybe_refresh"
    assert feed.force_refresh() is True, "force_refresh must pull regardless of interval"
    assert feed.n_bars == n0 + 1, "force_refresh did not append the freshly-closed bar"
    print("✓ force_refresh pulls latest bars regardless of interval")


def _test_entering_live_force_refreshes() -> None:
    """set_predict_mode(True) on a live feed calls force_refresh so the first
    prediction is on the freshest forming candle."""
    from core.engine import SimulationSession
    from unittest.mock import patch

    # Build a session whose simulator is a fake LiveMarketFeed, with a trained
    # model so predict mode is allowed.
    sess = SimulationSession.create(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    client = _FakeClient(n=120)
    feed = LiveMarketFeed(symbol="BTC/USDT", timeframe="1h", client=client)
    sess.bot_engine.simulator = feed
    sess.bot.market = feed.state_at(feed.n_bars - 1)
    # Make the model look trained so set_predict_mode proceeds.
    sess.train_model(epochs=60, lr=0.4)

    calls = {"n": 0}
    real_force = feed.force_refresh

    def _spy():
        calls["n"] += 1
        return real_force()

    with patch.object(feed, "force_refresh", _spy):
        on = sess.set_predict_mode(True)
    assert on is True, "predict mode should engage with a trained model"
    assert calls["n"] >= 1, "entering Live did not force_refresh the live feed"
    print("✓ entering Live mode force-refreshes the live feed")


async def _main() -> None:
    await _test_real_dates_on_candles_and_line()
    await _test_synthetic_falls_back_to_indices()
    _test_force_refresh_pulls_latest()
    _test_entering_live_force_refreshes()
    print("✓ chart-axis + live-refresh tests OK")


if __name__ == "__main__":
    asyncio.run(_main())
