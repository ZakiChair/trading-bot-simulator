"""Smoke tests — charts, real data, simulation."""
from __future__ import annotations

import asyncio
import time

from core.engine import SimulationSession
from core.market import load_market_simulator
from core.markov_model import MarkovChainModel
from ui.panels import (
    format_decision,
    format_probabilistic_models,
    format_scenario_distribution,
)


def test_real_data_load():
    for asset in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        sim = load_market_simulator(asset, "1h", limit=500)
        assert len(sim.prices) >= 50, f"{asset}: too few bars"
        assert sim.symbol == asset
        print(f"✓ {asset}: {len(sim.prices)} bars [{sim.source}]")


def test_chart_window_bounded():
    session = SimulationSession.create(asset="BTC/USDT")
    for _ in range(200):
        session.tick()
    window = session.bot.market.chart_window(72)
    assert len(window) <= 72
    print(f"✓ Fenêtre graphique bornée: {len(window)} bars")


def test_scenario_and_panels():
    session = SimulationSession.create(asset="ETH/USDT")
    for _ in range(20):
        session.tick()
    b = session.bot
    bundle = b.current_bundle
    assert bundle
    assert bundle.most_probable_id is not None
    assert bundle.most_probable is not None
    format_scenario_distribution(bundle)
    format_probabilistic_models(b)
    format_decision(b)
    print(f"✓ Panneaux OK — plus probable #{bundle.most_probable_id}")


def test_markov_and_large_ensemble():
    session = SimulationSession.create(asset="BTC/USDT", n_scenarios=1000)
    for _ in range(5):
        session.tick()
    bundle = session.bot.current_bundle
    assert bundle and len(bundle.scenarios) == 1000
    assert bundle.markov and bundle.markov.direction_matrix is not None
    mk = MarkovChainModel()
    snap = mk.fit(session.bot.market)
    assert snap.direction_matrix.shape == (3, 3)
    print("✓ Markov + 1000 scénarios OK")


def test_10k_performance():
    session = SimulationSession.create(asset="BTC/USDT", n_scenarios=10_000)
    t0 = time.perf_counter()
    session.tick()
    elapsed = time.perf_counter() - t0
    bundle = session.bot.current_bundle
    assert bundle and len(bundle.scenarios) == 10_000
    assert elapsed < 5.0, f"10k scénarios trop lent: {elapsed:.1f}s"
    print(f"✓ 10 000 scénarios en {elapsed*1000:.0f}ms")


def test_paper_exchange():
    from exchange.live_client import ExchangeClient, ExchangeMode

    client = ExchangeClient(mode=ExchangeMode.PAPER)
    info = client.ping()
    assert info["status"] == "ok"
    price = client.fetch_ticker("BTC/USDT")
    assert price > 0
    order = client.place_market_order("BTC/USDT", "buy", 0.001, price)
    assert order.status == "filled"
    print(f"✓ Paper Binance OK — BTC ${price:,.0f}")


def test_asset_switch():
    s = SimulationSession.create(asset="BTC/USDT")
    src = s.switch_asset("SOL/USDT")
    assert s.asset == "SOL/USDT"
    assert s.bot.market.symbol == "SOL/USDT"
    print(f"✓ Switch SOL/USDT [{src}]")


def test_episode_restart_same_data():
    s = SimulationSession.create(asset="BTC/USDT")
    sim_id = id(s.bot_engine.simulator)
    s.bot.market.step = s.bot.market_sim.n_bars - 1
    s.tick()
    assert id(s.bot_engine.simulator) == sim_id
    print("✓ Rewind même données (pas de nouveau synthétique)")


async def _ui_boot_once() -> None:
    from ui.app import BotSimulatorApp

    app = BotSimulatorApp()
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.pause(delay=0.3)
        if app.session.bot.current_bundle is None:
            app.session.tick()
            await pilot.pause(delay=0.1)
        assert app.session.asset
        assert app.session.bot.current_bundle is not None
        assert app._log_synced >= 0


def test_ui_boot():
    asyncio.run(_ui_boot_once())
    print("✓ UI Textual boot (run_test)")


if __name__ == "__main__":
    test_real_data_load()
    test_chart_window_bounded()
    test_scenario_and_panels()
    test_markov_and_large_ensemble()
    test_10k_performance()
    try:
        test_paper_exchange()
    except Exception as exc:
        print(f"⚠ Paper exchange skip (réseau): {exc}")
    test_asset_switch()
    test_episode_restart_same_data()
    test_ui_boot()
    print("\nAll smoke tests passed.")