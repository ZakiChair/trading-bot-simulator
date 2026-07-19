"""Headless Pilot regression tests for the UI runtime bugs from the handoff.

Covers:
  - layout "Personnalisé → Full" actually sticks (no checkbox feedback loop)
  - tab/chart regions don't shift tick-to-tick (borders stop jumping)
  - boogie is the boot default and axes stay fixed-width
"""
from __future__ import annotations

import asyncio
import os

# Redirect persistence to a throwaway file: several tests press keys (l/d) that
# call UISettings.save(), which would otherwise overwrite the user's real
# config/ui_settings.json (this has wiped saved layouts more than once).
os.environ.setdefault("BOT_UI_CONFIG", "/tmp/_bot_ui_runtime_test.json")

from exchange.live_client import ExchangeMode
from ui.app import BotSimulatorApp, PANEL_WIDGETS
from ui.charts import PriceChart


WATCH = ["#main-tabs", "#center", "#tab-market-layout", "#market-scroll", "#price-chart"]


def _regions(app) -> dict:
    out = {}
    for sel in WATCH:
        try:
            r = app.query_one(sel).region
            out[sel] = (r.x, r.y, r.width, r.height)
        except Exception:
            out[sel] = None
    return out


async def _test_custom_to_full_sticks() -> None:
    """Keyboard layout cycling + apply_preset('full') makes every panel visible
    and the preset sticks (no feedback loop snapping it back to custom)."""
    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        ui = app.ui_settings
        # Drop into custom by hiding a panel (the modal callback path).
        ui.panels["scenario_heatmap"] = False
        from config.ui_settings import CUSTOM_LAYOUT

        ui.layout_preset = CUSTOM_LAYOUT
        app._apply_panel_visibility()
        app._sync_control_panel()
        await pilot.pause()
        assert ui.layout_preset == "custom", ui.layout_preset
        # Apply the full preset.
        ui.apply_preset("full")
        app._apply_panel_visibility()
        app._sync_control_panel()
        await pilot.pause()
        assert ui.layout_preset == "full", f"preset bounced back: {ui.layout_preset}"
        assert all(ui.is_visible(p) for p in PANEL_WIDGETS), "panels not all visible"
    print("✓ custom → full sticks")


async def _test_regions_stable() -> None:
    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        prev = _regions(app)
        shifts = 0
        for _ in range(8):
            app.session.tick()
            app._refresh_ui()
            await pilot.pause()
            cur = _regions(app)
            shifts += sum(1 for k in cur if prev.get(k) != cur.get(k))
            prev = cur
        assert shifts == 0, f"{shifts} region shifts (borders jumping)"
    print("✓ tab/chart regions stable across ticks")


async def _test_tab_nav_and_cross_tab_panels() -> None:
    """Keys 1–5 actually switch tabs (regression: query_one had a bad id kwarg),
    the tab frame stays put across switches, and cross-tab quick-panel toggles
    take effect once you reach the tab."""
    from textual.widgets import Checkbox, TabbedContent

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        tc = app.query_one("#main-tabs", TabbedContent)
        frames = set()
        for key, expected in [
            ("2", "tab-scenarios"),
            ("3", "tab-models"),
            ("4", "tab-equity"),
            ("5", "tab-learning"),
            ("1", "tab-market"),
        ]:
            await pilot.press(key)
            await pilot.pause()
            assert tc.active == expected, f"key {key} -> {tc.active}, expected {expected}"
            r = app.query_one("#main-tabs").region
            frames.add((r.x, r.y, r.width, r.height))
        assert len(frames) == 1, f"tab frame shifted across switches: {frames}"

        # A panel that lives in another tab still honours the visibility toggle.
        app.ui_settings.panels["scenario_heatmap"] = False
        app._apply_panel_visibility()
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()
        assert app.query_one("#scenario-heatmap").has_class("hidden")
    print("✓ tab nav works + frame stable + cross-tab toggles apply")


async def _test_log_survives_bot_replacement() -> None:
    """Journal messages logged after an episode restart/switch still reach the
    RichLog (regression: a stale _log_synced index dropped them when bot.log was
    replaced by a fresh list)."""
    from textual.widgets import RichLog

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        # Independent of any persisted layout preset (a prior run may have saved
        # 'minimal', which hides the log and makes _sync_log a no-op).
        app.ui_settings.panels["log"] = True
        app._apply_panel_visibility()
        rl = app.query_one(RichLog)
        app.session._restart_episode()  # replaces bot.log with a fresh empty list
        app.session.bot.log_msg("SENTINEL-RESTART-LOG")
        app._sync_log()
        await pilot.pause()
        text = " ".join(getattr(s, "text", "") for s in rl.lines)
        assert "SENTINEL-RESTART-LOG" in text, "log message lost after bot replacement"
    print("✓ journal survives episode restart")


async def _test_keyboard_data_cycles() -> None:
    """The keyboard data-control keys (c/f/x/n/d/l) cycle their values instead
    of the removed mouse Select widgets."""
    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        from config.market_config import ASSETS, TIMEFRAMES

        a0, tf0 = app.session.asset, app.session.timeframe
        await pilot.press("c")          # next asset (threaded worker)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.session.asset != a0 or len(ASSETS) == 1

        await pilot.press("f")          # next timeframe (threaded worker)
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.session.timeframe != tf0 or len(TIMEFRAMES) == 1

        sp0 = app.ui_settings.tick_speed
        await pilot.press("d")          # next speed
        await pilot.pause()
        assert app.ui_settings.tick_speed != sp0

        lp0 = app.ui_settings.layout_preset
        await pilot.press("l")          # next layout
        await pilot.pause()
        assert app.ui_settings.layout_preset != lp0
    print("✓ keyboard data cycles work")


async def _test_log_in_bottom_bar() -> None:
    """The journal now lives in the full-width #log-bar, not the right sidebar."""
    from textual.widgets import RichLog

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        bar = app.query_one("#log-bar")
        rl = app.query_one(RichLog)
        # The RichLog is a descendant of the docked bottom bar.
        node = rl
        found = False
        while node is not None:
            if node is bar:
                found = True
                break
            node = node.parent
        assert found, "log is not inside #log-bar"
    print("✓ journal docked in full-width bottom bar")


async def _test_boogie_size_cycles() -> None:
    """The 'b' key resizes the price (boogie) chart window + height."""
    from ui.charts import PriceChart

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(170, 52)) as pilot:
        await pilot.pause()
        pc = app.query_one(PriceChart)
        w0, r0 = pc._window, pc._rows
        await pilot.press("b")
        await pilot.pause()
        assert (pc._window, pc._rows) != (w0, r0), "boogie size did not change"
    print("✓ boogie size cycles with 'b'")


async def _test_learning_tab_not_blank() -> None:
    """The Apprentissage tab renders even under the default 'trading' preset
    (regression: it only updated when its dashboard panel was visible)."""
    from ui.charts import LearningChart

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(170, 52)) as pilot:
        await pilot.pause()
        for _ in range(3):
            app.session.tick(); app._refresh_ui(); await pilot.pause()
        await pilot.press("5")
        await pilot.pause()
        lc = app.query_one(LearningChart)
        # border_title gets set by every update_data branch; blank means never updated.
        assert str(lc.border_title), "learning chart never rendered"
    print("✓ Apprentissage tab renders (not blank)")


async def _test_candles_default() -> None:
    # Hermetic: candlesticks are THE default (independent of a persisted pref
    # a real session may have flipped to "line" on disk).
    from config.ui_settings import PRICE_CHART_CANDLES, UISettings

    assert UISettings().price_chart_style == PRICE_CHART_CANDLES
    assert PriceChart()._style == PRICE_CHART_CANDLES
    print("✓ candlesticks are the boot default")


async def _test_candles_fit_width_all_sizes() -> None:
    """Pressing 'b' through every size preset keeps candles LEGIBLE — i.e. each
    candle gets at least _MIN_CELLS_PER_CANDLE cells, not merely <= 1 cell.

    Regression: the old guard only ensured ``candles <= cols`` (>= 1 cell each),
    which still rendered an illegible wall of slivers at the large windows. The
    chart now aggregates the window into composite candles, so the real
    invariant is ``candles * _MIN_CELLS_PER_CANDLE <= cols``."""
    from ui.charts import _MIN_CELLS_PER_CANDLE

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        # Feed enough bars that the largest window is actually populated.
        for _ in range(max(w for w, _ in BotSimulatorApp.BOOGIE_SIZES) + 40):
            app.session.tick()
        pc = app.query_one(PriceChart)
        pc.set_style("candles")  # guarantee the candle path runs (not vacuous)
        aggregated_at_least_once = False
        for i in range(len(BotSimulatorApp.BOOGIE_SIZES)):
            app._refresh_ui()
            await pilot.pause()
            cols = pc._plot_cols()
            drawn = pc._last_candle_count
            assert drawn * _MIN_CELLS_PER_CANDLE <= cols, (
                f"size #{i}: {drawn} candles in {cols} cols is illegible "
                f"(< {_MIN_CELLS_PER_CANDLE} cells/candle)"
            )
            assert drawn <= pc._window, f"size #{i}: more candles than bars"
            assert cols >= 40, f"plot width collapsed to {cols}"
            if drawn < pc._window:
                aggregated_at_least_once = True
            await pilot.press("b")
            await pilot.pause()
        assert aggregated_at_least_once, "aggregation never engaged on any size"
    print("✓ candles stay legible (>= 3 cells each) at every 'b' size")


async def _test_help_modal_opens_and_closes() -> None:
    """'h' opens the keyboard help overlay; any key dismisses it, no exception."""
    from ui.control_panel import HelpScreen

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        await pilot.press("h")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen), "help overlay did not open"
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen), "help overlay did not close"
    print("✓ help modal opens with 'h' and closes")


async def _test_info_bar_replaces_rail() -> None:
    """The left control rail is gone; a horizontal #info-bar sits up top and
    carries the live config state."""
    from textual.widgets import Static

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        assert not app.query("#controls"), "old control rail still mounted"
        bar = app.query_one("#info-bar", Static)
        # Independent of any persisted layout that may hide the info bar.
        app.ui_settings.panels["controls"] = True
        app._apply_panel_visibility()
        app._sync_control_panel()
        await pilot.pause()
        text = str(bar.render())
        assert app.session.timeframe in text, "info bar missing config state"
        assert bar.region.height == 1, "info bar is not a 1-line horizontal strip"
    print("✓ info bar replaces the left control rail")


async def _main() -> None:
    await _test_custom_to_full_sticks()
    await _test_regions_stable()
    await _test_tab_nav_and_cross_tab_panels()
    await _test_log_survives_bot_replacement()
    await _test_keyboard_data_cycles()
    await _test_log_in_bottom_bar()
    await _test_boogie_size_cycles()
    await _test_learning_tab_not_blank()
    await _test_candles_default()
    await _test_candles_fit_width_all_sizes()
    await _test_help_modal_opens_and_closes()
    await _test_info_bar_replaces_rail()
    print("✓ UI runtime regression tests OK")


if __name__ == "__main__":
    asyncio.run(_main())
