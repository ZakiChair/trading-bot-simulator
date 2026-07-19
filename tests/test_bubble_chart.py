"""Tests — bubble chart, layout presets, price chart style."""
from __future__ import annotations

from io import StringIO

from rich.console import Console

from config.ui_settings import (
    CUSTOM_LAYOUT,
    PRICE_CHART_CANDLES,
    PRICE_CHART_LINE,
    PANEL_DEFS,
    UISettings,
)
from core.engine import SimulationSession
from ui.panels import format_bubble_chart


def _panel_text(panel) -> str:
    buf = StringIO()
    Console(file=buf, width=120, force_terminal=True).print(panel)
    return buf.getvalue()


def test_bubble_chart_original_style():
    s = SimulationSession.create(n_scenarios=100)
    s.tick()
    fc = s.bot.current_bundle.next_candle
    assert fc is not None

    panel = format_bubble_chart(fc)
    text = _panel_text(panel)
    assert "Consensus" in text
    # La bulle ★ est le pick pondéré risque — étiquetée « Scénario d'action »
    # pour ne plus concurrencer le « plus probable » calibré du consensus.
    assert "Scénario d'action" in text
    assert "Top scénarios" in text
    assert "Classement scénarios" not in text


def test_bubble_chart_axes_are_fixed():
    """Abscissa (rendement) + ordinate (probabilité) stay constant."""
    from ui.panels import _BUBBLE_PROB_CEIL

    s = SimulationSession.create(n_scenarios=100)
    s.tick()
    fc = s.bot.current_bundle.next_candle
    assert fc is not None

    span = 0.05
    t1 = _panel_text(format_bubble_chart(fc, return_range=(-span, span)))
    # X ticks reflect the FROZEN range, not the bubbles' own min/max.
    assert f"{span:+.2%}" in t1          # +5.00%
    assert f"{-span:+.2%}" in t1         # -5.00%
    # Y ordinate uses the fixed probability ceiling (top label == ceiling).
    assert f"{_BUBBLE_PROB_CEIL * 100:>3.0f}%" in t1
    # Same forecast + same range → byte-identical render (no per-frame drift).
    t2 = _panel_text(format_bubble_chart(fc, return_range=(-span, span)))
    assert t1 == t2
    # A different frozen range changes the axis labels (range is honoured).
    t3 = _panel_text(format_bubble_chart(fc, return_range=(-0.10, 0.10)))
    assert "+10.00%" in t3


def test_default_price_chart_is_candles():
    ui = UISettings()
    assert ui.price_chart_style == PRICE_CHART_CANDLES


def test_apply_preset_full_enables_all_panels():
    ui = UISettings()
    ui.layout_preset = CUSTOM_LAYOUT
    for pid in PANEL_DEFS:
        ui.panels[pid] = False
    ui.apply_preset("full")
    assert ui.layout_preset == "full"
    assert all(ui.is_visible(pid) for pid in PANEL_DEFS)


def test_normalize_layout_preset_case_insensitive():
    from config.ui_settings import normalize_layout_preset

    assert normalize_layout_preset("Full") == "full"
    assert normalize_layout_preset("Personnalisé") == CUSTOM_LAYOUT


def test_price_style_toggle_values():
    ui = UISettings()
    ui.price_chart_style = PRICE_CHART_LINE
    assert ui.price_chart_style == PRICE_CHART_LINE


if __name__ == "__main__":
    test_bubble_chart_original_style()
    test_bubble_chart_axes_are_fixed()
    test_default_price_chart_is_candles()
    test_apply_preset_full_enables_all_panels()
    test_normalize_layout_preset_case_insensitive()
    test_price_style_toggle_values()
    print("✓ bubble chart tests OK")