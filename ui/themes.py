"""Trading-desk themes + curated cycle for the bot simulator."""
from __future__ import annotations

from textual.theme import Theme

OBSIDIAN = Theme(
    name="obsidian",
    primary="#4C9AFF",
    secondary="#6CB6FF",
    accent="#79C0FF",
    foreground="#C9D1D9",
    background="#0B0E14",
    surface="#12161F",
    panel="#181D27",
    success="#3FB950",
    warning="#D29922",
    error="#F85149",
    dark=True,
)

BLOOMBERG = Theme(
    name="bloomberg",
    primary="#FF8C00",
    secondary="#FFB347",
    accent="#FFD080",
    foreground="#E8E4DC",
    background="#0A0A0A",
    surface="#111010",
    panel="#1A1814",
    success="#00C853",
    warning="#FFD700",
    error="#FF5252",
    dark=True,
)

MATRIX = Theme(
    name="matrix",
    primary="#00E676",
    secondary="#69F0AE",
    accent="#B9F6CA",
    foreground="#B8FFD0",
    background="#001A0E",
    surface="#042818",
    panel="#082C1C",
    success="#00E676",
    warning="#C6FF00",
    error="#FF5252",
    dark=True,
)

ABYSS = Theme(
    name="abyss",
    primary="#22D3EE",
    secondary="#38BDF8",
    accent="#67E8F9",
    foreground="#CBD5E1",
    background="#0C1222",
    surface="#111827",
    panel="#1E293B",
    success="#34D399",
    warning="#FBBF24",
    error="#F87171",
    dark=True,
)

EMBER = Theme(
    name="ember",
    primary="#F472B6",
    secondary="#FB7185",
    accent="#FDA4AF",
    foreground="#E7E5E4",
    background="#0C0A09",
    surface="#1C1917",
    panel="#292524",
    success="#4ADE80",
    warning="#FACC15",
    error="#F87171",
    dark=True,
)

NEON = Theme(
    name="neon",
    primary="#A78BFA",
    secondary="#C084FC",
    accent="#E879F9",
    foreground="#E2E8F0",
    background="#0F0A1A",
    surface="#1A1030",
    panel="#251845",
    success="#34D399",
    warning="#FBBF24",
    error="#FB7185",
    dark=True,
)

CUTE = Theme(
    name="cute",
    primary="#B0125F",      # deep candy rose — borders/titles/active tab, reads on white & rose
    secondary="#E84A8A",    # bubblegum pink
    accent="#7A2FB0",       # orchid — highlights, axis ticks (deepened for contrast on white)
    foreground="#3D1730",   # deep plum ink — high contrast on white cards AND rose chrome
    background="#F6BBD5",   # ROSE is the canvas (inverted: pink dominates)
    surface="#F7C9DE",      # slightly deeper rose — header / tab chrome band, separates from canvas
    panel="#FFFFFF",        # WHITE cards (inverted: white is the panel surface)
    success="#0B7A4F",      # deep mint — hausse
    warning="#9C580C",      # warm amber — neutre (deepened so it reads on white)
    error="#C8204F",        # rose-red — baisse
    dark=False,
)

CUSTOM_THEMES: list[Theme] = [OBSIDIAN, BLOOMBERG, MATRIX, ABYSS, EMBER, NEON, CUTE]

THEME_CYCLE: list[str] = [
    "obsidian",
    "cute",
    "abyss",
    "neon",
    "ember",
    "tokyo-night",
    "catppuccin-mocha",
    "dracula",
    "nord",
    "gruvbox",
    "monokai",
    "solarized-dark",
    "flexoki",
    "textual-dark",
    "bloomberg",
    "matrix",
]

DEFAULT_THEME = "obsidian"