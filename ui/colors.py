"""Theme-aware colour helpers for Rich markup and plotext."""
from __future__ import annotations

from rich.color import Color as RichColor
from rich.style import Style
from textual.color import Color
from textual.widget import Widget


def theme_hex(widget: Widget, var: str, fallback: str = "#888888") -> str:
    try:
        app = widget.app
        if app is None:
            return fallback
        return app.theme_variables.get(var) or fallback
    except Exception:
        return fallback


def theme_rgb(widget: Widget, var: str, fallback: str = "#888888") -> tuple[int, int, int]:
    return Color.parse(theme_hex(widget, var, fallback)).rgb


class ThemeStyles:
    """Semantic Rich styles from the active Textual theme."""

    def __init__(self, widget: Widget) -> None:
        self._w = widget

    def hex(self, var: str, fallback: str = "#888888") -> str:
        return theme_hex(self._w, var, fallback)

    def rich(self, var: str, fallback: str = "#888888", *, bold: bool = False) -> str:
        rgb = Color.parse(self.hex(var, fallback)).rgb
        style = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        return f"bold {style}" if bold else style

    def style(self, var: str, fallback: str = "#888888", *, bold: bool = False) -> Style:
        rgb = Color.parse(self.hex(var, fallback)).rgb
        return Style(color=RichColor.from_rgb(*rgb), bold=bold)

    @property
    def positive(self) -> str:
        return self.rich("success", "#3FB950")

    @property
    def negative(self) -> str:
        return self.rich("error", "#F85149")

    @property
    def warning(self) -> str:
        return self.rich("warning", "#D29922")

    @property
    def accent(self) -> str:
        return self.rich("primary", "#4C9AFF", bold=True)

    @property
    def info(self) -> str:
        """Secondary highlight — replaces literal cyan/magenta on light themes."""
        return self.rich("accent", "#79C0FF")

    @property
    def heading(self) -> str:
        """Bold section heading colour (was literal 'bold cyan')."""
        return self.rich("primary", "#4C9AFF", bold=True)

    @property
    def border(self) -> str:
        """Hex for Rich Panel border_style (legible on the active theme)."""
        return self.hex("primary", "#4C9AFF")

    @property
    def muted(self) -> str:
        return self.rich("foreground", "#94A3B8")

    def dir_colors(self) -> dict[str, str]:
        """Direction → Rich style for up/flat/down, from the active theme."""
        return {"up": self.positive, "down": self.negative, "flat": self.warning}