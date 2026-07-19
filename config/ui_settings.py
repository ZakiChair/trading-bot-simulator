"""UI preferences: themes, visible panels, layout — persisted to disk."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from config.market_config import DEFAULT_ASSET, DEFAULT_TIMEFRAME, TIMEFRAMES
from ui.themes import DEFAULT_THEME

CONFIG_DIR = Path(__file__).resolve().parent


def _config_path() -> Path:
    """Path to the persisted UI settings.

    Honours ``BOT_UI_CONFIG`` so tests (which press keys that call ``save()``)
    can redirect persistence to a throwaway file instead of clobbering the
    user's real ``config/ui_settings.json`` — a bug that has wiped saved layouts
    more than once.
    """
    override = os.environ.get("BOT_UI_CONFIG")
    return Path(override) if override else CONFIG_DIR / "ui_settings.json"


UI_CONFIG_PATH = _config_path()

# All toggleable panels/widgets in the terminal.
PANEL_DEFS: dict[str, str] = {
    "status_strip": "Bandeau statut",
    "price_chart": "Graphique prix",
    "scenario_dist": "Distribution scénarios",
    "scenario_bubbles": "Bulles prochaine bougie",
    "scenario_table": "Table top scénarios",
    "scenario_heatmap": "Heatmap scénarios",
    "brain": "Décision de risque",
    "prob_models": "Modèles probabilistes",
    "metrics": "Métriques",
    "equity_chart": "Courbe equity",
    "learning_chart": "Courbe apprentissage",
    "trades": "Historique trades",
    "log": "Journal",
    "controls": "Bandeau infos (haut)",
}

CUSTOM_LAYOUT = "custom"

LAYOUT_PRESETS: dict[str, dict[str, bool]] = {
    "trading": {
        "status_strip": True,
        "price_chart": True,
        "scenario_dist": True,
        "scenario_bubbles": True,
        "scenario_table": True,
        "scenario_heatmap": False,
        "brain": True,
        "prob_models": True,
        "metrics": True,
        "equity_chart": True,
        "learning_chart": False,
        "trades": True,
        "log": True,
        "controls": True,
    },
    "analyse": {
        "status_strip": True,
        "price_chart": True,
        "scenario_dist": True,
        "scenario_bubbles": True,
        "scenario_table": True,
        "scenario_heatmap": True,
        "brain": True,
        "prob_models": True,
        "metrics": True,
        "equity_chart": False,
        "learning_chart": True,
        "trades": False,
        "log": True,
        "controls": True,
    },
    "minimal": {
        "status_strip": True,
        "price_chart": True,
        "scenario_dist": False,
        "scenario_bubbles": False,
        "scenario_table": False,
        "scenario_heatmap": False,
        "brain": False,
        "prob_models": False,
        "metrics": True,
        "equity_chart": True,
        "learning_chart": False,
        "trades": True,
        "log": False,
        "controls": False,
    },
    "full": {k: True for k in PANEL_DEFS},
}


def layout_select_options() -> list[tuple[str, str]]:
    """Options for layout Select widgets (includes saved custom layouts)."""
    opts = [(name.capitalize(), name) for name in LAYOUT_PRESETS]
    opts.append(("Personnalisé", CUSTOM_LAYOUT))
    return opts


def normalize_layout_preset(name: str) -> str:
    key = str(name).strip().lower()
    if key in ("personnalisé", "personnalise", "perso"):
        return CUSTOM_LAYOUT
    if key in LAYOUT_PRESETS:
        return key
    if key == CUSTOM_LAYOUT:
        return CUSTOM_LAYOUT
    return "trading"


PRICE_CHART_CANDLES = "candles"  # TradingView-style colored candlesticks (default)
PRICE_CHART_LINE = "line"        # continuous price line
PRICE_CHART_BOOGIE = "boogie"    # legacy alias — migrated to candles on load


@dataclass
class UISettings:
    theme: str = DEFAULT_THEME
    panels: dict[str, bool] = field(default_factory=lambda: dict(LAYOUT_PRESETS["trading"]))
    layout_preset: str = "trading"
    tick_speed: float = 0.35
    show_footer_hints: bool = True
    chart_height: int = 12
    asset: str = DEFAULT_ASSET
    timeframe: str = DEFAULT_TIMEFRAME  # persisté — un restart retombait toujours sur 1h
    price_chart_style: str = PRICE_CHART_CANDLES
    exchange_mode: str = "paper"
    boogie_size_idx: int = 1  # index into BotSimulatorApp.BOOGIE_SIZES (default 72×11)

    @classmethod
    def load(cls) -> UISettings:
        """Chargement CHAMP PAR CHAMP avec défauts.

        L'ancien ``try`` global effaçait TOUTES les préférences dès qu'un seul
        champ était corrompu (un ``tick_speed: "vite"`` suffisait) ; et un
        ``tick_speed`` négatif était accepté tel quel — le timer ne démarrait
        jamais et l'app semblait figée au boot. Chaque champ est maintenant
        parsé/validé indépendamment ; les invalides retombent sur leur défaut.
        """
        path = _config_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                return cls()
        except Exception:
            return cls()

        def field_of(key: str, cast, default, validate=None):
            try:
                val = cast(data[key])
                if validate is not None and not validate(val):
                    return default
                return val
            except Exception:
                return default

        try:
            raw_panels = data.get("panels", {})
            panels = {**LAYOUT_PRESETS["trading"],
                      **{k: bool(v) for k, v in raw_panels.items() if k in PANEL_DEFS}}
        except Exception:
            panels = dict(LAYOUT_PRESETS["trading"])
        for pid in PANEL_DEFS:
            panels.setdefault(pid, LAYOUT_PRESETS["trading"].get(pid, True))

        style = field_of("price_chart_style", str, PRICE_CHART_CANDLES)
        # Migrate the retired "boogie" braille style to candlesticks.
        if style == PRICE_CHART_BOOGIE:
            style = PRICE_CHART_CANDLES
        if style not in (PRICE_CHART_CANDLES, PRICE_CHART_LINE):
            legacy = data.get("market_chart_mode", "")
            style = PRICE_CHART_LINE if legacy == "price" else PRICE_CHART_CANDLES

        return cls(
            theme=field_of("theme", str, DEFAULT_THEME),
            panels=panels,
            layout_preset=normalize_layout_preset(data.get("layout_preset", "trading")),
            # 0 = pas-à-pas (légitime) ; négatif/aberrant → défaut.
            tick_speed=field_of("tick_speed", float, 0.35, lambda v: 0.0 <= v <= 10.0),
            show_footer_hints=field_of("show_footer_hints", bool, True),
            chart_height=field_of("chart_height", int, 12, lambda v: 4 <= v <= 60),
            asset=field_of("asset", str, DEFAULT_ASSET, lambda v: bool(v.strip())),
            timeframe=field_of("timeframe", str, DEFAULT_TIMEFRAME,
                               lambda v: v in TIMEFRAMES),
            price_chart_style=style,
            exchange_mode=field_of("exchange_mode", str, "paper",
                                   lambda v: v in ("simulation", "paper", "live")),
            boogie_size_idx=field_of("boogie_size_idx", int, 1, lambda v: 0 <= v < 16),
        )

    def save(self) -> None:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "theme": self.theme,
                    "panels": self.panels,
                    "layout_preset": self.layout_preset,
                    "tick_speed": self.tick_speed,
                    "show_footer_hints": self.show_footer_hints,
                    "chart_height": self.chart_height,
                    "asset": self.asset,
                    "timeframe": self.timeframe,
                    "price_chart_style": self.price_chart_style,
                    "exchange_mode": self.exchange_mode,
                    "boogie_size_idx": self.boogie_size_idx,
                },
                indent=2,
            )
        )

    def apply_preset(self, name: str) -> None:
        name = normalize_layout_preset(name)
        if name == CUSTOM_LAYOUT:
            self.layout_preset = name
            return
        if name not in LAYOUT_PRESETS:
            return
        self.layout_preset = name
        self.panels = {pid: bool(val) for pid, val in LAYOUT_PRESETS[name].items()}
        for pid in PANEL_DEFS:
            self.panels.setdefault(pid, True)

    def is_visible(self, panel_id: str) -> bool:
        return self.panels.get(panel_id, True)