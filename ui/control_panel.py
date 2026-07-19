"""Top info bar (horizontal state) + keyboard help modal.

Every action is driven by the keyboard. The old left rail is gone: live state
now reads as a single horizontal strip at the top of the screen, and the
shortcut cheat-sheet is one keypress away (``h``) in a modal overlay.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

# (key, label) reference — kept in sync with BotSimulatorApp.BINDINGS.
_KEYS_SIM = [
    ("espace", "Pause / reprise"),
    ("a", "Auto (boucle)"),
    ("s", "Avancer 1 pas"),
    ("r", "Reset session"),
]
_KEYS_MODEL = [
    ("e", "Entraîner 1 épisode"),
    ("g", "Entraîner le modèle (historique)"),
    ("m", "Mode : train ↔ paper ↔ live"),
    ("o", "Renforcement en ligne ON/OFF"),
]
_KEYS_DATA = [
    ("c", "Actif suivant (crypto)"),
    ("f", "Timeframe suivant"),
    ("x", "Source données suivante"),
    ("n", "Nb de scénarios suivant"),
    ("d", "Vitesse suivante"),
]
_KEYS_VIEW = [
    ("1-5", "Onglets Marché…Apprentissage"),
    ("b", "Taille du graphique prix"),
    ("l", "Layout suivant"),
    ("p", "Config affichage"),
    ("t", "Thème suivant"),
    ("v", "Bougies ↔ ligne continue"),
    ("h", "Cette aide"),
    ("q", "Quitter"),
]


def info_bar_markup(
    *,
    asset: str,
    timeframe: str,
    exchange_mode: str,
    n_scenarios: int,
    tick_speed: float,
    data_source: str,
    layout_preset: str,
    theme_name: str,
    model_status: dict,
    most_prob_id: int | None,
) -> str:
    """One-line horizontal config strip (config role; runtime is the status bar).

    Fields are kept compact so the line fits; CSS clips any overflow.
    """
    src = (
        "live" if data_source.startswith("live")
        else "ccxt" if data_source.startswith("ccxt")
        else "cache" if data_source.startswith("cache")
        else "synth"
    )
    ex_icon = {"simulation": "📊", "paper": "📝", "live": "🔴"}.get(exchange_mode, "●")
    speed = "pas-à-pas" if tick_speed <= 0 else f"{tick_speed:g}s"

    trained = model_status.get("trained", False)
    val = model_status.get("val_accuracy", 0.0)
    # Behaviour axis (run mode) — the single "mode". Shown apart from the SOURCE
    # chip so the two axes never read as the same control.
    run_mode = model_status.get("run_mode") or (
        "live" if model_status.get("predict_mode") else "train"
    )
    mode_txt = {
        "train": "[$warning]🎓 Entraîn.[/]",
        "paper": "[b magenta]🧪 Paper[/]",
        "live": "[b $success]🔮 LIVE[/]",
    }.get(run_mode, "[$warning]🎓 Entraîn.[/]")
    n_samp = int(model_status.get("n_samples", 0))
    if not trained:
        model_txt = "[$text-muted]○ non entraîné (g)[/]"
    elif n_samp and n_samp < 300:
        # Une « val 100 % » sur quelques dizaines de barres est du
        # sur-ajustement trivial — le badge doit porter le doute avec lui.
        model_txt = f"[$warning]🧠 val {val:.0%} (n={n_samp} ⚠)[/]"
    else:
        model_txt = f"[$success]🧠 val {val:.0%} (n={n_samp:,})[/]"
    star = f" ★#{most_prob_id}" if most_prob_id is not None else ""

    parts = [
        f"[b $primary]{asset.replace('/USDT', '')}[/] [$accent]{src}[/]",
        f"[b]{timeframe}[/]",
        f"src {ex_icon} [b]{exchange_mode}[/]",
        f"[b]{n_scenarios:,}[/] scén.{star}",
        f"⏱ [b]{speed}[/]",
        mode_txt,
        model_txt,
        f"layout [b]{layout_preset}[/]",
        f"[$accent]◆ {theme_name}[/]",
        "[dim]h = aide[/]",
    ]
    return "  [dim]│[/]  ".join(parts)


def _keys_markup() -> str:
    """Static shortcut reference (theme-agnostic markup)."""
    def block(title: str, rows: list[tuple[str, str]]) -> list[str]:
        out = [f"[b $primary]{title}[/]"]
        for key, label in rows:
            out.append(f"  [b $accent]{key:>6}[/]  {label}")
        return out

    lines: list[str] = []
    lines += block("SIMULATION", _KEYS_SIM)
    lines.append("")
    lines += block("MODÈLE", _KEYS_MODEL)
    lines.append("")
    lines += block("DONNÉES", _KEYS_DATA)
    lines.append("")
    lines += block("AFFICHAGE", _KEYS_VIEW)
    return "\n".join(lines)


class HelpScreen(ModalScreen[None]):
    """Keyboard cheat-sheet overlay — opened with ``h``, closed with any key."""

    CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-dialog {
        width: 56;
        height: auto;
        max-height: 90%;
        border: heavy $primary;
        border-title-color: $primary;
        border-title-style: bold;
        background: $panel;
        padding: 1 2;
        scrollbar-size-vertical: 1;
    }
    #help-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape,h,q,space,enter", "dismiss_help", "Fermer")]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-dialog") as box:
            box.border_title = "⌨  Raccourcis clavier"
            yield Static(_keys_markup())
            yield Static("[dim]Échap / h pour fermer[/]", id="help-hint")

    def action_dismiss_help(self) -> None:
        self.dismiss(None)

    def on_key(self, event) -> None:
        # Any key closes the overlay (the binding covers the common ones; this
        # catches the rest so the help never traps the user).
        event.stop()
        self.dismiss(None)
