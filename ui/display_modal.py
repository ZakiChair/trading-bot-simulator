"""Modal screen for fine-grained display configuration."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, Select, Static

from config.ui_settings import CUSTOM_LAYOUT, PANEL_DEFS, UISettings, layout_select_options


class DisplayConfigScreen(ModalScreen[bool]):
    """Choose which panels to show and apply layout presets."""

    CSS = """
    DisplayConfigScreen {
        align: center middle;
    }
    #display-dialog {
        width: 62;
        height: auto;
        max-height: 90%;
        border: heavy $primary;
        background: $panel;
        padding: 1 2;
    }
    #display-dialog .section-header {
        color: $primary;
        text-style: bold;
        margin: 1 0 0 0;
    }
    #panel-grid {
        grid-size: 2;
        grid-gutter: 0 2;
        height: auto;
        margin: 1 0;
    }
    #display-actions {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    #display-actions Button {
        margin: 0 1;
    }
    """

    def __init__(self, ui: UISettings) -> None:
        super().__init__()
        self._ui = ui
        self._snapshot = dict(ui.panels)
        self._snapshot_preset = ui.layout_preset

    def _apply_live(self) -> None:
        """Répercute immédiatement sur le dashboard — le texte de la modale
        promet « appliqués en direct », le code doit le faire (audit)."""
        try:
            self.app._apply_panel_visibility()
            self.app._sync_control_panel()
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="display-dialog"):
            yield Static("[b]Configuration de l'affichage[/b]", id="dlg-title")
            yield Static(
                "Cochez les éléments à afficher. Les changements sont appliqués en direct.",
                classes="hint",
            )
            yield Label("Preset de layout")
            yield Select(
                layout_select_options(),
                value=self._ui.layout_preset,
                id="dlg-layout",
            )
            yield Static("Panneaux visibles", classes="section-header")
            with Grid(id="panel-grid"):
                for pid, label in PANEL_DEFS.items():
                    yield Checkbox(label, value=self._ui.is_visible(pid), id=f"dlg-chk-{pid}")
            with Horizontal(id="display-actions"):
                yield Button("Appliquer", variant="success", id="dlg-apply")
                yield Button("Annuler", id="dlg-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dlg-cancel":
            # Annuler restaure panneaux ET preset (le preset changeait aussi).
            self._ui.panels = self._snapshot
            self._ui.layout_preset = self._snapshot_preset
            self._apply_live()
            self.dismiss(False)
            return
        if event.button.id == "dlg-apply":
            self._collect()
            self.dismiss(True)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "dlg-layout" and event.value:
            self._ui.apply_preset(str(event.value))
            for pid in PANEL_DEFS:
                try:
                    self.query_one(f"#dlg-chk-{pid}", Checkbox).value = self._ui.is_visible(pid)
                except Exception:
                    pass
            self._apply_live()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        pid = event.checkbox.id.replace("dlg-chk-", "") if event.checkbox.id else ""
        if pid in self._ui.panels:
            self._ui.panels[pid] = event.value
            self._ui.layout_preset = CUSTOM_LAYOUT
            # Le Select doit SUIVRE : sinon « Appliquer » relisait sa valeur
            # périmée (ex. « trading ») et ré-étiquetait un layout custom.
            try:
                sel = self.query_one("#dlg-layout", Select)
                if sel.value != CUSTOM_LAYOUT:
                    sel.value = CUSTOM_LAYOUT
            except Exception:
                pass
            self._apply_live()

    def _collect(self) -> None:
        # L'état live (self._ui) est déjà à jour à chaque interaction — ne PAS
        # ré-appliquer le preset du Select ici : quand une case avait basculé le
        # layout en « custom », la valeur périmée du Select écrasait panels ET
        # étiquette au moment d'Appliquer.
        for pid in PANEL_DEFS:
            try:
                self._ui.panels[pid] = self.query_one(f"#dlg-chk-{pid}", Checkbox).value
            except Exception:
                pass