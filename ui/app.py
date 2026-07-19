"""Terminal UI — themes, tabs, configurable panels."""
from __future__ import annotations

import traceback

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Footer,
    Header,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from config.ui_settings import (
    PRICE_CHART_CANDLES,
    PRICE_CHART_LINE,
    PANEL_DEFS,
    UISettings,
)
from core.engine import SimulationSession
from exchange.live_client import ExchangeMode
from ui.charts import EquityChart, LearningChart, PriceChart, ScenarioHeatmap
from ui.colors import ThemeStyles
from ui.control_panel import HelpScreen, info_bar_markup
from ui.display_modal import DisplayConfigScreen
from ui.panels import (
    bubble_chart_title,
    format_bubble_chart,
    format_decision,
    format_honesty,
    format_metrics,
    format_probabilistic_models,
    format_scenario_distribution,
    format_status_strip,
    format_top_scenarios,
    format_trades,
)
from ui.themes import CUSTOM_THEMES, DEFAULT_THEME, THEME_CYCLE

PANEL_WIDGETS: dict[str, list[str]] = {
    "status_strip": ["status-strip"],
    "price_chart": ["price-chart"],
    "scenario_dist": ["scenario-dist"],
    "scenario_bubbles": ["scenario-bubbles"],
    "scenario_table": ["top-scenarios"],
    "scenario_heatmap": ["scenario-heatmap"],
    "brain": ["brain-panel"],
    "prob_models": ["prob-models-panel"],
    "metrics": ["metrics-panel"],
    "equity_chart": ["equity-chart"],
    "learning_chart": ["learning-chart"],
    "trades": ["trades-panel"],
    "log": ["log-bar"],
    "controls": ["info-bar"],
}


class BotSimulatorApp(App):
    TITLE = "🤖 Probabilistic Trading Bot Simulator"
    SUB_TITLE = "Rough-vol + HMM + conformal · Budget de risque"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("space", "toggle_pause", "Pause", priority=True),
        Binding("s", "step_once", "Step", priority=True),
        Binding("e", "train_episode", "Train ép.", priority=True),
        Binding("g", "train_model", "Entraîner modèle", priority=True),
        Binding("m", "cycle_run_mode", "Mode T/P/L", priority=True),
        Binding("o", "toggle_online_learn", "Renforcement", priority=True),
        Binding("a", "auto_toggle", "Auto", priority=True),
        Binding("r", "reset", "Reset", priority=True),
        Binding("c", "cycle_asset", "Actif", priority=True),
        Binding("f", "cycle_timeframe", "Timeframe", priority=True),
        Binding("x", "cycle_exchange", "Source", priority=True),
        Binding("n", "cycle_scenarios", "Scénarios", priority=True),
        Binding("d", "cycle_speed", "Vitesse", priority=True),
        Binding("b", "cycle_boogie_size", "Taille graph", priority=True),
        Binding("l", "cycle_layout", "Layout", priority=True),
        Binding("p", "open_display_config", "Panels", priority=True),
        Binding("t", "cycle_theme", "Thème", priority=True),
        Binding("v", "toggle_price_style", "Bougies/Ligne", priority=True),
        Binding("h", "show_help", "Aide", priority=True),
        Binding("q", "quit", "Quit", priority=True),
        Binding("1", "show_tab('tab-market')", "Marché", show=False, priority=True),
        Binding("2", "show_tab('tab-scenarios')", "Scénarios", show=False, priority=True),
        Binding("3", "show_tab('tab-models')", "Modèles", show=False, priority=True),
        Binding("4", "show_tab('tab-equity')", "Equity", show=False, priority=True),
        Binding("5", "show_tab('tab-learning')", "Learn", show=False, priority=True),
    ]

    # Cycle orders for the keyboard data controls.
    SCENARIO_STEPS = [100, 500, 1000, 2500, 5000, 10000]
    SPEED_STEPS = [0.1, 0.25, 0.35, 0.6, 0.0]
    # (history window in bars, plot height) presets for the price chart. Larger
    # windows show more history; the chart aggregates bars into composite candles
    # so they stay legible (no cramming). Heights stay compact (<= 13 rows).
    BOOGIE_SIZES = [(60, 8), (110, 9), (190, 11), (300, 13)]



    NARROW_WIDTH = 110

    # Static border titles for the framed panels. Every dashboard widget now
    # carries its title in its OWN border (Textual border_title) — the same
    # treatment the charts use — instead of a Rich Panel nested inside the CSS
    # border (which double-framed) or a separate label row above it. The bubble
    # panel's title is dynamic and set in _refresh_ui.
    _PANEL_TITLES = {
        "#scenario-dist": "🎲 Distribution scénarios",
        "#brain-panel": "🎯 Décision de risque",
        "#prob-models-panel": "⚙ Pipeline probabiliste",
        "#metrics-panel": "⚡ Métriques",
        "#trades-panel": "📋 Rebalancements",
        "#top-scenarios": "🏆 Top scénarios",
        "#honesty-panel": "🔬 Honnêteté du modèle",
        "#log": "📜 Journal",
    }

    def _init_panel_titles(self) -> None:
        for sel, title in self._PANEL_TITLES.items():
            try:
                self.query_one(sel).border_title = title
            except Exception:
                pass

    def __init__(
        self,
        n_scenarios: int = 100,
        exchange_mode: ExchangeMode | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ui_settings = UISettings.load()
        if exchange_mode is None:
            exchange_mode = ExchangeMode(self.ui_settings.exchange_mode)
        self._exchange_mode = exchange_mode
        self.session = self._create_session(n_scenarios, exchange_mode)
        self._timer = None
        self._theme_idx = 0
        self._log_synced = 0
        self._log_list_id: int | None = None
        self._live_fallback: bool = False

    def _create_session(
        self,
        n_scenarios: int,
        exchange_mode: ExchangeMode,
    ) -> SimulationSession:
        try:
            self._live_fallback = False
            return SimulationSession.create(
                n_scenarios=n_scenarios,
                asset=self.ui_settings.asset,
                timeframe=self.ui_settings.timeframe,
                exchange_mode=exchange_mode,
            )
        except Exception:
            self._live_fallback = True
            return SimulationSession.create(
                n_scenarios=n_scenarios,
                asset=self.ui_settings.asset,
                timeframe=self.ui_settings.timeframe,
                exchange_mode=ExchangeMode.SIMULATION,
            )

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="info-bar")
        yield Static("", id="status-strip")
        with Horizontal(id="body"):
            with Vertical(id="center"):
                with TabbedContent(id="main-tabs"):
                    with TabPane("Marché", id="tab-market"):
                        with Vertical(id="tab-market-layout"):
                            yield PriceChart()
                            with VerticalScroll(id="market-scroll"):
                                # La décision de risque EN PREMIER : c'est ce que
                                # le bot fait maintenant et pourquoi — elle était
                                # reléguée sous la ligne de flottaison du scroll.
                                yield Static("", id="brain-panel", classes="panel-box")
                                yield Static("", id="scenario-bubbles", classes="panel-box")
                                yield Static("", id="scenario-dist", classes="panel-box")
                    with TabPane("Scénarios", id="tab-scenarios"):
                        yield Static("", id="top-scenarios", classes="panel-box")
                        yield ScenarioHeatmap()
                    with TabPane("Modèles", id="tab-models"):
                        with VerticalScroll(id="models-scroll"):
                            yield Static("", id="prob-models-panel", classes="panel-box")
                    with TabPane("Equity", id="tab-equity"):
                        yield EquityChart()
                    with TabPane("Apprentissage", id="tab-learning"):
                        with Vertical(id="tab-learning-layout"):
                            yield LearningChart()
                            yield Static("", id="honesty-panel", classes="panel-box")
            with Vertical(id="sidebar"):
                yield Static("", id="metrics-panel", classes="panel-box")
                yield Static("", id="trades-panel", classes="panel-box")
        with Vertical(id="log-bar"):
            yield RichLog(id="log", highlight=True, markup=True, max_lines=400)
        yield Footer()

    def on_mount(self) -> None:
        for theme in CUSTOM_THEMES:
            try:
                self.register_theme(theme)
            except Exception:
                pass

        want = self.ui_settings.theme or DEFAULT_THEME
        try:
            self.theme = want
            self._theme_idx = THEME_CYCLE.index(want) if want in THEME_CYCLE else 0
        except Exception:
            self.theme = DEFAULT_THEME

        try:
            self.query_one(PriceChart).set_style(self.ui_settings.price_chart_style)
        except Exception:
            pass
        self._init_panel_titles()
        self._apply_boogie_size()

        self.session.tick()

        self.session.auto_train = True
        speed = self.ui_settings.tick_speed
        if speed > 0:
            self._timer = self.set_interval(speed, self._tick)

        self._apply_panel_visibility()
        self._sync_control_panel()
        self._refresh_ui()
        self._update_narrow_mode()

        log = self.query_one(RichLog)
        n = self.session.bot_engine.n_scenarios
        style = "bougies" if self.ui_settings.price_chart_style == PRICE_CHART_CANDLES else "ligne"
        fallback_note = (
            "\n[yellow]⚠ Connexion live indisponible — repli simulation historique[/]"
            if self._live_fallback
            else ""
        )
        log.write(
            f"[bold]Bot probabiliste prêt[/] — {n:,} scénarios "
            "(cône rough-vol Student-t · régime HMM · calibration conforme · budget de risque).\n"
            f"Actif: [b]{self.session.asset}[/b] [{self.session.data_source}] "
            f"· mode [b]{self.session.exchange_mode.value}[/b]{fallback_note}\n"
            f"Graphique prix: [b]{style}[/b] — [dim]v[/] pour basculer bougies ↔ ligne.\n"
            "Tout au [b]clavier[/] : [b]c[/] actif · [b]f[/] timeframe · [b]x[/] exchange · "
            "[b]n[/] scénarios · [b]d[/] vitesse · [b]l[/] layout.\n"
            "Onglets [b]1-5[/] · [b]e[/] replay épisode · [b]g[/] entraîner modèle · [b]m[/] mode T/P/L.\n"
            "État courant en haut de l'écran · [b]h[/] pour la liste des raccourcis."
        )

    def on_resize(self, event: events.Resize) -> None:
        self._update_narrow_mode()

    def _update_narrow_mode(self) -> None:
        if self.size.width < self.NARROW_WIDTH:
            self.add_class("narrow")
        else:
            self.remove_class("narrow")

    # Charts that ARE the sole content of a tab: hiding them via the dashboard
    # panel toggle would blank the whole tab (the "Apprentissage tab is empty"
    # bug). Their tab membership already gates their visibility, so never let
    # the panel flag add `.hidden` to them. (The heatmap shares its tab with the
    # top-scenarios table, so it stays toggleable and is NOT listed here.)
    _TAB_RESIDENT_WIDGETS = frozenset({"equity-chart", "learning-chart"})

    def _apply_panel_visibility(self) -> None:
        for pid, wids in PANEL_WIDGETS.items():
            visible = self.ui_settings.is_visible(pid)
            for wid in wids:
                if wid in self._TAB_RESIDENT_WIDGETS:
                    # Always shown within its tab; ignore the dashboard flag.
                    try:
                        self.query_one(f"#{wid}").remove_class("hidden")
                    except Exception:
                        pass
                    continue
                try:
                    w = self.query_one(f"#{wid}")
                    w.set_class(not visible, "hidden")
                except Exception:
                    pass

    def _sync_control_panel(self, most_prob_id: int | None = None) -> None:
        if not self.ui_settings.is_visible("controls"):
            return
        try:
            self.query_one("#info-bar", Static).update(
                info_bar_markup(
                    asset=self.session.asset,
                    timeframe=self.session.timeframe,
                    exchange_mode=self.session.exchange_mode.value,
                    n_scenarios=self.session.bot_engine.n_scenarios,
                    tick_speed=self.ui_settings.tick_speed,
                    data_source=self.session.data_source,
                    layout_preset=self.ui_settings.layout_preset,
                    theme_name=self.theme,
                    model_status=self.session.model_status(),
                    most_prob_id=most_prob_id,
                )
            )
        except Exception:
            pass

    def _refresh_data_badge(self) -> None:
        self._sync_control_panel()

    def check_action(self, action: str, parameters: tuple) -> bool:
        """Bloque les raccourcis d'app quand une modale est ouverte.

        Les 21 bindings sont ``priority=True`` (ils doivent primer sur le focus
        des widgets), mais sans cette garde ils traversaient les écrans modaux :
        « d » changeait la vitesse derrière l'aide, « h » empilait une deuxième
        aide, « q » quittait sous la modale (audit 2026-07)."""
        from textual.screen import ModalScreen

        if isinstance(self.screen, ModalScreen):
            return False
        return True

    def _safe_update(self, selector: str, make) -> None:
        """Rend ``make()`` et met à jour le widget — le FORMATTEUR est dans le
        try : une exception d'un panneau dégrade ce panneau, pas tout l'écran
        (l'ancienne version n'attrapait que l'update, pas le rendu)."""
        try:
            self.query_one(selector).update(make() if callable(make) else make)
        except Exception:
            try:
                self.query_one(selector).update(
                    f"[red]panneau en erreur — voir journal[/]\n[dim]{traceback.format_exc(limit=1)}[/]"
                )
            except Exception:
                pass

    def _sync_log(self) -> None:
        if not self.ui_settings.is_visible("log"):
            return
        try:
            log_w = self.query_one(RichLog)
            log_list = self.session.bot.log
            # bot.log is a brand-new list after every new_episode (restart, asset/
            # timeframe/exchange switch, train…). When the list object changes,
            # restart syncing from its start so a stale index can't drop messages.
            if id(log_list) != self._log_list_id:
                self._log_list_id = id(log_list)
                self._log_synced = 0
            for msg in log_list[self._log_synced :]:
                log_w.write(msg)
            self._log_synced = len(log_list)
        except Exception:
            pass

    def _tick(self) -> None:
        if self.session.bot.paused or not self.session.auto_train:
            return
        if self.ui_settings.tick_speed <= 0:
            return
        try:
            self.session.tick()
            self._sync_log()
            self._refresh_ui()
        except Exception:
            self.query_one(RichLog).write(f"[red]{traceback.format_exc()}[/]")

    def _refresh_ui(self) -> None:
        bot = self.session.bot
        hist = self.session.history
        active_tab = self._current_tab_id()
        theme_styles = ThemeStyles(self)

        if self.ui_settings.is_visible("status_strip"):
            self._safe_update(
                "#status-strip",
                format_status_strip(
                    bot, self.session.auto_train, theme_styles, self.session.paper_verdict
                ),
            )

        if self.ui_settings.is_visible("price_chart"):
            try:
                self.query_one(PriceChart).update_data(bot)
            except Exception:
                pass
        # Tab-resident charts (equity / learning / heatmap / honesty) live inside
        # the TabbedContent. They are updated only when THEIR tab is active (plus
        # on tab switch, which calls _refresh_ui): rebuilding every plotext chart
        # of every hidden tab at each tick saturated the event loop (~15 ms de
        # refresh + ~50-90 ms de tick pour un timer à 100 ms — audit 2026-07).
        if active_tab in (None, "tab-equity"):
            try:
                self.query_one(EquityChart).update_data(bot, hist)
            except Exception:
                pass
        if active_tab in (None, "tab-learning"):
            try:
                self.query_one(LearningChart).update_data(
                    hist, self.session.train_report, model=bot.candle_model
                )
            except Exception:
                pass
            self._safe_update(
                "#honesty-panel",
                lambda: format_honesty(
                    bot, self.session.train_report, theme_styles,
                    online=self.session.model_status(),
                    paper_verdict=self.session.paper_verdict,
                ),
            )
        if active_tab in (None, "tab-scenarios"):
            try:
                self.query_one(ScenarioHeatmap).update_data(bot)
            except Exception:
                pass

        if self.ui_settings.is_visible("metrics"):
            self._safe_update("#metrics-panel", lambda: format_metrics(bot, theme_styles))
        if self.ui_settings.is_visible("trades"):
            self._safe_update("#trades-panel", lambda: format_trades(bot, styles=theme_styles))

        bundle = bot.current_bundle
        on_market = active_tab in (None, "tab-market")
        if bundle:
            if bundle.next_candle and on_market and self.ui_settings.is_visible("scenario_bubbles"):
                span = self.session.return_span
                self._safe_update(
                    "#scenario-bubbles",
                    lambda: format_bubble_chart(
                        bundle.next_candle,
                        styles=theme_styles,
                        return_range=(-span, span),
                        live_eval=getattr(bot, "live_eval", None),
                    ),
                )
                try:
                    self.query_one("#scenario-bubbles").border_title = bubble_chart_title(
                        bundle.next_candle
                    )
                except Exception:
                    pass
            if on_market and self.ui_settings.is_visible("scenario_dist"):
                self._safe_update(
                    "#scenario-dist", lambda: format_scenario_distribution(bundle, theme_styles)
                )
            if active_tab in (None, "tab-scenarios") and self.ui_settings.is_visible("scenario_table"):
                self._safe_update(
                    "#top-scenarios", lambda: format_top_scenarios(bundle, styles=theme_styles)
                )
        if on_market and self.ui_settings.is_visible("brain"):
            self._safe_update(
                "#brain-panel", lambda: format_decision(bot, theme_styles, history=hist)
            )
        if active_tab in (None, "tab-models") and self.ui_settings.is_visible("prob_models"):
            self._safe_update(
                "#prob-models-panel", lambda: format_probabilistic_models(bot, theme_styles)
            )

        sym = self.session.asset.replace("/USDT", "")
        n_sc = bot.scenario_engine.n_scenarios
        trace = bot.risk.last_trace
        expo_hint = f" · expo {trace.current_exposure:.0%}→{trace.target_exposure:.0%}" if trace else ""
        self.sub_title = (
            f"{sym} · {n_sc:,} scén. · Step {bot.market.step}{expo_hint} · "
            f"{'▶' if self.session.auto_train else '⏸'}"
        )

        try:
            mp_id = bundle.most_probable_id if bundle else None
            self._sync_control_panel(most_prob_id=mp_id)
        except Exception:
            pass

        self._restore_tab(active_tab)

    def _current_tab_id(self) -> str | None:
        try:
            return self.query_one("#main-tabs", TabbedContent).active
        except Exception:
            return None

    def _restore_tab(self, tab_id: str | None) -> None:
        if not tab_id:
            return
        try:
            tabs = self.query_one("#main-tabs", TabbedContent)
            if tabs.active != tab_id:
                tabs.active = tab_id
        except Exception:
            pass

    # ---- keyboard data-control cycles (replace the old mouse widgets) ------
    def _apply_speed(self, speed: float) -> None:
        self.ui_settings.tick_speed = speed
        if self._timer:
            self._timer.stop()
        if speed > 0:
            self._timer = self.set_interval(speed, self._tick)
        self.ui_settings.save()

    def action_cycle_speed(self) -> None:
        cur = self.ui_settings.tick_speed
        steps = self.SPEED_STEPS
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - cur))
        nxt = steps[(idx + 1) % len(steps)]
        self._apply_speed(nxt)
        self._sync_control_panel()
        label = "pas-à-pas" if nxt <= 0 else f"{nxt:g}s"
        self.notify(f"Vitesse: {label}", timeout=2)

    def _apply_boogie_size(self) -> None:
        idx = self.ui_settings.boogie_size_idx % len(self.BOOGIE_SIZES)
        window, rows = self.BOOGIE_SIZES[idx]
        try:
            self.query_one(PriceChart).set_size(window, rows)
            self.query_one(PriceChart).update_data(self.session.bot)
        except Exception:
            pass

    def action_cycle_boogie_size(self) -> None:
        self.ui_settings.boogie_size_idx = (
            self.ui_settings.boogie_size_idx + 1
        ) % len(self.BOOGIE_SIZES)
        self.ui_settings.save()
        self._apply_boogie_size()
        window, rows = self.BOOGIE_SIZES[self.ui_settings.boogie_size_idx % len(self.BOOGIE_SIZES)]
        self.action_show_tab("tab-market")
        self.notify(f"Graphique: {window} barres · hauteur {rows}", timeout=2)

    def action_cycle_scenarios(self) -> None:
        cur = self.session.bot_engine.n_scenarios
        steps = self.SCENARIO_STEPS
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - cur))
        nxt = steps[(idx + 1) % len(steps)]
        self._change_scenario_count(nxt)

    def action_cycle_layout(self) -> None:
        from config.ui_settings import LAYOUT_PRESETS, CUSTOM_LAYOUT

        order = list(LAYOUT_PRESETS.keys()) + [CUSTOM_LAYOUT]
        cur = self.ui_settings.layout_preset
        idx = order.index(cur) if cur in order else 0
        # Skip landing on "custom" via cycling — it's reached by toggling panels.
        nxt = order[(idx + 1) % len(order)]
        if nxt == CUSTOM_LAYOUT:
            nxt = order[0]
        self.ui_settings.apply_preset(nxt)
        self._apply_panel_visibility()
        self._sync_control_panel()
        self.ui_settings.save()
        self._refresh_ui()
        self.notify(f"Layout: {nxt}", timeout=2)

    def action_cycle_asset(self) -> None:
        from config.market_config import ASSETS

        cur = self.session.asset
        idx = ASSETS.index(cur) if cur in ASSETS else 0
        self._switch_asset(ASSETS[(idx + 1) % len(ASSETS)])

    def action_cycle_timeframe(self) -> None:
        from config.market_config import TIMEFRAMES

        cur = self.session.timeframe
        idx = TIMEFRAMES.index(cur) if cur in TIMEFRAMES else 0
        self._switch_timeframe(TIMEFRAMES[(idx + 1) % len(TIMEFRAMES)])

    def action_cycle_exchange(self) -> None:
        order = [ExchangeMode.SIMULATION, ExchangeMode.PAPER, ExchangeMode.LIVE]
        cur = self.session.exchange_mode
        idx = order.index(cur) if cur in order else 0
        self._change_exchange_mode(order[(idx + 1) % len(order)])

    @work(thread=True)
    def _change_exchange_mode(self, mode: ExchangeMode) -> None:
        log = self.query_one(RichLog)
        self.call_from_thread(log.write, f"[cyan]Connexion {mode.value}…[/]")
        try:
            source = self.session.set_exchange_mode(mode)
            self.ui_settings.exchange_mode = mode.value
            self.ui_settings.save()
            self.call_from_thread(log.write, f"[green]Exchange {mode.value} — {source}[/]")
        except Exception as exc:
            # Échec de connexion (ex. LIVE sans clés API) : message clair, pas
            # un traceback brut ; set_exchange_mode ne mute plus l'état avant
            # d'avoir réussi, donc la session reste cohérente sur son mode.
            self.call_from_thread(
                log.write,
                f"[yellow]⚠ Source {mode.value} indisponible : {exc} — "
                f"la session reste sur « {self.session.exchange_mode.value} ».[/]",
            )
            self.call_from_thread(
                lambda: self.notify(f"Source {mode.value} indisponible", severity="warning", timeout=3)
            )
        finally:
            self.call_from_thread(self._refresh_data_badge)
            self.call_from_thread(self._refresh_ui)

    @work(thread=True)
    def _change_scenario_count(self, n: int) -> None:
        n = max(100, min(10_000, n))
        if n == self.session.bot_engine.n_scenarios:
            return
        log = self.query_one(RichLog)
        self.call_from_thread(log.write, f"[cyan]{n:,} scénarios…[/]")
        try:
            risk = self.session.bot.risk
            prev_run_mode = self.session.run_mode
            prev_online = self.session.online_learn
            # Rebuild the session on the SAME asset/timeframe/exchange and keep
            # the learned model — previously this silently reset to BTC/sim and
            # dropped the model, which looked like "can't change scenarios".
            self.session = SimulationSession.create(
                n_scenarios=n,
                asset=self.session.asset,
                timeframe=self.session.timeframe,
                exchange_mode=self.session.exchange_mode,
            )
            self.session.bot.risk = risk
            self.session.online_learn = prev_online
            # Restaurer l'axe de comportement : presser n en LIVE/PAPER
            # retombait silencieusement en TRAIN (la session de prédiction en
            # cours était perdue et le bot se remettait à trader le replay).
            from core.run_mode import RunMode as _RM

            if prev_run_mode != _RM.TRAIN:
                self.session.set_run_mode(prev_run_mode)
            self.session.tick()
            self.call_from_thread(
                log.write,
                f"[green]{n:,} scénarios activés — cône rough-vol Student-t (martingale)[/]",
            )
            self.call_from_thread(self._refresh_data_badge)
            self.call_from_thread(self._refresh_ui)
            self.call_from_thread(lambda: self.notify(f"{n:,} scénarios", timeout=2))
        except Exception:
            self.call_from_thread(log.write, f"[red]{traceback.format_exc()}[/]")

    @work(thread=True)
    def _switch_timeframe(self, timeframe: str) -> None:
        log = self.query_one(RichLog)
        self.call_from_thread(log.write, f"[cyan]Timeframe {timeframe}…[/]")
        try:
            source = self.session.switch_timeframe(timeframe)
            self.ui_settings.timeframe = timeframe  # persiste (comme l'actif)
            self.ui_settings.save()
            self.call_from_thread(log.write, f"[green]{timeframe} chargé — {source}[/]")
            self.call_from_thread(self._refresh_data_badge)
            self.call_from_thread(self._refresh_ui)
        except Exception:
            self.call_from_thread(log.write, f"[red]{traceback.format_exc()}[/]")

    @work(thread=True)
    def _switch_asset(self, asset: str) -> None:
        log = self.query_one(RichLog)
        self.call_from_thread(log.write, f"[cyan]Chargement {asset}…[/]")
        try:
            source = self.session.switch_asset(asset)
            self.ui_settings.asset = asset
            self.ui_settings.save()
            self.call_from_thread(log.write, f"[green]{asset} chargé — {source}[/]")
            self.call_from_thread(self._refresh_data_badge)
            self.call_from_thread(self._refresh_ui)
        except Exception:
            self.call_from_thread(log.write, f"[red]{traceback.format_exc()}[/]")

    def action_show_tab(self, tab_id: str) -> None:
        try:
            self.query_one("#main-tabs", TabbedContent).active = tab_id
            # Les widgets lourds ne sont rafraîchis que sur leur onglet actif ;
            # à l'arrivée sur l'onglet, pousser un rendu à jour immédiatement.
            self._refresh_ui()
        except Exception:
            pass

    def action_toggle_price_style(self) -> None:
        chart = self.query_one(PriceChart)
        style = chart.toggle_style()
        self.ui_settings.price_chart_style = (
            PRICE_CHART_CANDLES if style == PRICE_CHART_CANDLES else PRICE_CHART_LINE
        )
        self.ui_settings.save()
        self.action_show_tab("tab-market")
        self._refresh_ui()
        label = "bougies colorées" if style == PRICE_CHART_CANDLES else "ligne continue"
        self.notify(f"Graphique prix: {label}", timeout=2)

    def action_cycle_theme(self) -> None:
        self._theme_idx = (self._theme_idx + 1) % len(THEME_CYCLE)
        name = THEME_CYCLE[self._theme_idx]
        try:
            self.theme = name
            self.ui_settings.theme = name
            self.ui_settings.save()
            self._sync_control_panel()
            self._refresh_ui()
            self.notify(f"Thème: {name}", timeout=2)
        except Exception:
            pass

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_open_display_config(self) -> None:
        def on_done(_applied: bool) -> None:
            self._apply_panel_visibility()
            self._sync_control_panel()
            self.ui_settings.save()
            self._refresh_ui()

        self.push_screen(DisplayConfigScreen(self.ui_settings), on_done)

    def action_toggle_pause(self) -> None:
        self.session.bot.paused = not self.session.bot.paused
        self._refresh_ui()

    def action_step_once(self) -> None:
        # Le pas manuel doit marcher PENDANT la pause — c'est même son seul
        # usage naturel (figer puis avancer barre par barre). bot.paused
        # court-circuite TradingBot.step, donc on le lève le temps d'un tick.
        bot = self.session.bot
        was_paused = bot.paused
        bot.paused = False
        try:
            self.session.tick()
        finally:
            bot.paused = was_paused
        self._sync_log()
        self._refresh_ui()

    def action_auto_toggle(self) -> None:
        self.session.auto_train = not self.session.auto_train
        self.session.bot.paused = False
        self._refresh_ui()

    def action_reset(self) -> None:
        n = self.session.bot_engine.n_scenarios
        risk = self.session.bot.risk
        risk.reset_state()  # repartir d'un lissage σ̂ propre sur la nouvelle marche
        # Conserver actif / timeframe / source de données : r réinitialise la
        # SESSION (livre, marche), pas la configuration — l'ancien create() nu
        # ramenait BTC/1h/simulation quel que soit l'état courant.
        asset = self.session.asset
        timeframe = self.session.timeframe
        exchange_mode = self.session.exchange_mode
        try:
            self.session = SimulationSession.create(
                n_scenarios=n,
                asset=asset,
                timeframe=timeframe,
                exchange_mode=exchange_mode,
            )
        except Exception:
            # Source live injoignable au moment du reset → repli historique,
            # même configuration d'actif/timeframe.
            self.session = SimulationSession.create(
                n_scenarios=n,
                asset=asset,
                timeframe=timeframe,
                exchange_mode=ExchangeMode.SIMULATION,
            )
        # Le bot frais porte déjà le modèle disque + l'attache exchange ; on ne
        # transplante QUE la politique de risque (paramètres, état déjà reset).
        self.session.bot.risk = risk
        self.session.tick()
        self.query_one(RichLog).clear()
        self._log_synced = 0
        self.query_one(RichLog).write(
            f"[cyan]Session réinitialisée — {asset} {timeframe} "
            f"[{self.session.data_source}] (politique de risque conservée)[/]"
        )
        self._sync_log()
        self._refresh_ui()

    @work(thread=True)
    def action_train_episode(self) -> None:
        log = self.query_one(RichLog)
        self.call_from_thread(log.write, "[bold]Replay épisode (budget de risque)…[/]")
        try:
            stats = self.session.train_episode(steps=400)
            for msg in stats.get("log", self.session.bot.log[-5:]):
                self.call_from_thread(log.write, msg)
            self.call_from_thread(
                log.write,
                f"[green]Épisode {stats['episode']} — PnL {stats['pnl']:+,.0f} · "
                f"vol réalisée {stats['realized_vol']:.0%} (cible {stats['target_vol']:.0%}) · "
                f"expo moy. {stats['avg_exposure']:.0%} · coûts {stats['fees']:,.0f}$[/]",
            )
            self.call_from_thread(self._refresh_ui)
        except Exception:
            self.call_from_thread(log.write, f"[red]{traceback.format_exc()}[/]")

    @work(thread=True)
    def action_train_model(self) -> None:
        """Train the next-candle model by gradient descent on the loaded history."""
        log = self.query_one(RichLog)
        self.call_from_thread(
            log.write,
            "[bold]🧠 Entraînement du modèle — descente de gradient sur l'historique…[/]",
        )

        def progress(epoch: int, loss: float, val_acc: float) -> None:
            if epoch % 100 == 0:
                self.call_from_thread(
                    log.write, f"[dim]  epoch {epoch:4d} · perte {loss:.4f} · val {val_acc:.1%}[/]"
                )

        try:
            rep = self.session.train_model(epochs=400, lr=0.5, progress=progress)
            for msg in self.session.bot.log[-3:]:
                self.call_from_thread(log.write, msg)
            if rep.n_train:
                self.call_from_thread(
                    self.notify,
                    f"Modèle entraîné — val {rep.val_accuracy:.0%} · passage en mode LIVE",
                    timeout=3,
                )
                self.call_from_thread(self.action_show_tab, "tab-learning")
            self.call_from_thread(self._refresh_ui)
        except Exception:
            self.call_from_thread(log.write, f"[red]{traceback.format_exc()}[/]")

    def action_cycle_run_mode(self) -> None:
        from core.run_mode import RunMode

        mode = self.session.cycle_run_mode()
        self._sync_log()
        self._refresh_ui()
        msg = {
            RunMode.TRAIN: "🎓 Entraînement — le bot rejoue l'historique et apprend",
            RunMode.PAPER: "🧪 Paper — rentable ou non : politique gelée, net de frais "
            "(backtest sur cache, paper-trading sur live)",
            RunMode.LIVE: "🔮 Live — estime la prochaine bougie et s'auto-évalue",
        }[mode]
        self.notify(msg, timeout=3)

    def action_toggle_online_learn(self) -> None:
        on = self.session.toggle_online_learn()
        self._sync_log()
        self._refresh_ui()
        self.notify(
            "🔄 Renforcement en ligne ACTIVÉ — chaque bougie clôturée ajuste le "
            "modèle (suivi de régime + calibration, pas d'edge directionnel)"
            if on
            else "Renforcement en ligne coupé — retour aux poids entraînés (figés)",
            timeout=3,
        )