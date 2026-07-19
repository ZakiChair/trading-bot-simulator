"""Simulation engine: training loops, asset switching, live exchange."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import dataclasses

from config.market_config import DEFAULT_ASSET, DEFAULT_TIMEFRAME, WARMUP_BARS
from core.bot import BotState, TradingBot
from core.decision import RiskPolicy
from core.live_feed import LiveMarketFeed
from core.market import MarketSimulator, load_market_simulator
from core.paper_eval import PaperVerdict, build_verdict, verdict_lines
from core.portfolio import Action
from core.run_mode import RunMode
from core.thresholds import adaptive_threshold as _compute_dir_threshold
from exchange.live_client import ExchangeClient, ExchangeMode
from ml.candle_model import TrainReport, load_model_with_note, train_candle_model

_RETURN_SPAN_K = 3.0
_RETURN_SPAN_MIN = 0.01
_RETURN_SPAN_MAX = 0.15


def _compute_return_span(prices) -> float:
    """Frozen ±half-range for the bubble-chart abscissa (rendement).

    ``k·σ`` of the loaded history's log returns, clamped — computed once per
    asset/timeframe so the chart's X axis stays constant for the whole session
    instead of rescaling every tick.
    """
    if prices is None or len(prices) < 3:
        return 0.04
    r = np.diff(np.log(np.asarray(prices, dtype=float)))
    sigma = float(np.std(r)) if len(r) else 0.01
    return float(min(_RETURN_SPAN_MAX, max(_RETURN_SPAN_MIN, _RETURN_SPAN_K * sigma)))


def _load_market(
    asset: str,
    timeframe: str,
    exchange_mode: ExchangeMode,
) -> MarketSimulator:
    if exchange_mode == ExchangeMode.SIMULATION:
        return load_market_simulator(asset, timeframe)
    return LiveMarketFeed.create(asset, timeframe, mode=exchange_mode)


def _model_loaded_line(saved) -> str:
    """Journal line for a freshly loaded model — val QUALIFIÉE par la taille
    d'échantillon (un « val 100 % » sur 66 barres est du sur-ajustement trivial,
    pas une prouesse), et le vrai chemin vers LIVE (la touche m CYCLE
    Entraînement → Paper → Live ; l'ancien hint « m pour LIVE » envoyait
    l'utilisateur sur un backtest paper non demandé)."""
    n = int(getattr(saved, "n_samples", 0))
    caveat = " — indicatif, échantillon réduit" if n < 300 else ""
    return (
        f"🧠 Modèle chargé (val {saved.val_accuracy:.0%} sur {n} barres{caveat}) "
        f"— touche m ×2 pour LIVE (m cycle Entraînement → Paper → Live)."
    )


def _anchored_model_copy(model):
    """Copie du modèle de bougie SUR SES POIDS D'ANCRE (batch validés).

    Le renforcement en ligne (mode LIVE) dérive les poids en mémoire — sur les
    mêmes bougies OOS que le backtest paper rejoue ensuite. Passer le modèle
    dérivé au bot paper contaminait le verdict « politique gelée » (drift 0.63
    mesuré après 120 bougies) et le rendait non reproductible. La copie repart
    des poids d'ancre ; le bot maison garde sa dérive pour sa session Live.
    Sans dérive en cours, le modèle est partagé tel quel (le chemin paper ne
    mute jamais les poids : ``learn=False`` et pas de _reinforce_online).
    """
    if model is None or not getattr(model, "trained", False):
        return model
    if getattr(model, "anchor_weights", None) is None or not getattr(model, "n_online", 0):
        return model
    import copy

    clone = copy.copy(model)  # shallow — les tableaux mutables sont remplacés ici
    clone.weights = model.anchor_weights.copy()
    clone.bias = model.anchor_bias.copy()
    clone.anchor_weights = model.anchor_weights.copy()
    clone.anchor_bias = model.anchor_bias.copy()
    clone.n_online = 0
    clone.online_loss_history = []
    return clone


def _attach_exchange(portfolio, client) -> None:
    """N'attache l'exécution d'ordres qu'en mode PAPER (fills simulés).

    Un client LIVE n'est JAMAIS branché sur le portefeuille : aucun run mode du
    simulateur ne passe d'ordre réel (le mode LIVE = prédiction seule) —
    l'attacher rendait des ordres réels atteignables depuis les boucles
    TRAIN/PAPER quand la source était Binance LIVE (audit 2026-07). Ceinture
    supplémentaire : ``ExchangeClient.place_market_order`` exige aussi
    ``BINANCE_ALLOW_REAL_ORDERS=1`` pour tout ordre réel.
    """
    if client is not None and client.mode == ExchangeMode.PAPER:
        portfolio.exchange = client


@dataclass
class SimulationSession:
    bot_engine: TradingBot
    bot: BotState
    asset: str = DEFAULT_ASSET
    timeframe: str = DEFAULT_TIMEFRAME
    data_source: str = ""
    exchange_mode: ExchangeMode = ExchangeMode.SIMULATION
    exchange_client: ExchangeClient | None = None
    auto_train: bool = False
    online_learn: bool = True  # Live: fold each settled candle into the model (online SGD)
    steps_per_tick: int = 1
    history: list[dict] = field(default_factory=list)
    return_span: float = 0.04  # frozen ±half-range for the bubble-chart X axis
    train_report: TrainReport | None = None  # last supervised training run (learning curve)
    run_mode: RunMode = RunMode.TRAIN  # behaviour axis: train | paper | live
    paper_verdict: PaperVerdict | None = None  # last/running paper profitability verdict
    _eval_start: int = -1  # first bar of the current Live walk-forward session
    _home_bot: BotState | None = None  # train/live bot stashed while a paper bot is active
    _paper_started: bool = False
    _paper_done: bool = False
    _paper_forward: bool = False  # live forward paper-trade vs cached OOS backtest
    _paper_start: int = -1  # first bar of the current paper session
    _paper_last_trades: int = 0  # trade count at the last logged running verdict
    # Segment d'equity : incrémenté à chaque REMPLACEMENT du bot (épisode,
    # switch, aller-retour paper). L'EquityChart ne trace que le segment
    # courant — sinon la courbe collait bout à bout deux portefeuilles
    # différents (ex. livre TRAIN à 10 400 $ suivi du livre paper frais à
    # 10 000 $) et le buy & hold se calculait sur cette fenêtre mélangée.
    _equity_epoch: int = 0
    _epoch_bot_id: int | None = None

    @classmethod
    def create(
        cls,
        n_scenarios: int = 100,
        asset: str = DEFAULT_ASSET,
        timeframe: str = DEFAULT_TIMEFRAME,
        seed: int = 42,
        exchange_mode: ExchangeMode = ExchangeMode.SIMULATION,
    ) -> SimulationSession:
        market = _load_market(asset, timeframe, exchange_mode)
        client: ExchangeClient | None = None
        if exchange_mode != ExchangeMode.SIMULATION:
            client = (
                market.client
                if isinstance(market, LiveMarketFeed)
                else ExchangeClient(mode=exchange_mode)
            )

        engine = TradingBot(simulator=market, n_scenarios=n_scenarios)
        # Reuse a previously trained model if present (a stale/corrupt file is
        # reported to the journal instead of silently looking "non entraîné").
        saved, saved_note = load_model_with_note(asset, timeframe)
        # Boot en TRAIN (replay + budget de risque, visiblement actif). L'ancien
        # auto-boot en LIVE quand un modèle existait déposait l'utilisateur sur
        # un mode qui « ne fait rien » (prédictions ~50/50, zéro trade) — la
        # pire première impression possible (audit UX). Le modèle reste chargé ;
        # ``m`` bascule en LIVE quand on le veut.
        run_mode = RunMode.TRAIN
        bot = engine.new_episode(
            episode=1,
            market_sim=market,
            candle_model=saved,
            predict_mode=False,
            use_model=False,
        )
        bot.scenario_engine.timeframe = timeframe
        bot.mode = run_mode.value
        if saved is not None:
            bot.log_msg(_model_loaded_line(saved))
        elif saved_note:
            bot.log_msg(saved_note)
        _attach_exchange(bot.portfolio, client)

        return cls(
            bot_engine=engine,
            bot=bot,
            asset=asset,
            timeframe=timeframe,
            data_source=market.source,
            exchange_mode=exchange_mode,
            exchange_client=client,
            run_mode=run_mode,
            return_span=_compute_return_span(getattr(market, "prices", None)),
        )

    # ------------------------------------------------------------------ #
    # Run mode (behaviour axis): train | paper | live
    # ------------------------------------------------------------------ #
    def _model_trained(self) -> bool:
        m = self.bot.candle_model
        return bool(m is not None and getattr(m, "trained", False))

    def _leave_paper(self) -> None:
        """If a transient paper bot is active, restore the stashed train/live bot.

        Paper runs on a *copy* of the bot (same risk-policy parameters, fresh
        portfolio), swapped into ``self.bot`` so the whole UI shows the paper
        run. Leaving paper restores the real train/live bot untouched.
        """
        if self.run_mode != RunMode.PAPER:
            return
        if self._home_bot is not None:
            self.bot = self._home_bot
            self._home_bot = None
        self.run_mode = RunMode.LIVE if self.bot.predict_mode else RunMode.TRAIN
        self._paper_started = False
        self._paper_done = False
        self._paper_forward = False
        self._paper_last_trades = 0
        self.paper_verdict = None

    def switch_timeframe(self, timeframe: str) -> str:
        self._leave_paper()
        risk = self.bot.risk
        # Ré-ancrage : le σ̂ lissé / P(turbulent) EWMA appartiennent à la marche
        # PRÉCÉDENTE (autre échelle de vol par timeframe) — les transporter
        # dimensionnait les premières décisions de la nouvelle série avec la
        # mémoire de l'ancienne (gotcha documenté CLAUDE.md, jamais appliqué).
        risk.reset_state()
        market = _load_market(self.asset, timeframe, self.exchange_mode)
        self.timeframe = timeframe
        self.data_source = market.source
        self.return_span = _compute_return_span(getattr(market, "prices", None))
        self.bot_engine.simulator = market
        saved, saved_note = load_model_with_note(self.asset, timeframe)
        self.bot = self.bot_engine.new_episode(
            episode=1,
            risk=risk,
            market_sim=market,
            candle_model=saved,
            predict_mode=False,
        )
        self.bot.scenario_engine.timeframe = timeframe
        # Rester en TRAIN après un switch (même logique que le boot) — le
        # modèle chargé est signalé, ``m`` bascule en LIVE explicitement.
        self.run_mode = RunMode.TRAIN
        self.bot.mode = self.run_mode.value
        if saved is not None:
            self.bot.log_msg(_model_loaded_line(saved))
        elif saved_note:
            self.bot.log_msg(saved_note)
        _attach_exchange(self.bot.portfolio, self.exchange_client)
        self.bot.log_msg(f"── Timeframe {timeframe} [{market.source}] ──")
        self._record_snapshot()
        return market.source

    def switch_asset(self, asset: str) -> str:
        self._leave_paper()
        risk = self.bot.risk
        risk.reset_state()  # nouvelle série de prix = nouvelle marche EWMA
        market = _load_market(asset, self.timeframe, self.exchange_mode)
        self.asset = asset
        self.data_source = market.source
        self.return_span = _compute_return_span(getattr(market, "prices", None))
        self.bot_engine.simulator = market
        saved, saved_note = load_model_with_note(asset, self.timeframe)
        self.bot = self.bot_engine.new_episode(
            episode=1,
            risk=risk,
            market_sim=market,
            candle_model=saved,
            predict_mode=False,
        )
        self.bot.scenario_engine.timeframe = self.timeframe
        self.run_mode = RunMode.TRAIN
        self.bot.mode = self.run_mode.value
        if saved is not None:
            self.bot.log_msg(_model_loaded_line(saved))
        elif saved_note:
            self.bot.log_msg(saved_note)
        _attach_exchange(self.bot.portfolio, self.exchange_client)
        self.bot.log_msg(f"── Actif chargé: {asset} [{market.source}] ──")
        self._record_snapshot()
        return market.source

    def set_exchange_mode(self, mode: ExchangeMode) -> str:
        self._leave_paper()
        risk = self.bot.risk
        # Charger la nouvelle source AVANT de muter l'état : si la connexion
        # échoue (ex. LIVE sans clés API), la session doit rester COHÉRENTE sur
        # son mode actuel — l'ancien ordre laissait exchange_mode=LIVE avec un
        # feed paper (badge 🔴 mensonger + cycle x désynchronisé).
        market = _load_market(self.asset, self.timeframe, mode)
        self.exchange_mode = mode
        risk.reset_state()  # nouvelle marche (les barres/indices changent de source)
        self.data_source = market.source
        self.return_span = _compute_return_span(getattr(market, "prices", None))
        self.bot_engine.simulator = market

        if mode == ExchangeMode.SIMULATION:
            self.exchange_client = None
        else:
            self.exchange_client = (
                market.client if isinstance(market, LiveMarketFeed) else ExchangeClient(mode=mode)
            )

        # Same (asset, timeframe) — only the data SOURCE changes; keep the model
        # and the current behaviour (run mode). Source no longer touches bot.mode.
        prev_model = self.bot.candle_model
        prev_predict = self.bot.predict_mode
        prev_use_model = self.bot.use_model
        self.bot = self.bot_engine.new_episode(
            episode=1,
            risk=risk,
            market_sim=market,
            candle_model=prev_model,
            predict_mode=prev_predict,
            use_model=prev_use_model,
        )
        _attach_exchange(self.bot.portfolio, self.exchange_client)
        self.bot.mode = self.run_mode.value

        label = {
            "simulation": "Historique (cache)",
            "paper": "Binance papier",
            "live": "Binance RÉEL",
        }[mode.value]
        self.bot.log_msg(f"── Source données: {label} [{market.source}] ──")
        self._record_snapshot()
        return market.source

    def _restart_episode(self) -> None:
        risk = self.bot.risk
        risk.reset_state()  # rewind = nouvelle marche, pas la suite de l'ancienne
        prev_model = self.bot.candle_model
        prev_predict = self.bot.predict_mode
        ep = self.bot.episode + 1
        self.bot = self.bot_engine.new_episode(
            episode=ep,
            risk=risk,
            market_sim=self.bot_engine.simulator,
            candle_model=prev_model,
            predict_mode=prev_predict,
        )
        _attach_exchange(self.bot.portfolio, self.exchange_client)
        self.bot.log_msg(
            f"── Épisode {ep} | {self.asset} [{self.data_source}] (rewind) ──"
        )

    def tick(self) -> bool:
        # Dispatch on the behaviour axis (run mode):
        #   LIVE  → predict the next candle, never trade (sits on the latest bar).
        #   PAPER → trade frozen (no learning), accrue a profitability verdict.
        #   TRAIN → replay history, trade and learn.
        if self.run_mode == RunMode.LIVE:
            return self._predict_tick()
        if self.run_mode == RunMode.PAPER:
            return self._paper_tick()

        sim = self.bot_engine.simulator
        if isinstance(sim, LiveMarketFeed):
            try:
                sim.maybe_refresh()
            except Exception as exc:  # panne réseau transitoire — on garde les barres connues
                self.bot.log_msg(f"[yellow]⚠ rafraîchissement live échoué: {exc}[/]")
            if self.bot.market.step >= sim.n_bars - 1:
                # Tête du feed : pas de nouvelle bougie clôturée → on ATTEND.
                # (L'ancien code re-traitait la même barre à chaque tick.)
                self._record_snapshot()
                return True

        # Sur un feed temps réel : avancer-puis-décider (exécution au close de
        # la bougie fraîchement clôturée, pas au close vieux d'une barre).
        stepper = (
            self.bot_engine.step_live_forward
            if isinstance(sim, LiveMarketFeed)
            else self.bot_engine.step
        )
        for _ in range(self.steps_per_tick):
            if not stepper(self.bot):
                if isinstance(sim, LiveMarketFeed):
                    break  # fin de données live = attendre, pas redémarrer
                self._restart_episode()
                if not self.bot_engine.step(self.bot):
                    return False
        self._record_snapshot()
        return True

    def _begin_live_session(self) -> None:
        """Start a fresh Live walk-forward: reset the scorecard and anchor the
        bot on the first candle to predict.

        * **Simulation / cached history** — start the walk near the end of the
          series (≈ last 20%, after the model's training split) so the bot
          predicts candles it mostly hasn't trained on, then chains forward bar
          by bar until the data runs out.
        * **Live / paper feed** — anchor on the latest closed bar and predict the
          candle currently forming; new bars settle previous predictions.
        """
        sim = self.bot_engine.simulator
        n = sim.n_bars
        if isinstance(sim, LiveMarketFeed):
            start = max(0, n - 1)
        else:
            # Start strictly AFTER the bars the model was trained on, so the
            # reliability note is measured on genuinely unseen candles. When the
            # model carries its training end (train_end_step), anchor just past
            # it; otherwise fall back to the out-of-sample tail (~last 20%).
            model = self.bot.candle_model
            train_end = int(getattr(model, "train_end_step", -1)) if model else -1
            if train_end >= 0:
                start = min(max(WARMUP_BARS, train_end + 1), n - 2)
            else:
                start = min(max(WARMUP_BARS, int(n * 0.8)), n - 2)
            start = max(0, start)
        self._eval_start = start
        self.bot.market = sim.state_at(start)
        self.bot.live_eval.reset()
        # La marche Live saute sur la queue OOS : la trace de risque affichée
        # doit se ré-ancrer sur CES barres, pas prolonger l'EWMA du replay.
        self.bot.risk.reset_state()
        # Score predictions with the same volatility-adaptive flat band the
        # forecast uses (prefer the trained model's stored threshold).
        model = self.bot.candle_model
        thr = float(getattr(model, "dir_threshold", 0.0) or 0.0) if model else 0.0
        if thr <= 0.0:
            thr = _compute_dir_threshold(getattr(sim, "prices", None))
        self.bot.live_eval.dir_threshold = thr
        self.bot.current_bundle = None
        # Pin the clean batch weights as the anchor and start each Live walk from
        # them: the online self-reinforcement adapts WITHIN this out-of-sample
        # session, never carrying drift across sessions — that keeps the
        # reliability note honest and the adaptation bounded.
        if model is not None and getattr(model, "trained", False):
            model.ensure_anchor()
            if self.online_learn:
                model.reset_online()
        extra = (
            " · 🔄 renforcement en ligne actif (suivi de régime, pas d'edge)"
            if self.online_learn else ""
        )
        self.bot.log_msg(
            f"▶ Session Live — départ bougie #{start}, le bot enchaîne les "
            f"prédictions et s'auto-évalue à chaque clôture.{extra}"
        )

    def _predict_tick(self) -> bool:
        """Live tick: chain predictions candle by candle and self-evaluate.

        Each tick (1) settles the pending prediction if its target candle has
        closed — scoring prediction vs reality — then (2) advances onto that
        closed candle and (3) predicts the next one. When the history is
        exhausted the session is finalized with a reliability grade.
        """
        bot = self.bot
        ev = bot.live_eval
        sim = self.bot_engine.simulator
        if not ev.started:
            self._begin_live_session()

        if isinstance(sim, LiveMarketFeed):
            try:
                sim.maybe_refresh()
            except Exception as exc:  # panne réseau transitoire — on garde les barres connues
                bot.log_msg(f"[yellow]⚠ rafraîchissement live échoué: {exc}[/]")
        last_closed = max(0, sim.n_bars - 1)

        # (1) Settle the pending prediction once its target candle has closed.
        if ev.pending is not None:
            tgt = ev.pending.target_step
            if tgt > last_closed:
                # Target candle not closed yet (live: still forming). Hold steady.
                self._record_snapshot()
                return True
            # Capture the predict-time features BEFORE settle() nulls the pending.
            pend_feats = ev.pending.features
            pend_model_driven = ev.pending.model_driven
            realized_price = float(sim.prices[tgt])
            settled = ev.settle(realized_price)
            if settled is not None:
                bot.log_msg(ev.settle_log_line(settled))
                # Self-reinforcement: the just-settled candle becomes one online
                # SGD step on the model that predicted it.
                self._reinforce_online(settled, pend_feats, pend_model_driven)
            bot.market = sim.state_at(tgt)  # walk forward onto the closed candle

        # (2) End of history (simulation walk-forward complete) → finalize.
        if bot.market.step >= sim.n_bars - 1 and not isinstance(sim, LiveMarketFeed):
            for line in ev.finalize():
                bot.log_msg(line)
            self._record_snapshot()
            return True

        # (3) Predict the next candle from the current (closed) bar. The scoring
        # band follows the CURRENT bar's adaptive threshold (same band the
        # forecast classifies with) — a session-frozen band diverged from the
        # per-bar band whenever the volatility drifted mid-session.
        ev.dir_threshold = _compute_dir_threshold(bot.market.history)
        bundle = self.bot_engine.predict_only(bot)
        fc = bundle.next_candle if bundle else None
        if fc is not None:
            pend = ev.open_from_forecast(fc)
            # Stash the exact standardised features the model predicted with, so
            # the online update (on settle) learns on precisely this input.
            model = bot.candle_model
            if self.online_learn and model is not None and getattr(model, "trained", False):
                pend.features = model.features_asof(bot.market.history)
            bot.log_msg(ev.pending_log_line())
        self._record_snapshot()
        return True

    def _reinforce_online(self, settled, features, model_driven: bool) -> None:
        """Self-reinforcement: fold a just-settled candle into the model as one
        online SGD step. Guards: Live online learning on, the forecast was
        model-driven, and we captured the predict-time features. In-memory only —
        never persisted (persisting online drift would resurrect the non-OOS
        look-ahead bug). Honest scope: regime tracking + calibration, not alpha.
        """
        model = self.bot.candle_model
        if not (self.online_learn and model_driven and features is not None):
            return
        if model is None or not getattr(model, "trained", False):
            return
        idx = {"up": 0, "flat": 1, "down": 2}.get(settled.realized_dir)
        if idx is None:
            return
        info = model.online_update(features, idx)
        if not info:
            return
        # Compact journal line every few updates — show the model adapting
        # without flooding the log.
        if model.n_online <= 1 or model.n_online % 10 == 0:
            self.bot.log_msg(
                f"[dim]🔄 renforcement en ligne #{model.n_online} · "
                f"lr={info['lr']:.4f} · NLL≈{model.online_recent_nll():.2f} · "
                f"dérive {model.online_drift():.2f}[/]"
            )

    def train_episode(self, steps: int = 400) -> dict:
        self._leave_paper()  # an episode replays on the real bot, not a paper copy
        self.bot = self.bot_engine.run_training_episode(self.bot, max_steps=steps)
        b = self.bot
        tf = getattr(b.scenario_engine, "timeframe", self.timeframe)
        stats = {
            "episode": b.episode - 1,
            "pnl": b.portfolio.total_pnl(b.market.price),
            "realized_vol": b.metrics.realized_vol_ann(tf),
            "target_vol": b.risk.sigma_target_ann,
            "avg_exposure": b.metrics.avg_exposure,
            "win_rate": b.portfolio.win_rate(),
            "trades": len(b.portfolio.trades),
            "fees": b.portfolio.fees_paid + b.portfolio.slippage_paid,
            "log": list(b.log[-8:]),
            "asset": self.asset,
        }
        risk = b.risk
        prev_model = b.candle_model
        prev_predict = b.predict_mode
        ep = b.episode
        # Walk-forward : l'épisode suivant reprend là où celui-ci s'est arrêté
        # (fenêtre NEUVE, PnL différent) — l'ancien redémarrage systématique à
        # WARMUP rendait chaque épisode identique au octet (audit UX).
        sim = self.bot_engine.simulator
        nxt_start = b.market.step
        if nxt_start >= sim.n_bars - 60:
            nxt_start = WARMUP_BARS  # historique épuisé → on boucle
            risk.reset_state()  # saut discontinu → l'EWMA de l'ancienne marche ne vaut plus
        self.bot = self.bot_engine.new_episode(
            episode=ep,
            risk=risk,
            market_sim=sim,
            candle_model=prev_model,
            predict_mode=prev_predict,
            use_model=b.use_model,
            start_step=nxt_start,
        )
        _attach_exchange(self.bot.portfolio, self.exchange_client)
        self.bot.mode = self.run_mode.value
        self._record_snapshot()
        return stats

    # ------------------------------------------------------------------ #
    # Supervised next-candle model: gradient-descent training + live mode
    # ------------------------------------------------------------------ #
    def train_model(self, epochs: int = 400, lr: float = 0.5, progress=None) -> TrainReport:
        """Train the next-candle model by gradient descent on the loaded history.

        Builds a look-ahead-free dataset over the current asset/timeframe, runs
        the epochs, persists the weights, keeps the loss/accuracy curve in
        ``self.train_report`` and switches the bot to **Live** so the learned
        predictions immediately drive the bubbles.
        """
        self._leave_paper()  # train the real bot, never a transient paper copy
        prices = getattr(self.bot_engine.simulator, "prices", None)
        model, report = train_candle_model(
            prices,
            symbol=self.asset,
            timeframe=self.timeframe,
            epochs=epochs,
            lr=lr,
            progress=progress,
        )
        self.train_report = report
        if model.trained:
            model.save()
            self.bot.candle_model = model
            self.bot.predict_mode = True
            self.bot.use_model = True
            self.run_mode = RunMode.LIVE
            self.bot.mode = RunMode.LIVE.value
            self.bot.log_msg(
                f"🧠 Modèle entraîné — {report.n_train} barres · {epochs} epochs · "
                f"perte {report.loss_history[0]:.3f}→{report.final_loss:.3f} (↓ apprend) · "
                f"val {report.val_accuracy:.0%} vs classe maj. {report.val_majority:.0%}"
            )
            self.bot.log_msg("▶ Mode LIVE — le bot estime la prochaine bougie (plus de trades)")
            self.bot.live_eval.started = False  # force a fresh walk-forward session
            self._predict_tick()  # immediate fresh estimate on the latest bar
        else:
            self.bot.log_msg("⚠ Historique insuffisant pour entraîner le modèle")
            self._record_snapshot()
        return report

    # ------------------------------------------------------------------ #
    # Paper mode: frozen profitability evaluation (backtest / forward)
    # ------------------------------------------------------------------ #
    def set_run_mode(self, mode: RunMode) -> RunMode:
        """Switch the behaviour axis: train | paper | live.

        TRAIN/LIVE run on the persistent bot (LIVE = predict-only). PAPER swaps in
        a transient copy with a fresh portfolio so a profitability backtest never
        perturbs the home bot's book, then restores it on exit.
        """
        if mode == self.run_mode:
            return self.run_mode

        if self.run_mode == RunMode.PAPER:
            self._leave_paper()  # restore the stashed train/live bot first

        if mode == RunMode.PAPER:
            self._home_bot = self.bot
            self.bot = self._make_paper_bot()
            self.run_mode = RunMode.PAPER
            self.bot.mode = RunMode.PAPER.value
            self._paper_started = False
            self._begin_paper_session()
            return self.run_mode

        if mode == RunMode.LIVE:
            self.set_predict_mode(True)  # sets run_mode LIVE (or TRAIN if no model)
            return self.run_mode

        self.set_predict_mode(False)  # → TRAIN
        return self.run_mode

    def cycle_run_mode(self) -> RunMode:
        """train → paper → live → train. Skips Live when no model is trained."""
        nxt = self.run_mode.next()
        skipped_live = nxt == RunMode.LIVE and not self._model_trained()
        if skipped_live:
            nxt = RunMode.TRAIN
        mode = self.set_run_mode(nxt)
        if skipped_live:
            # Logger APRÈS set_run_mode : en quittant PAPER, self.bot est le bot
            # restauré — l'ancien ordre écrivait l'explication dans le journal
            # du bot paper transitoire, jeté juste après (jamais affichée).
            self.bot.log_msg(
                "⚠ Pas de modèle entraîné — Live ignoré (touche G pour entraîner). "
                "Retour à Entraînement."
            )
        return mode

    def _make_paper_bot(self) -> BotState:
        """A copy of the bot for a paper run: same risk-policy parameters and
        candle model, fresh portfolio — so the verdict reflects the CURRENT
        configuration without polluting the home bot's book. The risk policy is
        deterministic (no learned state), so a parameter copy IS a frozen copy.
        """
        home = self.bot
        sim = self.bot_engine.simulator
        frozen = dataclasses.replace(home.risk, last_trace=None)
        # dataclasses.replace copie AUSSI l'état EWMA (_sigma_state/_pturb_state
        # sont des champs init) : le backtest paper démarrait sur la queue OOS
        # avec le σ̂ lissé capturé ~1000 barres plus tôt sur la marche TRAIN —
        # verdict dépendant de l'activité antérieure. Copie de PARAMÈTRES
        # seulement : l'état se ré-ancre sur les premières barres du backtest.
        frozen.reset_state()
        # Même exigence pour le modèle de bougie : la dérive du renforcement en
        # ligne acquise en LIVE (sur les MÊMES bougies OOS que le backtest
        # rejoue) contaminait le verdict « gelé ». Le paper trade sur la copie
        # aux poids d'ANCRE (batch validés) ; le bot maison garde sa dérive.
        paper_model = _anchored_model_copy(home.candle_model)
        pb = self.bot_engine.new_episode(
            episode=home.episode,
            risk=frozen,
            market_sim=sim,
            candle_model=paper_model,
            predict_mode=False,  # paper TRADES — it is not predict-only
            use_model=self._model_trained(),
        )
        pb.scenario_engine.timeframe = self.timeframe
        _attach_exchange(pb.portfolio, self.exchange_client)
        pb.mode = RunMode.PAPER.value
        return pb

    def _begin_paper_session(self) -> None:
        """Anchor a fresh paper run and reset its scorecard.

        * cached history → start in the out-of-sample tail (past the model's
          training split), run to the end, then emit a final verdict.
        * live feed → anchor on the latest bar and accrue PnL going forward.
        """
        sim = self.bot_engine.simulator
        forward = isinstance(sim, LiveMarketFeed)
        if forward and hasattr(sim, "force_refresh"):
            try:
                sim.force_refresh()
            except Exception as exc:  # network hiccup — keep what we have
                self.bot.log_msg(f"[yellow]⚠ rafraîchissement live échoué: {exc}[/]")
        n = sim.n_bars
        if forward:
            start = max(0, n - 1)
        else:
            model = self.bot.candle_model
            train_end = int(getattr(model, "train_end_step", -1)) if model else -1
            if train_end >= 0:
                start = min(max(WARMUP_BARS, train_end + 1), n - 2)
            else:
                start = min(max(WARMUP_BARS, int(n * 0.8)), n - 2)
            start = max(0, start)
        self._paper_forward = forward
        self._paper_start = start
        self._paper_started = True
        self._paper_done = False
        self._paper_last_trades = 0
        self.paper_verdict = None
        self.bot.market = sim.state_at(start)
        self.bot.risk.reset_state()  # ré-ancrage : l'EWMA démarre sur les barres du backtest
        fee_bps = self.bot.portfolio.fee_rate * 1e4
        if forward:
            self.bot.log_msg(
                f"▶ Mode PAPER (paper-trading live) — argent fictif, politique de risque "
                f"déterministe, frais {fee_bps:.0f} bps + slippage · départ bougie #{start} · "
                f"le PnL s'accumule à chaque clôture (rentable ou non, en marche avant)."
            )
        else:
            self.bot.log_msg(
                f"▶ Mode PAPER (backtest hors-échantillon) — politique de risque déterministe, "
                f"frais {fee_bps:.0f} bps + slippage · départ bougie #{start} → verdict "
                f"« rentable ou non » (avec test de significativité) à la fin de l'historique."
            )

    def _paper_tick(self) -> bool:
        """Paper tick: trade the frozen bot, never learn, accrue the verdict."""
        bot = self.bot
        sim = self.bot_engine.simulator
        if not self._paper_started:
            self._begin_paper_session()
        if isinstance(sim, LiveMarketFeed):
            try:
                sim.maybe_refresh()
            except Exception as exc:  # panne réseau transitoire — on garde les barres connues
                bot.log_msg(f"[yellow]⚠ rafraîchissement live échoué: {exc}[/]")
            if bot.market.step >= sim.n_bars - 1:
                # Tête du feed (paper forward) : pas de nouvelle bougie → on
                # attend au lieu de re-trader la même barre à chaque tick.
                self._record_snapshot()
                return True
        if self._paper_done:
            self._record_snapshot()  # backtest finished — hold on the verdict
            return True

        prices = getattr(sim, "prices", None)
        start_price = (
            float(prices[self._paper_start])
            if prices is not None and 0 <= self._paper_start < len(prices)
            else bot.market.price
        )
        forward = self._paper_forward
        if forward:
            # Marche avant temps réel : la décision est déclenchée par la
            # clôture d'une bougie → avancer dessus et exécuter à SON close
            # (l'ancien ordre remplissait au close de la barre précédente et
            # encaissait rétroactivement le mouvement de la bougie close).
            alive = self.bot_engine.step_live_forward(bot)
        else:
            alive = self.bot_engine.step(bot, learn=False)  # frozen backtest
        final = (not forward) and ((not alive) or bot.market.step >= sim.n_bars - 1)

        if final and bot.portfolio.position > 0:
            # Close the open position for a clean, fully-realised verdict.
            bot.portfolio.execute(
                Action.SELL, bot.market.price, bot.market.step, reason="fin backtest"
            )
        bars = max(1, bot.market.step - self._paper_start)
        self.paper_verdict = build_verdict(
            bot.portfolio,
            bot.market.price,
            start_price=start_price,
            bars=bars,
            forward=forward,
            final=final,
            equity_curve=bot.metrics.equity_curve,
        )
        if final and not self._paper_done:
            self._paper_done = True
            for line in verdict_lines(self.paper_verdict):
                bot.log_msg(line)
        elif forward:
            # Log a running verdict whenever a new trade lands (no flooding).
            n_tr = len(bot.portfolio.trades)
            if n_tr != self._paper_last_trades:
                self._paper_last_trades = n_tr
                for line in verdict_lines(self.paper_verdict):
                    bot.log_msg(line)
        self._record_snapshot()
        return True

    def set_predict_mode(self, on: bool) -> bool:
        """Switch Live (prédiction du modèle) ↔ Entraînement (replay + apprentissage)."""
        model = self.bot.candle_model
        if on and (model is None or not getattr(model, "trained", False)):
            self.bot.log_msg("⚠ Aucun modèle entraîné — lance d'abord l'entraînement (touche G)")
            self.bot.predict_mode = False
            self.bot.use_model = False
            self.run_mode = RunMode.TRAIN
            self.bot.mode = RunMode.TRAIN.value
            return False
        self.bot.predict_mode = on
        self.bot.use_model = bool(on)  # Live drives the model; Train doesn't
        if on:
            self.run_mode = RunMode.LIVE
            self.bot.mode = RunMode.LIVE.value
            self.bot.live_eval.started = False  # force a fresh walk-forward session
            sim = self.bot_engine.simulator
            if isinstance(sim, LiveMarketFeed):
                # Pull the freshest bars NOW so we estimate the candle currently
                # forming on live data, not one up to a refresh-interval stale.
                try:
                    sim.force_refresh()
                except Exception as exc:  # network hiccup — keep what we have
                    self.bot.log_msg(f"[yellow]⚠ rafraîchissement live échoué: {exc}[/]")
                self.bot.log_msg(
                    f"▶ Mode LIVE — 🔴 données Binance temps réel ({sim.n_bars} barres, "
                    f"dernière clôture chargée) · le bot estime la bougie en cours "
                    f"et s'évalue à chaque clôture."
                )
            else:
                self.bot.log_msg(
                    "▶ Mode prédiction — ⏪ [yellow]données HISTORIQUES (replay), "
                    "pas le marché live[/]. Pour des prédictions sur le marché en "
                    "direct, choisis l'exchange « Paper Binance » ou « LIVE »."
                )
            self._predict_tick()
        else:
            # Leaving Live: close the books and report the reliability grade.
            for line in self.bot.live_eval.finalize():
                self.bot.log_msg(line)
            self.run_mode = RunMode.TRAIN
            self.bot.mode = RunMode.TRAIN.value
            self.bot.log_msg("■ Mode Entraînement — le bot rejoue l'historique et apprend")
        return on

    def toggle_predict_mode(self) -> bool:
        return self.set_predict_mode(not self.bot.predict_mode)

    def set_online_learn(self, on: bool) -> bool:
        """Enable/disable Live self-reinforcement (online SGD on settled candles)."""
        self.online_learn = bool(on)
        model = self.bot.candle_model
        if on:
            if model is not None and getattr(model, "trained", False):
                model.ensure_anchor()
            self.bot.log_msg(
                "🔄 Renforcement en ligne ACTIVÉ — chaque bougie clôturée ajuste "
                "le modèle (suivi de régime + calibration, pas d'edge directionnel)."
            )
        else:
            # Stop adapting and snap back to the validated batch weights.
            if model is not None:
                model.reset_online()
            self.bot.log_msg(
                "⏸ Renforcement en ligne COUPÉ — retour aux poids entraînés (figés)."
            )
        return self.online_learn

    def toggle_online_learn(self) -> bool:
        return self.set_online_learn(not self.online_learn)

    def model_status(self) -> dict:
        """Learned-model snapshot for the UI (status strip, metrics, badges)."""
        m = self.bot.candle_model
        trained = bool(m is not None and getattr(m, "trained", False))
        return {
            "trained": trained,
            "run_mode": self.run_mode.value,
            "predict_mode": self.bot.predict_mode,
            "val_accuracy": float(getattr(m, "val_accuracy", 0.0)) if trained else 0.0,
            "n_samples": int(getattr(m, "n_samples", 0)) if trained else 0,
            "epochs": int(getattr(m, "epochs_trained", 0)) if trained else 0,
            "online_learn": self.online_learn,
            "n_online": int(getattr(m, "n_online", 0)) if trained else 0,
            "online_drift": float(m.online_drift()) if trained else 0.0,
            "online_nll": float(m.online_recent_nll()) if trained else 0.0,
        }

    def _record_snapshot(self) -> None:
        b = self.bot
        if id(b) != self._epoch_bot_id:
            self._epoch_bot_id = id(b)
            self._equity_epoch += 1
        trace = b.risk.last_trace
        tf = getattr(b.scenario_engine, "timeframe", self.timeframe)
        # Dédoublonnage : les ticks « d'attente » en tête de feed live (aucune
        # nouvelle bougie) ne doivent pas inonder l'historique — à 0,1 s/tick le
        # cap de 500 instantanés ne couvrait sinon que ~50 s d'equity.
        if self.history:
            last = self.history[-1]
            if last.get("step") == b.market.step and last.get("asset") == self.asset:
                self.history[-1] = {**last, "price": b.market.price,
                                    "equity": b.portfolio.mark_to_market(b.market.price)}
                return
        snap = {
            "step": b.market.step,
            "price": b.market.price,
            "equity": b.portfolio.mark_to_market(b.market.price),
            "pnl": b.portfolio.total_pnl(b.market.price),
            "exposure": b.portfolio.exposure(b.market.price),
            "target_exposure": trace.target_exposure if trace else 0.0,
            "sigma_ann": trace.sigma_ann if trace else 0.0,
            "realized_vol": b.metrics.realized_vol_ann(tf),
            "asset": self.asset,
            "exchange_mode": self.exchange_mode.value,
            "epoch": self._equity_epoch,
        }
        self.history.append(snap)
        if len(self.history) > 500:
            self.history = self.history[-500:]