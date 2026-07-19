"""Autonomous trading bot — scenario generation + risk-budget decisions.

Refonte : l'ancienne boucle « scénario sélectionné → logits d'action → récompense
sur le rendement 1 barre » apprenait du bruit par construction (la direction
1-barre est ≈50/50, mesuré — ``ANALYSE_CRITIQUE_MODELE.md`` §1-2) et, sous le
forecast martingale honnête, ne tradait presque jamais. Elle est remplacée par la
**politique de budget de risque** (``core/decision.py``) : à chaque barre le bot
calcule une exposition cible à partir des têtes *validées* (σ̂ rough-vol, régime
HMM, porte d'edge Wilson) et rebalance — net de frais et de slippage — seulement
quand l'écart dépasse la bande de coûts.

Le « comportement » par mode ne change pas : TRAIN rejoue l'historique (et sert à
entraîner le modèle de bougie), PAPER évalue la rentabilité gelée hors
échantillon, LIVE prédit la prochaine bougie sans trader (``predict_only``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from config.market_config import WARMUP_BARS
from core.decision import DecisionTrace, RiskPolicy, bars_per_year
from core.live_eval import LiveEvaluator
from core.market import MarketSimulator, MarketState
from core.portfolio import Portfolio
from core.scenarios import ScenarioBundle, ScenarioEngine
from ml.candle_model import CandleModel


@dataclass
class RunMetrics:
    """Compteurs honnêtes de la session (remplace l'« autonomie » cosmétique).

    Le succès du bot se mesure sur des cibles atteignables : la vol réalisée de
    l'equity doit suivre la cible (tracking), le turnover doit rester borné, les
    coûts sont comptés. Aucun score composite flatteur.
    """

    total_steps: int = 0
    n_rebalances: int = 0
    exposure_sum: float = 0.0        # pour l'exposition moyenne
    equity_curve: list[float] = field(default_factory=list, repr=False)

    def record(self, equity: float, exposure: float, rebalanced: bool) -> None:
        self.total_steps += 1
        self.exposure_sum += exposure
        if rebalanced:
            self.n_rebalances += 1
        self.equity_curve.append(float(equity))
        if len(self.equity_curve) > 600:
            self.equity_curve = self.equity_curve[-600:]

    @property
    def avg_exposure(self) -> float:
        return self.exposure_sum / self.total_steps if self.total_steps else 0.0

    @property
    def turnover(self) -> float:
        return self.n_rebalances / self.total_steps if self.total_steps else 0.0

    def realized_vol_ann(self, timeframe: str, window: int = 60) -> float:
        """Vol annualisée réalisée de l'equity (fenêtre glissante) — à comparer
        à la cible σ_target : LA métrique du job du bot."""
        eq = np.asarray(self.equity_curve[-(window + 1):], dtype=np.float64)
        if len(eq) < 8 or np.any(eq <= 0):
            return 0.0
        r = np.diff(np.log(eq))
        return float(np.std(r) * math.sqrt(bars_per_year(timeframe)))


@dataclass
class BotState:
    market: MarketState
    market_sim: MarketSimulator
    portfolio: Portfolio
    risk: RiskPolicy
    scenario_engine: ScenarioEngine
    current_bundle: ScenarioBundle | None = None
    episode: int = 1
    running: bool = False
    paused: bool = False
    mode: str = "train"  # run-mode label: train | paper | live (see core.run_mode)
    candle_model: CandleModel | None = None  # learned next-candle model
    predict_mode: bool = False  # True = Live: predict-only (no trading), self-evaluate
    use_model: bool = False  # feed the learned candle model into scenario generation
    live_eval: LiveEvaluator = field(default_factory=LiveEvaluator)  # Live self-scoring
    metrics: RunMetrics = field(default_factory=RunMetrics)
    log: list[str] = field(default_factory=list)

    def log_msg(self, msg: str) -> None:
        self.log.append(msg)
        if len(self.log) > 200:
            self.log = self.log[-200:]

    @property
    def last_trace(self) -> DecisionTrace | None:
        return self.risk.last_trace


@dataclass
class TradingBot:
    simulator: MarketSimulator
    n_scenarios: int = 100
    horizon: int = 12
    initial_cash: float = 10_000.0

    def new_episode(
        self,
        episode: int = 1,
        seed_offset: int = 0,
        risk: RiskPolicy | None = None,
        market_sim: MarketSimulator | None = None,
        candle_model: CandleModel | None = None,
        predict_mode: bool = False,
        use_model: bool | None = None,
        start_step: int | None = None,
    ) -> BotState:
        """``start_step`` ancre l'épisode sur une barre précise — les épisodes
        successifs deviennent des SEGMENTS walk-forward (fenêtres différentes,
        PnL différents) au lieu de rejouer la même fenêtre à l'identique."""
        sim = market_sim or self.simulator
        start = min(WARMUP_BARS if start_step is None else max(start_step, WARMUP_BARS),
                    sim.n_bars - 2)
        state = sim.reset_step(start)
        # By default ``use_model`` follows ``predict_mode`` (the old behaviour:
        # Live drove the model, Train didn't). Paper mode passes it explicitly so
        # it can trade WITH the model while predict_mode stays False.
        um = predict_mode if use_model is None else use_model
        return BotState(
            market=state,
            market_sim=sim,
            portfolio=Portfolio(initial_cash=self.initial_cash, cash=self.initial_cash),
            risk=risk or RiskPolicy(),
            scenario_engine=ScenarioEngine(
                n_scenarios=self.n_scenarios,
                horizon=self.horizon,
                seed=episode,
                timeframe=getattr(sim, "timeframe", "1h"),
            ),
            episode=episode,
            candle_model=candle_model,
            predict_mode=predict_mode,
            use_model=um,
        )

    def _decide_and_trade(self, bot: BotState, bundle: ScenarioBundle) -> DecisionTrace:
        """Décision de risque sur la barre courante + rebalancement éventuel."""
        market = bot.market
        price = market.price
        model = bundle.model
        sigma_bar = float(model.estimated_vol) if model else max(market.ewma_volatility(), 1e-4)
        vol_regime = getattr(model, "vol_regime", "normal") if model else "normal"
        regime_probs = getattr(model, "vol_regime_probs", None) if model else None
        p_turb = regime_probs.get("turbulent") if regime_probs else None

        # Porte directionnelle : uniquement un edge Wilson SIGNIFICATIF, mesuré
        # sur des bougies réellement hors échantillon (live_eval). Par défaut la
        # porte est fermée — l'abstention directionnelle est structurelle.
        fc = bundle.next_candle
        edge_ok = bool(getattr(bot.live_eval, "significant_edge", False))
        p_up = float(fc.prob_up) if fc is not None and fc.model_driven else None
        p_down = float(fc.prob_down) if fc is not None and fc.model_driven else None

        trace = bot.risk.target_exposure(
            sigma_bar=sigma_bar,
            timeframe=bot.scenario_engine.timeframe,
            vol_regime=vol_regime,
            p_turbulent=p_turb,
            prob_up=p_up,
            prob_down=p_down,
            edge_significant=edge_ok,
        )
        trace = bot.risk.decide(
            trace,
            bot.portfolio.exposure(price),
            cost_per_side=bot.portfolio.cost_per_side(sigma_bar),
        )
        if trace.rebalanced:
            trade = bot.portfolio.rebalance(
                trace.exec_exposure,
                price,
                market.step,
                sigma_bar=sigma_bar,
                reason=trace.reason,
                symbol=market.symbol,
            )
            if trade is not None:
                bot.log_msg(
                    f"[{trade.action.value}] @{trade.price:,.0f} → exposition "
                    f"{trade.exposure_after:.0%} · {trace.reason}"
                )
            else:
                trace.rebalanced = False  # écart sous le minimum exécutable
        return trace

    def step(self, bot: BotState, learn: bool = True) -> bool:
        """Advance one bar: generate scenarios, decide exposure, rebalance.

        ``learn`` est conservé pour compatibilité d'API mais la politique de
        risque est déterministe — il n'y a plus rien à « apprendre » sur le
        rendement 1 barre (l'apprentissage réel du projet vit dans le modèle de
        bougie, touche ``g``, et son renforcement en ligne en mode Live).
        Returns False when the data is exhausted.
        """
        if bot.paused:
            return True

        market = bot.market
        bundle = bot.scenario_engine.generate(
            market,
            candle_model=bot.candle_model,
            use_model=bot.use_model,
        )
        bot.current_bundle = bundle
        bundle.selected_id = bundle.most_probable_id

        forecast = bundle.next_candle
        if forecast:
            bot.log_msg(forecast.summary_line())
            bot.log_msg(forecast.detail_line())

        trace = self._decide_and_trade(bot, bundle)

        # Advance market
        next_state = bot.market_sim.advance(market)
        if next_state is None:
            bot.running = False
            return False
        bot.market = next_state

        bot.metrics.record(
            equity=bot.portfolio.mark_to_market(bot.market.price),
            exposure=bot.portfolio.exposure(bot.market.price),
            rebalanced=trace.rebalanced,
        )
        return True

    def step_live_forward(self, bot: BotState) -> bool:
        """Pas « marche avant » pour un feed temps réel : avancer D'ABORD sur la
        bougie fraîchement clôturée, puis décider et exécuter à SON close — un
        prix qui existe *maintenant*.

        L'ordre replay (décider à t, exécuter au close de t, avancer) est la
        sémantique backtest standard, mais en tête de feed la décision n'est
        déclenchée que par la clôture de t+1 : exécuter au close de t (vieux
        d'une barre entière) encaissait rétroactivement le mouvement de la
        bougie qui venait de clôturer — un prix inatteignable. Pas de fuite
        d'information dans les deux cas ; c'est le prix d'exécution qui change.
        Returns False à la tête du feed (pas de nouvelle bougie → attendre).
        """
        if bot.paused:
            return True
        next_state = bot.market_sim.advance(bot.market)
        if next_state is None:
            return False
        bot.market = next_state
        bundle = bot.scenario_engine.generate(
            bot.market,
            candle_model=bot.candle_model,
            use_model=bot.use_model,
        )
        bot.current_bundle = bundle
        bundle.selected_id = bundle.most_probable_id
        forecast = bundle.next_candle
        if forecast:
            bot.log_msg(forecast.summary_line())
            bot.log_msg(forecast.detail_line())
        trace = self._decide_and_trade(bot, bundle)
        bot.metrics.record(
            equity=bot.portfolio.mark_to_market(bot.market.price),
            exposure=bot.portfolio.exposure(bot.market.price),
            rebalanced=trace.rebalanced,
        )
        return True

    def predict_only(self, bot: BotState) -> ScenarioBundle:
        """Live mode: estimate the next candle for the *current* bar without
        advancing, trading or learning. The bot's job here is purely to
        predict — no orders are placed, the market is not stepped.
        """
        market = bot.market
        bundle = bot.scenario_engine.generate(
            market,
            candle_model=bot.candle_model,
            use_model=bot.use_model,
        )
        bundle.selected_id = bundle.most_probable_id  # highlight the estimate, no trade
        bot.current_bundle = bundle
        # Trace de risque affichée aussi en Live (sans trade) : l'UI montre en
        # continu l'exposition que le bot *prendrait* — le raisonnement reste
        # visible même quand il n'exécute pas.
        model = bundle.model
        sigma_bar = float(model.estimated_vol) if model else max(market.ewma_volatility(), 1e-4)
        regime_probs = getattr(model, "vol_regime_probs", None) if model else None
        fc = bundle.next_candle
        trace = bot.risk.target_exposure(
            sigma_bar=sigma_bar,
            timeframe=bot.scenario_engine.timeframe,
            vol_regime=getattr(model, "vol_regime", "normal") if model else "normal",
            p_turbulent=regime_probs.get("turbulent") if regime_probs else None,
            prob_up=float(fc.prob_up) if fc is not None and fc.model_driven else None,
            prob_down=float(fc.prob_down) if fc is not None and fc.model_driven else None,
            edge_significant=bool(getattr(bot.live_eval, "significant_edge", False)),
        )
        trace = bot.risk.decide(trace, bot.portfolio.exposure(market.price),
                                cost_per_side=bot.portfolio.cost_per_side(sigma_bar))
        # Live = prédiction seule : AUCUNE jambe n'est exécutée. La trace doit
        # le dire — l'ancien panneau affichait « ⚖ rebalance 71%→95% » (une
        # action jamais faite) en contradiction avec « aucun trade » à côté.
        if trace.rebalanced:
            trace.rebalanced = False
            trace.reason = f"Live (aucun trade) — prendrait : {trace.reason}"
        else:
            trace.reason = f"Live (aucun trade) — {trace.reason}"
        return bundle

    def run_training_episode(self, bot: BotState, max_steps: int = 500) -> BotState:
        bot.running = True
        bot.mode = "train"
        steps = 0
        while steps < max_steps and self.step(bot):
            steps += 1

        bot.running = False
        bot.episode += 1
        pnl = bot.portfolio.total_pnl(bot.market.price)
        rv = bot.metrics.realized_vol_ann(bot.scenario_engine.timeframe)
        bot.log_msg(
            f"═══ Épisode {bot.episode - 1} terminé | PnL: {pnl:+,.2f} "
            f"| vol réalisée {rv:.0%} (cible {bot.risk.sigma_target_ann:.0%}) "
            f"| exposition moy. {bot.metrics.avg_exposure:.0%} "
            f"| {bot.metrics.n_rebalances} rebalancements ═══"
        )
        return bot
