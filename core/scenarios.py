"""Vectorized Monte Carlo scenario engine (rough-vol Student-t cone).

Refonte : l'échantillonnage de *chemins* de Markov (direction + régime) a été
retiré du chemin chaud. Mesuré (§8.4), il ne façonnait plus les trajectoires —
le cône est une martingale dimensionnée par la tête rough-vol validée — et ne
servait qu'à teinter la softmax de sélection d'un terme décoratif, au prix de
2 chaînes échantillonnées n×h à chaque tick. La matrice de transition de
direction reste estimée (``MarkovChainModel.fit``) : c'est l'entrée du forecast
statistique de la prochaine bougie (``core/next_candle.py``) et une statistique
descriptive honnête (fréquences conditionnelles 1 barre).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.market import MarketState
from core.markov_model import MarkovChainModel, MarkovSnapshot
from core.cone import ConeModel, ConeParams
from core.next_candle import NextCandleForecast, NextCandlePredictor
from core.prob_models import ProbModelSnapshot
from core.thresholds import adaptive_threshold
from ml.vol_model import SIGMA_FLOOR

_PATH_STORE_TOP_K = 50
_DIR_THRESHOLD = 0.003  # legacy fallback only

# Shrinkage applied to the *directional* drift of the Monte-Carlo cone.
# Measured on real data (tests/measure_model.py::_scenario_engine_eval,
# walk-forward, no leak): any non-zero drift made BOTH the terminal-return
# density (Gaussian NLL, same σ) AND the directional call (Brier) STRICTLY WORSE
# than a zero-drift martingale — BTC NLL −57%, ETH −34%; sign-acc 0.46/0.28 ≤
# coin-flip. This is the §1-2 result (1-bar direction ≈ 50/50) surfacing in the
# scenario engine: a drifted cone is a directional bet the data does not
# support. The cone is therefore centred at the current price (martingale) and
# sized by the *validated* volatility head (§8.3). Kept as a tunable knob so the
# choice remains measurable (frozen by tests/test_scenario_drift.py).
_DRIFT_SHRINK = 0.0

# Report the *forecast* direction masses (prob_up/prob_down/expected_return and
# direction_probs) from the **equal-weight** Monte-Carlo path frequencies — the
# raw generative predictive density — instead of the softmax-weighted masses.
# Measured (§8.4): the per-scenario softmax weight carries the drawdown penalty
# (-0.3*max_dd^2/sigma^2), which mechanically under-weights down-paths (larger
# drawdowns) and tilts prob_up/prob_down upward -> weighted dir-Brier (0.67) >
# climatology (0.52) > equal-weight (0.50). The drawdown penalty is a legitimate
# *risk preference for scenario display* (it still shapes ``probability``,
# ``confidence`` and the most-probable pick), but it must not masquerade as a
# directional *forecast*. So: forecast = equal-weight (honest density),
# risk-pick = softmax-weighted (drawdown-aware).
_FORECAST_EQUAL_WEIGHT = True


@dataclass
class Scenario:
    id: int
    path: np.ndarray
    terminal_return: float
    max_drawdown: float
    probability: float
    direction: str
    markov_log_prob: float = 0.0  # legacy field (plus d'échantillonnage Markov)
    markov_path: str = ""         # legacy field

@dataclass
class ScenarioBundle:
    scenarios: list[Scenario]
    horizon: int
    current_price: float
    expected_return: float
    prob_up: float
    prob_down: float
    confidence: float
    model: ProbModelSnapshot | None = None
    markov: MarkovSnapshot | None = None
    selected_id: int | None = None
    most_probable_id: int | None = None
    next_candle: NextCandleForecast | None = None
    timeframe: str = "1h"

    @property
    def best_scenario(self) -> Scenario | None:
        if self.selected_id is None:
            return None
        return next((s for s in self.scenarios if s.id == self.selected_id), None)

    @property
    def most_probable(self) -> Scenario | None:
        if self.most_probable_id is None:
            return None
        return next((s for s in self.scenarios if s.id == self.most_probable_id), None)

    def top_n(self, n: int = 5) -> list[Scenario]:
        return sorted(self.scenarios, key=lambda s: s.probability, reverse=True)[:n]

    def direction_counts(self) -> dict[str, int]:
        counts = {"up": 0, "down": 0, "flat": 0}
        for s in self.scenarios:
            counts[s.direction] += 1
        return counts

    def direction_probs(self) -> dict[str, float]:
        """Directional *forecast* consensus. Equal-weight (the honest generative
        density) when _FORECAST_EQUAL_WEIGHT, else the softmax-weighted masses —
        the same forecast/selection split as prob_up/prob_down (§8.4)."""
        masses = {"up": 0.0, "down": 0.0, "flat": 0.0}
        if _FORECAST_EQUAL_WEIGHT:
            n = max(len(self.scenarios), 1)
            for s in self.scenarios:
                masses[s.direction] += 1.0 / n
        else:
            for s in self.scenarios:
                masses[s.direction] += s.probability
        return masses

    def terminal_quantiles(self, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[float, float]:
        """Quantiles équipondérés du rendement terminal — la lecture honnête du
        cône (densité générative), pour l'affichage des bandes."""
        rets = np.array([s.terminal_return for s in self.scenarios])
        if len(rets) == 0:
            return {q: 0.0 for q in qs}
        return {float(q): float(np.quantile(rets, q)) for q in qs}

    def verdict(self) -> str:
        # Lead with the honest directional FORECAST = the equal-weight density
        # (direction_probs). The "most probable" scenario is a RISK-weighted
        # display pick (softmax of likelihood − drawdown penalty), NOT a
        # direction forecast — presenting its UP/DOWN as the verdict would
        # re-introduce, in this widget, the upward tilt §8.4 removed from the
        # numbers. So we keep the two visibly distinct.
        masses = self.direction_probs()
        dominant = max(masses, key=masses.get)  # type: ignore[arg-type]
        dom_p = masses[dominant]
        dir_fr = {"up": "HAUSSE", "down": "BAISSE", "flat": "NEUTRE"}[dominant]
        head = f"Forecast directionnel {dir_fr} {dom_p:.0%} (densité équipondérée)"
        mp = self.most_probable
        if mp is None:
            return head
        return (
            f"{head} | scénario d'action #{mp.id} "
            f"(pondéré vraisemblance × risque, DD {mp.max_drawdown:.1%})"
        )


def _classify_returns(returns: np.ndarray, threshold: float = _DIR_THRESHOLD) -> np.ndarray:
    """Classify *terminal* (horizon) returns into up/flat/down.

    ``threshold`` is the volatility-adaptive band scaled by ``√horizon`` at the
    call site (terminal returns accumulate over the horizon), so the flat band
    means "small relative to a typical multi-bar move" instead of a fixed 0.3%
    that swallowed almost every 12-bar path.
    """
    return np.where(
        returns > threshold,
        "up",
        np.where(returns < -threshold, "down", "flat"),
    )


class ScenarioEngine:
    """Vectorized Monte Carlo path generator (rough-vol + Student-t)."""

    def __init__(
        self,
        n_scenarios: int = 100,
        horizon: int = 12,
        seed: int = 0,
        timeframe: str = "1h",
    ) -> None:
        self.n_scenarios = max(100, min(10_000, n_scenarios))
        self.horizon = horizon
        self.timeframe = timeframe
        self._rng = np.random.default_rng(seed)
        self._markov = MarkovChainModel(seed=seed)
        self._next_candle = NextCandlePredictor(seed=seed)
        # Heavy-tailed rough cone (Student-t innovations + rough-vol term
        # structure). The validated upgrade over the Gaussian flat-σ cone:
        # tests/measure_model.py::_cone_ladder_eval showed t_rough fixes the
        # 95/99 % tail under-coverage without hurting CRPS or directional Brier;
        # leverage was measured to regress direction (§8.4) and is left OFF.
        self._cone: ConeModel | None = None
        self._cone_step: int = -10 ** 9
        self._cone_key: tuple = ()
        self._cone_refit_every: int = 24
        # Volatility-regime HMM (3-state Baum-Welch on smoothed log-RV). Validated
        # to separate next-bar |return| better than the hardcoded heuristic; a
        # descriptive/generative regime, refit on the same cached cadence.
        self._regime_hmm = None
        self._regime_step: int = -10 ** 9
        self._regime_refit_every: int = 96
        self._vol_regime: str = "normal"
        self._vol_regime_probs: dict[str, float] = {}

    def _fit_cone(self, state: MarketState) -> None:
        """Refit the heavy-tailed rough cone at most every ``_cone_refit_every``
        bars (the ν MLE + rough-kernel fit is the only non-trivial cost); reuse
        the fitted ν/term-structure each tick. Mirrors VolForecaster's caching."""
        step = int(getattr(state, "step", len(state.history) - 1))
        end = step + 1
        key = (getattr(state, "symbol", ""), getattr(state, "source", ""))
        stale = (
            self._cone is None or key != self._cone_key
            or step < self._cone_step
            or step - self._cone_step >= self._cone_refit_every
        )
        if stale:
            o = state.opens[:end] if getattr(state, "opens", None) is not None else None
            hi = state.highs[:end] if getattr(state, "highs", None) is not None else None
            lo = state.lows[:end] if getattr(state, "lows", None) is not None else None
            self._cone = ConeModel.fit(o, hi, lo, state.history)
            self._cone_step, self._cone_key = step, key

    def _fit_regime(self, state: MarketState) -> None:
        """Refit the volatility-regime HMM at most every ``_regime_refit_every``
        bars; the causal live regime is read each tick from the cached model."""
        from core.hmm import RegimeHMM

        step = int(getattr(state, "step", len(state.history) - 1))
        end = step + 1
        stale = (
            self._regime_hmm is None or step < self._regime_step
            or step - self._regime_step >= self._regime_refit_every
        )
        o = state.opens[:end] if getattr(state, "opens", None) is not None else None
        hi = state.highs[:end] if getattr(state, "highs", None) is not None else None
        lo = state.lows[:end] if getattr(state, "lows", None) is not None else None
        if stale:
            # Light fit for the live loop: single init, capped window (the regime
            # is stable; keeps the per-refit hitch small in the TUI).
            fitted = RegimeHMM.fit(o, hi, lo, state.history, kind="vol3",
                                   n_init=1, n_iter=22, max_bars=1800)
            if fitted is not None:
                self._regime_hmm = fitted
                self._regime_step = step
        if self._regime_hmm is not None:
            # Fenêtre bornée pour le filtre causal : la passe forward est O(T) et
            # tournait sur TOUTE l'histoire à chaque tick (~31 ms à 1500 barres,
            # croissant sans borne). 1000 barres suffisent largement à mixer le
            # filtre (persistance ~0.85 ⇒ mémoire effective de quelques dizaines
            # de barres) — même régime, coût borné.
            w = 1000
            hist = state.history[-w:]
            hi_w = hi[-w:] if hi is not None else None
            lo_w = lo[-w:] if lo is not None else None
            o_w = o[-w:] if o is not None else None
            # opens inclus : real_ohlc en a besoin pour authentifier le range —
            # sans lui, le filtre retombait silencieusement sur le proxy r².
            self._vol_regime_probs = self._regime_hmm.regime_probs(
                hist, hi_w, lo_w, opens=o_w
            )
            self._vol_regime = max(self._vol_regime_probs, key=self._vol_regime_probs.get) \
                if self._vol_regime_probs else "normal"

    def generate(
        self,
        state: MarketState,
        *,
        candle_model=None,
        use_model: bool = False,
    ) -> ScenarioBundle:
        n = self.n_scenarios
        h = self.horizon
        price = state.price
        mom = state.momentum()

        # Matrice de transition de direction — l'entrée du forecast statistique
        # de la prochaine bougie et une statistique descriptive (plus aucun
        # échantillonnage de chemins : mesuré décoratif, §8.4).
        markov_snap = self._markov.fit(state)

        # --- Heavy-tailed rough-vol cone (validated path generator) ---------
        # The cone is sized by the rough-vol *term structure* (per-bar variance
        # forecast, the shoot-out-winning σ head) and shaped by Student-t
        # innovations (fat tails — the validated 95/99 % tail-coverage fix). The
        # centre stays a martingale (drift 0): the 1-bar direction is ≈50/50, so
        # a drifted cone was measured to hurt both density and direction (§1-2,
        # §8.4). The fit (ν, rough kernel) is cached and refit every N bars.
        self._fit_cone(state)
        self._fit_regime(state)
        end = int(state.step) + 1
        o = state.opens[:end] if getattr(state, "opens", None) is not None else None
        hi = state.highs[:end] if getattr(state, "highs", None) is not None else None
        lo = state.lows[:end] if getattr(state, "lows", None) is not None else None
        var_path = self._cone.var_path(o, hi, lo, state.history, h, rough=True)
        # One-step σ — the risk scale fed to the decision layer, the bubbles and
        # the conformal band. Floored at the module-wide SIGMA_FLOOR (1e-4), NOT
        # the old hardcoded 0.003: that constant bound on ~1/3 of calm BTC-1h
        # bars and silently clamped every consumer to an arbitrary width.
        vol = float(max(np.sqrt(var_path[0]), SIGMA_FLOOR))
        cone_params = ConeParams(nu=self._cone.nu, leverage=0.0, rough=True)
        log_paths = self._cone.sample_logpaths(self._rng, var_path, n, cone_params)

        paths = price * np.exp(log_paths)
        # Terminal LOG returns: the adaptive flat band is estimated on log
        # returns, and the log scale keeps the martingale centre symmetric
        # (arithmetic exp(Σr)−1 has a Jensen +h·σ²/2 bias that showed a small
        # but permanently positive E[r] on a no-edge model).
        terminal_returns = log_paths[:, -1].copy()

        full = np.concatenate([np.full((n, 1), price), paths], axis=1)
        running_max = np.maximum.accumulate(full, axis=1)
        max_dd = ((full - running_max) / running_max).min(axis=1)

        # Terminal-return likelihood at the TERMINAL scale: an h-bar return has
        # variance ≈ Σ_k var_k, not the one-step σ². Scoring 12-bar outcomes
        # against a 1-bar σ² made the kernel ~h× too peaked — the softmax
        # collapsed onto near-zero-return paths and the highlighted "scénario
        # d'action" was structurally a do-nothing trajectory.
        term_var = float(np.sum(var_path)) + 1e-8
        # Likelihood centre is shrunk with the same knob: a cone centred at the
        # martingale must not be re-weighted back toward momentum, or the
        # UI-facing prob_up/prob_down/expected_return inherit the directional
        # bias the measurement rejected (weighted dir-Brier ≫ equal-weight).
        lik_center = _DRIFT_SHRINK * mom
        log_likelihoods = -0.5 * ((terminal_returns - lik_center) ** 2) / term_var
        log_likelihoods -= 0.3 * (max_dd ** 2) / term_var

        logits = log_likelihoods - log_likelihoods.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        # Flat band for the *terminal* return ≈ half a typical h-bar move
        # (k·σ·√h), i.e. the per-bar adaptive band scaled by the horizon, so
        # classification matches the learned model's volatility-adaptive logic.
        term_thr = adaptive_threshold(state.history) * float(np.sqrt(max(h, 1)))
        directions = _classify_returns(terminal_returns, term_thr)
        most_probable_id = int(np.argmax(probs))
        # FORECAST (direction density) vs RISK-PICK (drawdown-weighted) — see
        # _FORECAST_EQUAL_WEIGHT. Equal-weight = the honest generative density.
        if _FORECAST_EQUAL_WEIGHT:
            expected_return = float(terminal_returns.mean())
            prob_up = float((directions == "up").mean())
            prob_down = float((directions == "down").mean())
        else:
            expected_return = float(np.dot(terminal_returns, probs))
            prob_up = float(probs[directions == "up"].sum())
            prob_down = float(probs[directions == "down"].sum())
        confidence = float(probs[most_probable_id])  # risk-pick confidence (softmax)

        store_k = min(_PATH_STORE_TOP_K, n)
        top_idx = np.argsort(probs)[-store_k:][::-1]
        path_store = {int(i): full[i] for i in top_idx}

        scenarios: list[Scenario] = []
        for i in range(n):
            stored_path = path_store.get(i, np.array([price, paths[i, -1]]))
            scenarios.append(
                Scenario(
                    id=i,
                    path=stored_path,
                    terminal_return=float(terminal_returns[i]),
                    max_drawdown=float(max_dd[i]),
                    probability=float(probs[i]),
                    direction=str(directions[i]),
                )
            )

        model = ProbModelSnapshot(
            estimated_vol=vol,
            estimated_momentum=mom,
            regime=state.regime.value,
            horizon=h,
            n_scenarios=n,
            mean_path_vol=float(np.sqrt(var_path).mean()),
            most_probable_id=most_probable_id,
            most_probable_prob=confidence,
            vol_regime=self._vol_regime,
            vol_regime_probs=dict(self._vol_regime_probs),
            cone_nu=float(getattr(self._cone, "nu", 0.0) or 0.0),
            rough_hurst=float(getattr(getattr(self._cone, "_rough", None), "hurst", 0.0) or 0.0),
        )

        next_forecast = self._next_candle.predict(
            state,
            self.timeframe,
            markov_snap,
            self._markov,
            model=candle_model,
            use_model=use_model,
            vol=vol,
            nu=float(getattr(self._cone, "nu", 0.0) or 0.0) or None,
        )

        return ScenarioBundle(
            scenarios=scenarios,
            horizon=h,
            current_price=price,
            expected_return=expected_return,
            prob_up=prob_up,
            prob_down=prob_down,
            confidence=confidence,
            model=model,
            markov=markov_snap,
            most_probable_id=most_probable_id,
            next_candle=next_forecast,
            timeframe=self.timeframe,
        )
