"""Next-candle probabilistic forecast — Markov + GBM + backtest calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.market import MarketState
from core.markov_model import DIRECTION_STATES, MarkovChainModel, MarkovSnapshot
from core.thresholds import adaptive_threshold, classify_dir, classify_idx
from core.timeutil import format_ts, timeframe_ms

_DIR_THRESHOLD = 0.003  # legacy fallback; live classification uses adaptive k·σ
_DIR_FR = {"up": "HAUSSE", "down": "BAISSE", "flat": "NEUTRE"}
_DIR_STYLE = {"up": "green", "down": "red", "flat": "yellow"}


@dataclass
class CandleBubble:
    """One probabilistic scenario for the next candle (bubble chart unit)."""

    id: int
    direction: str
    return_pct: float
    probability: float
    predicted_price: float
    source: str  # markov | gbm | empirical | blend

    @property
    def label_fr(self) -> str:
        return _DIR_FR[self.direction]

    @property
    def style(self) -> str:
        return _DIR_STYLE[self.direction]


@dataclass
class NextCandleForecast:
    """Aggregated forecast for the immediate next bar."""

    timeframe: str
    current_price: float
    bubbles: list[CandleBubble]
    most_probable: CandleBubble
    prob_up: float
    prob_down: float
    prob_flat: float
    expected_return: float
    markov_from: str
    markov_to: str
    markov_transition_prob: float
    backtest_hit_rate: float
    gbm_drift: float
    gbm_vol: float
    n_backtest_bars: int
    model_driven: bool = False  # True when a trained model drives the direction mass
    # --- identity of the candle being predicted (Live mode) ----------------
    target_step: int = -1       # bar index of the candle we are forecasting
    target_ts: int | None = None  # UTC ms of that candle's open (None = unknown)
    base_step: int = -1         # bar the forecast is anchored on (the last close)

    @property
    def target_dt(self) -> str:
        """``JJ/MM HH:MM`` of the predicted candle (or ``—`` if unknown)."""
        return format_ts(self.target_ts)

    @property
    def consensus_margin(self) -> float:
        """Écart entre la 1ʳᵉ et la 2ᵉ masse calibrée — 0 = pile équiprobable."""
        m = sorted((self.prob_up, self.prob_flat, self.prob_down), reverse=True)
        return float(m[0] - m[1])

    @property
    def low_confidence(self) -> bool:
        """True quand l'argmax n'est qu'une courte tête (marge top-2 < 5 pts).

        Sur ce substrat martingale les masses gravitent autour de ⅓/⅓/⅓ (marge
        moyenne mesurée ≈ 5 pts, 2026-07-24) : afficher « NEUTRE (P=35 %) » sans
        qualificatif faisait passer un quasi-tirage au sort pour un appel assumé
        — et NEUTRE sortait 46 % du temps pour 35 % réalisé, non parce que le
        modèle « croit » au neutre mais parce que l'argmax départage des masses
        presque égales. La règle de décision reste l'argmax (optimale pour le
        hit rate, et c'est elle que score la fiabilité) ; seul l'AFFICHAGE dit
        désormais quand l'appel est faible."""
        return self.consensus_margin < 0.05

    def candle_line(self) -> str:
        """One line naming the candle being predicted + its date/time."""
        n = f"#{self.target_step}" if self.target_step >= 0 else ""
        return (
            f"🎯 Bougie visée {n} [{self.timeframe}] — "
            f"ouverture {self.target_dt} (UTC)"
        )

    def summary_line(self) -> str:
        # Mener avec le CONSENSUS CALIBRÉ (argmax des masses) — la même lecture
        # que le bandeau, le panneau bulles et la fiabilité Live. La bulle ★
        # (pick pondéré risque) reste dans detail_line, étiquetée comme telle.
        masses = {"up": self.prob_up, "flat": self.prob_flat, "down": self.prob_down}
        top = max(masses, key=masses.get)  # type: ignore[arg-type]
        tag = "🧠 modèle" if self.model_driven else "📊 stat"
        n = f"#{self.target_step} " if self.target_step >= 0 else ""
        weak = (
            f", Δ{self.consensus_margin:.0%} — quasi équiprobable"
            if self.low_confidence else ""
        )
        return (
            f"🔮 Bougie {n}[{self.timeframe}·{tag}] {self.target_dt} → "
            f"[{_DIR_STYLE[top]}]{_DIR_FR[top]}[/] (P={masses[top]:.0%}{weak}) | "
            f"P(↑/●/↓)={self.prob_up:.0%}/{self.prob_flat:.0%}/{self.prob_down:.0%} | "
            f"E[r]={self.expected_return:+.3%} | "
            f"Markov {self.markov_from}→{self.markov_to} ({self.markov_transition_prob:.1%}) | "
            f"backtest={self.backtest_hit_rate:.0%}"
        )

    def detail_line(self) -> str:
        mp = self.most_probable
        return (
            f"   ★ Scénario #{mp.id} [{mp.source}] — "
            f"cible ${mp.predicted_price:,.2f} | "
            f"σ_GBM={self.gbm_vol:.4f} μ={self.gbm_drift:+.5f} | "
            f"{self.n_backtest_bars} barres calibrées"
        )


def _classify(ret: float, threshold: float = _DIR_THRESHOLD) -> str:
    return classify_dir(ret, threshold)


def _classify_idx(ret: float, threshold: float = _DIR_THRESHOLD) -> int:
    return classify_idx(ret, threshold)


class NextCandlePredictor:
    """Hybrid Markov + stochastic (GBM) next-bar forecaster with backtest calibration."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(
        self,
        state: MarketState,
        timeframe: str,
        markov_snap: MarkovSnapshot,
        markov_model: MarkovChainModel,
        *,
        model=None,
        use_model: bool = False,
        vol: float | None = None,
        nu: float | None = None,
    ) -> NextCandleForecast:
        price = state.price
        # Next-bar σ for the magnitude head. The scenario engine passes the
        # cached rough-vol forecast (range-efficient) so the bubble spread / GBM
        # band share the same sharper σ; direct callers fall back to the EWMA
        # estimate. Floored at SIGMA_FLOOR (1e-4) — the old 0.003 constant bound
        # on ~1/3 of calm 1h bars and overstated every displayed move there.
        from ml.vol_model import SIGMA_FLOOR
        vol = max(vol if vol is not None else state.ewma_volatility(), SIGMA_FLOOR)
        mom = state.momentum()
        returns = state.returns(window=min(250, len(state.history) - 1))
        # Volatility-adaptive flat band (per timeframe), not a fixed 0.3%.
        thr = adaptive_threshold(getattr(state, "history", None))

        markov_probs, from_dir, to_dir, trans_p = self._markov_next_probs(markov_snap)
        empirical = self._empirical_by_direction(returns, thr)
        hit_rate, n_bt = self._backtest_hit_rate(returns, thr)

        # Live (prédiction) mode: a trained model OWNS the direction mass — its
        # learned hausse/neutre/baisse probabilities drive the bubbles, while GBM
        # still supplies the return magnitude/spread. DIRECTION_STATES is
        # ("up","flat","down"), the same order the model emits, so no remapping.
        dir_probs = markov_probs
        model_driven = False
        if (
            use_model
            and model is not None
            and getattr(model, "trained", False)
            and len(state.history) >= 36
        ):
            learned = np.asarray(model.predict_proba(state.history), dtype=np.float64)
            if learned.shape == (3,) and np.isfinite(learned).all():
                dir_probs = learned
                model_driven = True

        # Martingale centre (no momentum drift): the 1-bar direction is ≈50/50,
        # so the cone is centred at the current price (see core/scenarios.py and
        # tests/test_scenario_drift). The only directional tilt allowed is the one
        # the *calibrated model* (or Markov) asserts via dir_probs — never a
        # hardcoded momentum drift. In stat mode dir_probs≈⅓⅓⅓ ⇒ ~0 tilt.
        gbm_drift = vol * dir_probs[0] * 0.5 - vol * dir_probs[2] * 0.5
        gbm_mean = gbm_drift
        gbm_vol = vol

        # Blend the direction mass (Markov, or learned in live mode) with
        # GBM / empirical returns.
        bubbles: list[CandleBubble] = []
        blend_weights = {"up": 0.0, "flat": 0.0, "down": 0.0}
        for d_idx, direction in enumerate(DIRECTION_STATES):
            p_dir = float(dir_probs[d_idx])
            blend_weights[direction] = p_dir
            emp_rets = empirical.get(direction, np.array([0.0]))
            for q, w in zip((0.25, 0.50, 0.75), (0.25, 0.50, 0.25)):
                ret_q = float(np.quantile(emp_rets, q)) if len(emp_rets) > 3 else gbm_mean
                gbm_ret = gbm_mean + (q - 0.5) * gbm_vol * 2
                ret_blend = 0.55 * ret_q + 0.45 * gbm_ret
                prob = p_dir * w
                if prob < 0.005:
                    continue
                bubbles.append(
                    CandleBubble(
                        id=len(bubbles),
                        direction=direction,
                        return_pct=ret_blend,
                        probability=prob,
                        predicted_price=price * (1.0 + ret_blend),
                        source="blend",
                    )
                )

        # GBM Monte Carlo micro-scenarios (stochastic layer). In live mode the
        # learned distribution should own the headline, so the GBM mass is
        # damped — it still textures the chart but no longer flips which
        # direction is most probable.
        gbm_share = 0.12 if model_driven else 0.35
        n_mc = 24
        # Student-t micro-shocks (fat tails), consistent with the scenario cone.
        # Standardised to unit variance so ν only changes tail shape, not σ.
        if nu is not None and 2.0 < float(nu) < 1e6:
            from core.cone import standardized_t
            shocks = gbm_mean + gbm_vol * standardized_t(self._rng, float(nu), n_mc)
        else:
            shocks = self._rng.normal(gbm_mean, gbm_vol, n_mc)
        for shock in shocks:
            direction = _classify(float(shock), thr)
            prob = 1.0 / n_mc * gbm_share
            bubbles.append(
                CandleBubble(
                    id=len(bubbles),
                    direction=direction,
                    return_pct=float(shock),
                    probability=prob,
                    predicted_price=price * (1.0 + shock),
                    source="gbm",
                )
            )

        # Normalize probabilities
        total_p = sum(b.probability for b in bubbles) or 1.0
        for b in bubbles:
            b.probability /= total_p

        bubbles.sort(key=lambda b: b.probability, reverse=True)
        for i, b in enumerate(bubbles):
            b.id = i

        # Merge near-duplicate bubbles for cleaner chart (top 12)
        bubbles = self._merge_bubbles(bubbles, max_bubbles=12)

        # Headline direction masses = the CALIBRATED source (learned model in
        # live mode, Markov row in stat mode) — NOT the bubble cloud. The cloud
        # mixes in 24 random micro-shocks for chart texture; averaging them into
        # the headline diluted the calibrated probabilities toward ⅓/⅓/⅓ and
        # added bar-to-bar sampling noise to the very number the live Brier
        # scores. Display texture must not contaminate the scored forecast.
        masses = {
            "up": float(dir_probs[0]),
            "flat": float(dir_probs[1]),
            "down": float(dir_probs[2]),
        }
        exp_ret = 0.0
        for b in bubbles:
            exp_ret += b.probability * b.return_pct

        mp = max(bubbles, key=lambda b: b.probability)

        base_step = int(state.step)
        target_step = base_step + 1
        target_ts = self._target_timestamp(state, target_step, timeframe)

        return NextCandleForecast(
            timeframe=timeframe,
            current_price=price,
            bubbles=bubbles,
            most_probable=mp,
            prob_up=masses["up"],
            prob_down=masses["down"],
            prob_flat=masses["flat"],
            expected_return=exp_ret,
            markov_from=from_dir,
            markov_to=to_dir,
            markov_transition_prob=trans_p,
            backtest_hit_rate=hit_rate,
            gbm_drift=gbm_drift,
            gbm_vol=gbm_vol,
            n_backtest_bars=n_bt,
            model_driven=model_driven,
            target_step=target_step,
            target_ts=target_ts,
            base_step=base_step,
        )

    @staticmethod
    def _target_timestamp(
        state: MarketState, target_step: int, timeframe: str
    ) -> int | None:
        """UTC ms of the candle at ``target_step``.

        Uses the loaded timestamps when that bar already exists; otherwise
        extrapolates from the last known bar by one timeframe (live: the next
        candle hasn't opened yet).
        """
        ts = getattr(state, "timestamps", None)
        if ts is None or len(ts) == 0:
            return None
        if 0 <= target_step < len(ts):
            return int(ts[target_step])
        step = timeframe_ms(timeframe or "1h")
        return int(ts[-1]) + step * (target_step - (len(ts) - 1))

    def _markov_next_probs(
        self, snap: MarkovSnapshot
    ) -> tuple[np.ndarray, str, str, float]:
        if snap.direction_matrix is None:
            probs = np.array([1 / 3, 1 / 3, 1 / 3])
            return probs, snap.start_direction, "flat", 1 / 3

        from_idx = DIRECTION_STATES.index(snap.start_direction)
        probs = snap.direction_matrix[from_idx].copy()
        to_idx = int(np.argmax(probs))
        return (
            probs,
            snap.start_direction,
            DIRECTION_STATES[to_idx],
            float(probs[to_idx]),
        )

    def _empirical_by_direction(
        self, returns: np.ndarray, threshold: float = _DIR_THRESHOLD
    ) -> dict[str, np.ndarray]:
        buckets: dict[str, list[float]] = {"up": [], "flat": [], "down": []}
        for r in returns:
            buckets[_classify(float(r), threshold)].append(float(r))
        return {k: np.array(v) if v else np.array([0.0]) for k, v in buckets.items()}

    def _backtest_hit_rate(
        self, returns: np.ndarray, threshold: float = _DIR_THRESHOLD
    ) -> tuple[float, int]:
        """Walk-forward directional hit rate of a 1st-order Markov predictor.

        Honest (no look-ahead): the transition counts are built **online** from
        bars ``0..i`` only, and used to predict bar ``i+1`` before that bar is
        folded into the counts. This replaces the previous in-sample estimate
        that fitted the transition matrix on the whole series and then "predicted"
        the very same returns (always optimistic).
        """
        n = len(returns)
        if n < 20:
            return 0.5, max(n - 1, 0)

        counts = np.ones((3, 3))  # Laplace prior so early predictions are defined
        correct = 0
        total = 0
        prev = _classify_idx(float(returns[0]), threshold)
        for i in range(1, n):
            actual = _classify_idx(float(returns[i]), threshold)
            pred = int(np.argmax(counts[prev]))  # uses only bars seen so far
            if pred == actual:
                correct += 1
            total += 1
            counts[prev, actual] += 1  # learn the transition AFTER predicting
            prev = actual
        return correct / max(total, 1), total

    def _merge_bubbles(
        self, bubbles: list[CandleBubble], max_bubbles: int = 12
    ) -> list[CandleBubble]:
        if len(bubbles) <= max_bubbles:
            return bubbles

        merged: list[CandleBubble] = []
        used = set()
        for b in bubbles:
            if b.id in used:
                continue
            cluster = [b]
            used.add(b.id)
            for other in bubbles:
                if other.id in used:
                    continue
                if (
                    other.direction == b.direction
                    and abs(other.return_pct - b.return_pct) < 0.004
                ):
                    cluster.append(other)
                    used.add(other.id)
            total_w = sum(c.probability for c in cluster)
            ret_w = sum(c.probability * c.return_pct for c in cluster) / total_w
            merged.append(
                CandleBubble(
                    id=len(merged),
                    direction=b.direction,
                    return_pct=ret_w,
                    probability=total_w,
                    predicted_price=b.predicted_price,
                    source=b.source,
                )
            )
            if len(merged) >= max_bubbles:
                break

        total = sum(m.probability for m in merged) or 1.0
        for m in merged:
            m.probability /= total
        merged.sort(key=lambda m: m.probability, reverse=True)
        for i, m in enumerate(merged):
            m.id = i
        return merged