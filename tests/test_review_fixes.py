"""Tests des correctifs de la revue « convergence neutre » (2026-07-24).

1. La porte Wilson du tilt ne se juge plus que sur les appels DIRECTIONNELS :
   un skill de fréquence de classe (flats bien datés = timing de vol, zéro
   information de signe) ouvrait l'ancienne porte globale.
2. Le wrap de fin d'historique des épisodes walk-forward est marqué (stats +
   journal) au lieu de rejouer la même fenêtre en silence.
3. Le bandeau qualifie l'appel de « quasi équiprobable » quand la marge top-2
   des masses calibrées est < 5 pts (l'argmax reste la règle scorée).
4. L'entraînement utilise l'historique profond du cache et l'ancrage OOS des
   marches Live/Paper se fait par TIMESTAMP (train_end_ts), plus par index.
"""
from __future__ import annotations

import os
import tempfile

_MODELS_TMP = tempfile.mkdtemp(prefix="bot-models-review-")
os.environ["BOT_MODELS_DIR"] = _MODELS_TMP  # jamais écraser les vrais models/*.npz

import numpy as np

from config.market_config import WARMUP_BARS
from core.engine import SimulationSession
from core.live_eval import LiveEvaluator
from core.next_candle import CandleBubble, NextCandleForecast
from ml.candle_model import CandleModel, train_candle_model


class _FC:
    """Forecast forgé : la direction scorée = argmax des masses (comme en prod)."""

    def __init__(self, step: int, probs: tuple[float, float, float]):
        self.prob_up, self.prob_flat, self.prob_down = probs
        idx = max(range(3), key=lambda i: probs[i])
        d = ("up", "flat", "down")[idx]

        class _MP:
            direction, return_pct, probability = d, 0.0, probs[idx]

        self.most_probable = _MP()
        self.target_step, self.target_ts = step, None
        self.current_price, self.model_driven, self.timeframe = 100.0, True, "1h"
        self.expected_return, self.gbm_vol = 0.0, 0.0


_UP = (0.6, 0.2, 0.2)
_FLAT = (0.2, 0.6, 0.2)
_DOWN = (0.2, 0.2, 0.6)


def _feed(ev: LiveEvaluator, probs, realized_price: float, n: int, start: int) -> int:
    for i in range(n):
        ev.open_from_forecast(_FC(start + i, probs))
        ev.settle(realized_price)
    return start + n


def test_gate_ignores_flat_frequency_skill():
    """Skill de fréquence pur (flats bien datés, signe au hasard) : l'ancienne
    porte globale s'ouvrait, la porte directionnelle doit rester fermée."""
    ev = LiveEvaluator()
    ev.reset()
    step = 0
    step = _feed(ev, _FLAT, 100.0, 45, step)   # 45 appels flat justes
    step = _feed(ev, _FLAT, 105.0, 15, step)   # 15 appels flat ratés (hausse)
    step = _feed(ev, _UP, 105.0, 10, step)     # 10 appels up justes
    step = _feed(ev, _UP, 95.0, 20, step)      # 20 appels up ratés (baisse)

    assert ev.n_eval == 90 and abs(ev.accuracy - 55 / 90) < 1e-9
    # Le scénario reproduit bien le défaut : le test GLOBAL (ex-porte) passe…
    assert ev.acc_lower_bound > ev.baseline, "scénario censé ouvrir l'ancienne porte"
    # …mais la porte directionnelle voit 10/30 contre une majorité à 20/30.
    assert ev.n_dir == 30 and ev.n_dir_correct == 10
    assert abs(ev.dir_baseline - 20 / 30) < 1e-9
    assert ev.significant_edge is False
    print("✓ porte fermée sur skill de fréquence (ancienne porte : ouverte)")


def test_gate_opens_on_real_directional_skill():
    ev = LiveEvaluator()
    ev.reset()
    step = 0
    step = _feed(ev, _UP, 105.0, 16, step)     # up appelé, hausse réalisée
    step = _feed(ev, _UP, 95.0, 4, step)
    step = _feed(ev, _DOWN, 95.0, 16, step)    # down appelé, baisse réalisée
    step = _feed(ev, _DOWN, 105.0, 4, step)

    assert ev.n_dir == 40 and ev.n_dir_correct == 32
    assert abs(ev.dir_baseline - 0.5) < 1e-9
    assert ev.significant_edge is True
    label, _style, needs = ev.grade()
    assert "edge directionnel" in label and needs is False
    assert any("Appels directionnels" in l for l in ev.finalize())
    print("✓ porte ouverte sur vrai skill directionnel (80 % sur 40 appels)")


def test_wrap_is_flagged_and_logged():
    s = SimulationSession.create(asset="BTC/USDT")
    sim = s.bot_engine.simulator
    n = sim.n_bars
    # Ancrer l'épisode tout près de la fin : il franchit n-60 en 5 pas → wrap.
    s.bot = s.bot_engine.new_episode(
        episode=1, risk=s.bot.risk, market_sim=sim,
        candle_model=None, predict_mode=False, start_step=n - 62,
    )
    st1 = s.train_episode(steps=5)
    assert st1["window"][0] == n - 62 and st1["wrapped"] is False
    assert s.bot.episode_wrapped is True, "le bot suivant doit porter le marqueur ⟲"
    assert any("⟲" in m for m in s.bot.log), "le wrap doit s'annoncer au journal"
    st2 = s.train_episode(steps=5)
    assert st2["wrapped"] is True and st2["window"][0] == WARMUP_BARS
    print("✓ wrap d'historique marqué (stats + journal) au lieu d'un rejeu muet")


def _forecast(probs: tuple[float, float, float]) -> NextCandleForecast:
    b = CandleBubble(id=0, direction="flat", return_pct=0.0,
                     probability=1.0, predicted_price=100.0, source="blend")
    return NextCandleForecast(
        timeframe="1h", current_price=100.0, bubbles=[b], most_probable=b,
        prob_up=probs[0], prob_flat=probs[1], prob_down=probs[2],
        expected_return=0.0, markov_from="flat", markov_to="flat",
        markov_transition_prob=probs[1], backtest_hit_rate=0.5,
        gbm_drift=0.0, gbm_vol=0.01, n_backtest_bars=100,
    )


def test_summary_flags_near_uniform_masses():
    weak = _forecast((0.34, 0.33, 0.33))
    assert weak.low_confidence and "quasi équiprobable" in weak.summary_line()
    strong = _forecast((0.50, 0.30, 0.20))
    assert not strong.low_confidence
    assert "quasi équiprobable" not in strong.summary_line()
    print("✓ bandeau : courte tête annoncée, appel net non pollué")


def test_train_end_ts_roundtrip():
    rng = np.random.default_rng(7)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400)))
    ts = 1_700_000_000_000 + np.arange(400, dtype=np.int64) * 3_600_000
    model, _rep = train_candle_model(prices, timestamps=ts, symbol="TEST/USDT",
                                     timeframe="1h", epochs=30)
    assert model.trained
    cut = int(len(prices) * 0.85)
    assert model.train_end_ts == int(ts[cut - 1])
    path = model.save()
    loaded = CandleModel.load(path)
    assert loaded.train_end_ts == model.train_end_ts
    print("✓ train_end_ts calculé (fin de zone touchée) et persisté (round-trip)")


def test_oos_anchor_by_timestamp_and_index_fallbacks():
    s = SimulationSession.create(asset="BTC/USDT")
    sim = s.bot_engine.simulator
    n = sim.n_bars
    if sim.timestamps is None:
        print("⚠ pas de timestamps (source synthétique ?) — repli index seulement")
    else:
        k = min(900, n - 10)
        s.bot.candle_model = CandleModel(trained=True,
                                         train_end_ts=int(sim.timestamps[k]))
        assert s._oos_walk_start(sim) == max(WARMUP_BARS, k + 1)
    # Index hors fenêtre (modèle entraîné sur un tableau plus long, sans ts) :
    # l'ancien code rognait à n-2 (marche OOS d'une barre) — désormais queue 20 %.
    s.bot.candle_model = CandleModel(trained=True, train_end_step=n + 5000)
    assert s._oos_walk_start(sim) == min(max(WARMUP_BARS, int(n * 0.8)), n - 2)
    # Ancien schéma (index valide dans la fenêtre) : comportement inchangé.
    s.bot.candle_model = CandleModel(trained=True, train_end_step=min(1200, n - 10))
    assert s._oos_walk_start(sim) == min(1200, n - 10) + 1
    print("✓ ancrage OOS : timestamp d'abord, replis index/queue sans rognage à n-2")


def test_train_model_uses_deep_history():
    s = SimulationSession.create(asset="BTC/USDT")
    sim = s.bot_engine.simulator
    if not str(sim.source).startswith(("cache", "ccxt", "live")):
        print("⚠ source non réelle — test historique profond sauté")
        return
    rep = s.train_model(epochs=30)
    model = s.bot.candle_model
    assert model is not None and model.trained and rep.n_train > 0
    assert model.train_end_ts > 0, "les timestamps de session doivent ancrer le modèle"
    deep = any(m.startswith("🗄") for m in s.bot.log)
    session_cap = sim.n_bars  # borne large : n_samples > fenêtre ⇒ historique profond
    if deep:
        assert model.n_samples > session_cap, (
            f"historique profond annoncé mais n_samples={model.n_samples} "
            f"≤ fenêtre {session_cap}"
        )
        runway = sim.n_bars - s._eval_start
        assert runway > 300, f"marche OOS trop courte ({runway} barres) — ancrage ts cassé ?"
        print(f"✓ entraînement profond : {model.n_samples} échantillons, "
              f"marche Live OOS de {runway} barres")
    else:
        print("⚠ cache profond indisponible/désaligné — repli fenêtre session (accepté)")


if __name__ == "__main__":
    test_gate_ignores_flat_frequency_skill()
    test_gate_opens_on_real_directional_skill()
    test_wrap_is_flagged_and_logged()
    test_summary_flags_near_uniform_masses()
    test_train_end_ts_roundtrip()
    test_oos_anchor_by_timestamp_and_index_fallbacks()
    test_train_model_uses_deep_history()
    print("✓ review-fixes tests OK")
