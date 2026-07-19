"""Tests — Live mode chained walk-forward + self-evaluation reliability."""
from __future__ import annotations

import numpy as np

from core.engine import SimulationSession
from core.live_eval import LiveEvaluator, classify
from core.timeutil import format_ts, timeframe_minutes, timeframe_ms
from ml.candle_model import train_candle_model


def test_timeframe_helpers():
    assert timeframe_minutes("1m") == 1
    assert timeframe_minutes("15m") == 15
    assert timeframe_minutes("1h") == 60
    assert timeframe_minutes("4h") == 240
    assert timeframe_minutes("1d") == 60 * 24
    assert timeframe_ms("1h") == 3_600_000
    # known epoch ms → fixed UTC render
    assert format_ts(0) == "01/01 00:00"
    assert format_ts(None) == "—"


def test_forecast_names_target_candle():
    s = SimulationSession.create(asset="BTC/USDT")
    step0 = s.bot.market.step
    s.tick()
    fc = s.bot.current_bundle.next_candle
    # Le forecast généré sur la barre step0 vise la bougie suivante. (En TRAIN
    # le tick avance ensuite le marché sur cette bougie ; en LIVE il reste.)
    assert fc.target_step == step0 + 1
    # cached/synthetic data carries timestamps → target datetime is resolvable
    if fc.target_ts is not None:
        assert fc.target_dt != "—"
        # Guard the ms-resolution bug: real cached bars must decode to a recent
        # year, not 1970 (a double ns→ms divide lands the epoch near 00:29).
        from datetime import datetime, timezone

        year = datetime.fromtimestamp(fc.target_ts / 1000.0, tz=timezone.utc).year
        if s.bot_engine.simulator.source.startswith(("cache", "ccxt", "live")):
            assert year >= 2020, f"timestamps decoded to {year} — ms/ns resolution bug"
    assert f"#{fc.target_step}" in fc.summary_line()
    assert f"#{fc.target_step}" in fc.candle_line()


def test_evaluator_scores_direction_and_grade():
    ev = LiveEvaluator()
    ev.reset()

    class _MP:
        def __init__(self, d, r):
            self.direction, self.return_pct, self.probability = d, r, 0.5

    class _FC:
        def __init__(self, step, d, r):
            self.most_probable = _MP(d, r)
            self.target_step, self.target_ts = step, None
            self.current_price, self.model_driven, self.timeframe = 100.0, True, "1h"
            # Confident UP forecast → drives the Brier path in settle().
            self.prob_up, self.prob_flat, self.prob_down = 0.7, 0.2, 0.1

    # Predict UP, realise +5% → correct
    ev.open_from_forecast(_FC(1, "up", 0.05))
    s = ev.settle(105.0)
    assert s.correct and s.realized_dir == "up"
    # Predict UP, realise -5% → wrong
    ev.open_from_forecast(_FC(2, "up", 0.05))
    s = ev.settle(95.0)
    assert not s.correct and s.realized_dir == "down"

    assert ev.n_eval == 2
    assert ev.n_correct == 1
    assert abs(ev.accuracy - 0.5) < 1e-9
    # Brier accumulated over both settles: correct-up (0.3²+0.2²+0.1²=0.14) +
    # wrong-up where down realised (0.7²+0.2²+0.9²=1.34) → mean 0.74.
    assert abs(ev.mean_brier - 0.74) < 1e-6
    assert 0.0 <= ev.reliability_note <= 100.0
    # too few samples → flagged as needing more training
    _label, _style, needs = ev.grade()
    assert needs is True
    report = ev.finalize()
    assert any("fiabilité" in line.lower() for line in report)


def test_live_walk_forward_chains_and_finalizes():
    s = SimulationSession.create(asset="BTC/USDT")
    prices = s.bot_engine.simulator.prices
    model, _ = train_candle_model(prices, symbol="BTC/USDT", timeframe="1h", epochs=50)
    s.bot.candle_model = model

    on = s.set_predict_mode(True)
    assert on is True
    assert s.bot.predict_mode

    start_step = s.bot.market.step
    # Walk a chunk of candles: the bot must advance and evaluate, not freeze.
    for _ in range(40):
        s.tick()
    assert s.bot.market.step > start_step, "Live mode must chain forward, not freeze"
    assert s.bot.live_eval.n_eval > 0, "predictions must be settled as candles close"

    # Toggling Live off finalizes and produces a reliability grade.
    s.set_predict_mode(False)
    assert s.bot.live_eval.finalized
    assert any("fiabilité" in m.lower() for m in s.bot.log)


def test_classify_thresholds():
    assert classify(0.01) == "up"
    assert classify(-0.01) == "down"
    assert classify(0.0) == "flat"


if __name__ == "__main__":
    test_timeframe_helpers()
    test_forecast_names_target_candle()
    test_evaluator_scores_direction_and_grade()
    test_live_walk_forward_chains_and_finalizes()
    test_classify_thresholds()
    print("✓ live-eval walk-forward tests OK")
