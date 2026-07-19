"""Tests — learned next-candle model (gradient descent, persistence, parity)."""
from __future__ import annotations

import numpy as np

from core.market import MarketSimulator
from ml.candle_model import (
    DIRECTIONS,
    N_FEATURES,
    CandleModel,
    build_dataset,
    candle_features,
    classify_return,
    load_model,
    model_path,
    train_candle_model,
)


def _prices() -> np.ndarray:
    return MarketSimulator(n_bars=1200, seed=11).prices


def test_features_are_lookahead_free_and_finite():
    p = _prices()
    # Features at bar i must equal features recomputed from the prefix only.
    f_full = candle_features(p[:300])
    f_prefix = candle_features(p[:300].copy())
    assert np.allclose(f_full, f_prefix)
    assert f_full.shape == (N_FEATURES,)
    assert np.isfinite(f_full).all()
    # Features for bar 299 depend only on prices[:300]; appending future bars
    # (prices[:400]) and re-slicing to the same prefix yields the same vector.
    assert np.allclose(candle_features(p[:300]), candle_features(p[:400][:300]))


def test_build_dataset_shapes_and_labels():
    p = _prices()
    X, y = build_dataset(p, warmup=35)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == N_FEATURES
    assert set(np.unique(y)).issubset({0, 1, 2})
    assert not np.isnan(X).any()
    # Label parity with classify_return on the realised next-bar log return,
    # using the *local* band (the band live inference recomputes per bar).
    from core.thresholds import adaptive_threshold
    i = 100
    thr_i = adaptive_threshold(p[: i + 1])
    expected = classify_return(float(np.log(p[i + 1] / p[i])), thr_i)
    assert y[i - 35] == expected


def test_training_reduces_loss_and_predicts_distribution():
    p = _prices()
    model, rep = train_candle_model(p, symbol="TST/USDT", timeframe="1h", epochs=300, lr=0.5)
    assert model.trained
    assert rep.loss_history[0] >= rep.final_loss            # gradient descent lowered the loss
    assert rep.loss_history[0] > 1.0                        # starts near uniform (ln 3 ≈ 1.0986)
    proba = model.predict_proba(p[:400])
    assert proba.shape == (3,)
    assert abs(float(proba.sum()) - 1.0) < 1e-6
    assert model.predict_direction(p[:400]) in DIRECTIONS


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    from pathlib import Path

    p = _prices()
    model, _ = train_candle_model(p, symbol="RT/USDT", timeframe="4h", epochs=120, lr=0.5)
    out = Path(tempfile.mkdtemp()) / "rt.npz"
    model.save(out)
    loaded = CandleModel.load(out)
    assert loaded.trained and loaded.symbol == "RT/USDT" and loaded.timeframe == "4h"
    assert np.allclose(model.predict_proba(p[:400]), loaded.predict_proba(p[:400]))


def test_session_train_switches_to_live_and_drives_bubbles():
    """End-to-end: training flips the bot to Live and the model drives bubbles."""
    import tempfile
    from pathlib import Path

    import ml.candle_model as cm
    from core.engine import SimulationSession
    from exchange.live_client import ExchangeMode

    orig_dir = cm.MODELS_DIR
    cm.MODELS_DIR = Path(tempfile.mkdtemp())  # isolate persisted weights from real ones
    try:
        s = SimulationSession.create(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
        assert s.model_status()["trained"] is False
        rep = s.train_model(epochs=120, lr=0.5)
        assert rep.n_train > 0 and rep.val_majority > 0.0
        status = s.model_status()
        assert status["trained"] is True
        assert status["predict_mode"] is True              # auto-switch to Live
        assert s.train_report is not None and s.train_report.loss_history
        s.tick()
        assert s.bot.current_bundle.next_candle.model_driven is True

        # Live = chained walk-forward prediction: the bot advances candle by
        # candle and self-evaluates each closed bar, but never trades.
        step0 = s.bot.market.step
        trades0 = len(s.bot.portfolio.trades)
        for _ in range(5):
            s.tick()
        assert s.bot.market.step > step0, "Live mode must chain forward, not freeze"
        assert len(s.bot.portfolio.trades) == trades0, "Live mode placed trades (should only predict)"
        assert s.bot.live_eval.n_eval > 0, "Live mode must self-evaluate closed candles"

        # Toggling back to Entraînement returns to the statistical forecast + replay.
        s.set_predict_mode(False)
        s.tick()
        assert s.bot.current_bundle.next_candle.model_driven is False
    finally:
        cm.MODELS_DIR = orig_dir


if __name__ == "__main__":
    test_features_are_lookahead_free_and_finite()
    test_build_dataset_shapes_and_labels()
    test_training_reduces_loss_and_predicts_distribution()
    test_save_load_roundtrip()
    test_session_train_switches_to_live_and_drives_bubbles()
    print("✓ candle model tests OK")
