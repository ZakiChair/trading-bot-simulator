"""Tests — Live self-reinforcement: online SGD on the candle model.

The model adapts to each *settled* candle by one gradient step (regime tracking
+ calibration), anchored to the validated batch weights so it can't wander, and
kept in memory only (never persisted — persisting online drift would resurrect
the non-OOS look-ahead bug). These tests pin that behaviour:

* one online step reduces the loss on its own sample and pins an anchor;
* the trust region bounds drift even under an adversarial one-sided stream;
* ``features_asof`` reproduces exactly the vector ``predict_proba`` used;
* ``reset_online`` restores the clean weights;
* the engine reinforces in Live (n_online == n_eval), stays bounded, and a toggle
  off snaps the weights back to the anchor with no further updates.
"""
from __future__ import annotations

import numpy as np

from core.live_eval import SettledPrediction
from ml.candle_model import _softmax, train_candle_model


def _model(seed: int = 3, n: int = 800):
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(rng.normal(0, 0.01, n).cumsum())
    model, _ = train_candle_model(prices, symbol="TEST/USDT", timeframe="1h", epochs=200)
    assert model.trained
    return model, prices


def test_online_update_reduces_loss_and_sets_anchor():
    model, prices = _model()
    x = model.features_asof(prices[:500])
    assert x is not None and x.shape == (model.weights.shape[0],)
    assert model.anchor_weights is None and model.n_online == 0
    nlls = [model.online_update(x, 0)["nll"] for _ in range(30)]  # repeatedly "up"
    assert model.anchor_weights is not None, "first update must pin the anchor"
    assert model.n_online == 30
    assert nlls[-1] < nlls[0], "online step must reduce the loss on its own sample"
    assert model.online_recent_nll() > 0.0
    assert not np.allclose(model.weights, model.anchor_weights), "weights must move"


def test_trust_region_bounds_drift_under_adversarial_stream():
    model, prices = _model()
    x = model.features_asof(prices[:500])
    drifts = []
    for _ in range(200):  # hammer a single target — worst case for drift
        model.online_update(x, 0)
        drifts.append(model.online_drift())
    assert max(drifts) <= model.online_max_drift + 1e-9, "trust region must cap drift"
    assert np.isfinite(model.weights).all()


def test_features_asof_parity_and_guards():
    model, prices = _model()
    x = model.features_asof(prices[:500])
    # softmax((x @ W + b)/T) must equal predict_proba exactly (same input path).
    z = (x @ model.weights + model.bias) / model.temperature
    expected = _softmax(z.reshape(1, -1))[0]
    assert np.allclose(expected, model.predict_proba(prices[:500]))
    # Guards: too-short history and untrained model return None.
    assert model.features_asof(prices[:3]) is None
    model.trained = False
    assert model.features_asof(prices[:500]) is None


def test_reset_restores_clean_weights():
    model, prices = _model()
    x = model.features_asof(prices[:500])
    clean = model.weights.copy()
    for _ in range(20):
        model.online_update(x, 0)
    assert model.n_online == 20 and not np.allclose(model.weights, clean)
    model.reset_online()
    assert model.n_online == 0
    assert model.online_loss_history == []
    assert np.allclose(model.weights, clean), "reset must restore the batch weights"


def test_update_guards_return_empty():
    model, prices = _model()
    x = model.features_asof(prices[:500])
    assert model.online_update(None, 0) == {}        # no features
    assert model.online_update(x, 9) == {}           # bad target index
    model.trained = False
    assert model.online_update(x, 0) == {}            # untrained
    assert model.n_online == 0


def test_engine_live_reinforces_and_is_bounded():
    import tempfile
    from pathlib import Path

    import ml.candle_model as cm
    from core.engine import SimulationSession
    from exchange.live_client import ExchangeMode

    orig_dir = cm.MODELS_DIR
    cm.MODELS_DIR = Path(tempfile.mkdtemp())  # isolate persisted weights from real ones
    try:
        s = SimulationSession.create(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
        s.train_model(epochs=120, lr=0.5)
        assert s.model_status()["predict_mode"] is True and s.online_learn is True
        model = s.bot.candle_model

        # Walk the OOS tail; each settled, model-driven candle = 1 online step.
        for _ in range(60):
            s.tick()
            # No-leak parity: while a forecast is pending, its stashed features are
            # exactly what the model would compute from the current bar.
            pend = s.bot.live_eval.pending
            if pend is not None and pend.features is not None:
                assert np.allclose(pend.features, model.features_asof(s.bot.market.history))

        assert model.n_online > 0, "Live must reinforce the model online"
        assert model.n_online == s.bot.live_eval.n_eval, "one step per settled candle"
        assert model.online_drift() <= model.online_max_drift + 1e-9, "drift stays bounded"
        assert not np.allclose(model.weights, model.anchor_weights), "weights adapted"

        # Toggle off → snap back to the validated batch weights, freeze updates.
        s.set_online_learn(False)
        assert model.n_online == 0 and np.allclose(model.weights, model.anchor_weights)
        for _ in range(10):
            s.tick()
        assert model.n_online == 0, "no online updates once disabled"
        assert np.allclose(model.weights, model.anchor_weights), "weights stay frozen"
    finally:
        cm.MODELS_DIR = orig_dir


def test_reinforce_online_guards():
    """The engine guard must skip non-model-driven forecasts and missing features."""
    import tempfile
    from pathlib import Path

    import ml.candle_model as cm
    from core.engine import SimulationSession
    from exchange.live_client import ExchangeMode

    orig_dir = cm.MODELS_DIR
    cm.MODELS_DIR = Path(tempfile.mkdtemp())
    try:
        s = SimulationSession.create(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
        s.train_model(epochs=120, lr=0.5)
        model = s.bot.candle_model
        model.reset_online()
        settled = SettledPrediction(
            target_step=model.train_end_step + 1, target_ts=None,
            predicted_dir="up", realized_dir="down",
            predicted_ret=0.0, realized_ret=-0.01, correct=False,
        )
        x = model.features_asof(s.bot.market.history)
        s._reinforce_online(settled, None, True)        # no features → skip
        s._reinforce_online(settled, x, False)          # not model-driven → skip
        assert model.n_online == 0
        s._reinforce_online(settled, x, True)           # valid → one step
        assert model.n_online == 1
    finally:
        cm.MODELS_DIR = orig_dir


if __name__ == "__main__":
    test_online_update_reduces_loss_and_sets_anchor()
    test_trust_region_bounds_drift_under_adversarial_stream()
    test_features_asof_parity_and_guards()
    test_reset_restores_clean_weights()
    test_update_guards_return_empty()
    test_engine_live_reinforces_and_is_bounded()
    test_reinforce_online_guards()
    print("✓ online learning tests OK")
