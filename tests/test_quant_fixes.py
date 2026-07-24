"""Regression tests for the quant audit fixes (RF1–RF5) — refonte 2026-07.

RF1 (l'apprentissage de la policy) a été retiré avec la policy elle-même : la
couche de décision est désormais la politique de budget de risque déterministe
(``core/decision.py``, testée dans ``tests/test_decision.py``). Le test RF1
vérifie maintenant que la décision est bien pilotée par les têtes VALIDÉES
(σ̂ rough-vol + régime HMM) et qu'elle est déterministe (pas d'ε-exploration).

Run: PYTHONPATH=. .venv/bin/python tests/test_quant_fixes.py
"""
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import numpy as np

import ml.candle_model as cm
from core.live_eval import LiveEvaluator, wilson_lower_bound
from core.next_candle import NextCandlePredictor
from core.thresholds import adaptive_threshold
from core.engine import SimulationSession


@contextlib.contextmanager
def _isolated_models():
    """Les tests qui entraînent NE DOIVENT PAS écraser models/*.npz de
    l'utilisateur (audit UX : lancer la suite remplaçait silencieusement le
    modèle réel par un artefact de test)."""
    orig = cm.MODELS_DIR
    cm.MODELS_DIR = Path(tempfile.mkdtemp())
    try:
        yield
    finally:
        cm.MODELS_DIR = orig


def test_decision_driven_by_validated_heads():
    """RF1 (refonte): la décision vient de σ̂/régime, elle est déterministe et
    sa trace est complète (l'UI peut montrer le pourquoi)."""
    s = SimulationSession.create(n_scenarios=150)
    from core.run_mode import RunMode
    s.set_run_mode(RunMode.TRAIN)
    for _ in range(30):
        s.tick()
    tr = s.bot.risk.last_trace
    assert tr is not None
    assert tr.sigma_bar > 0 and tr.sigma_ann > 0
    assert tr.regime in ("calm", "normal", "turbulent")
    assert 0.0 <= tr.target_exposure <= 1.0
    assert tr.reason, "chaque décision doit expliquer son pourquoi"
    # Pas d'edge Wilson en replay → le tilt directionnel doit rester fermé.
    assert tr.dir_tilt == 0.0
    print("✓ RF1 (refonte) décision pilotée par σ̂ rough-vol + régime HMM, tilt fermé")


def test_adaptive_threshold_scales_with_vol():
    """RF4: calm series → small band, noisy series → larger band, both clamped."""
    rng = np.random.default_rng(0)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, 500)))
    noisy = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500)))
    t_calm = adaptive_threshold(calm)
    t_noisy = adaptive_threshold(noisy)
    assert t_noisy > t_calm, (t_calm, t_noisy)
    assert 3e-4 <= t_calm <= 5e-2 and 3e-4 <= t_noisy <= 5e-2
    print(f"✓ RF4 adaptive threshold scales (calm {t_calm:.4f} < noisy {t_noisy:.4f})")


def test_backtest_walk_forward_not_circular():
    """RF2: hit rate is computed online (no look-ahead) and stays plausible."""
    pred = NextCandlePredictor(seed=1)
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, 300)
    hit, n = pred._backtest_hit_rate(returns, 0.003)
    # On i.i.d. noise a 1st-order Markov predictor has no real edge → ~ chance,
    # certainly not the inflated in-sample figure the old code produced.
    assert 0.2 <= hit <= 0.75, hit
    assert n == len(returns) - 1
    print(f"✓ RF2 walk-forward backtest honest ({hit:.0%} over {n} bars)")


def test_wilson_significance_gate():
    """RF5: a lucky small sample is NOT graded 'fiable'; a real large edge is."""
    assert wilson_lower_bound(7, 10) < 0.5  # 70% of 10 is not significant
    assert wilson_lower_bound(70, 100) > 0.5

    ev = LiveEvaluator(); ev.reset()
    ev.n_eval, ev.n_correct = 10, 8
    ev.realized_counts = {"up": 8, "flat": 1, "down": 1}
    label, _style, needs = ev.grade()
    assert needs and "insuffisant" in label, label  # small sample → no verdict

    ev.n_eval, ev.n_correct = 80, 60
    ev.realized_counts = {"up": 40, "flat": 20, "down": 20}  # baseline 50%
    # Depuis la revue 2026-07-24 la porte/l'étiquette « edge directionnel » ne
    # se jugent que sur les appels DIRECTIONNELS (le skill global peut venir de
    # la seule fréquence des flats = timing de vol, sans signe).
    ev.n_dir, ev.n_dir_correct = 80, 60
    ev.dir_realized_counts = {"up": 40, "flat": 0, "down": 40}  # majorité 50 %
    label, _style, needs = ev.grade()
    assert not needs and "fiable" in label, label  # real significant edge
    # Le même hit global SANS appels directionnels ne doit plus suffire.
    ev.n_dir = ev.n_dir_correct = 0
    ev.dir_realized_counts = {"up": 0, "flat": 0, "down": 0}
    assert ev.significant_edge is False
    print("✓ RF5 Wilson significance gate works (directional calls only)")


def test_strict_oos_start_uses_train_end():
    """RF3: live walk starts strictly after the model's trained bars."""
    with _isolated_models():
        s = SimulationSession.create(n_scenarios=120)
        rep = s.train_model(epochs=60, lr=0.5)
        model = s.bot.candle_model
        assert model is not None and model.train_end_step >= 0
        # toggle predict mode off→on to (re)begin a fresh live session
        s.set_predict_mode(False)
        s.set_predict_mode(True)
        sim = s.bot_engine.simulator
        if model.train_end_ts > 0 and sim.timestamps is not None:
            # Entraînement sur historique profond : l'index d'entraînement ne se
            # transpose pas dans la fenêtre de session — l'invariant OOS se
            # vérifie en TEMPS (première barre strictement après la zone touchée).
            assert int(sim.timestamps[s._eval_start]) > model.train_end_ts, (
                s._eval_start, model.train_end_ts,
            )
            print(f"✓ RF3 live walk starts OOS (ts[{s._eval_start}] > train_end_ts)")
        else:
            assert s._eval_start > model.train_end_step, (
                s._eval_start, model.train_end_step,
            )
            print(f"✓ RF3 live walk starts OOS (start {s._eval_start} > train_end {model.train_end_step})")


if __name__ == "__main__":
    test_decision_driven_by_validated_heads()
    test_adaptive_threshold_scales_with_vol()
    test_backtest_walk_forward_not_circular()
    test_wilson_significance_gate()
    test_strict_oos_start_uses_train_end()
    print("✓ quant fixes tests OK")
