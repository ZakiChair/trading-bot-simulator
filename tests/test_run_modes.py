"""Tests — the three run modes (train | paper | live) and the paper verdict.

``run_mode`` is the single **behaviour** axis, orthogonal to the data **source**
(:class:`ExchangeMode`). TRAIN replays history with the risk-budget policy,
PAPER evaluates profitability on a **copy** of the bot (never perturbing the
home bot's book), LIVE predicts the next candle without trading. These pin that
separation and the verdict.

Run: ``BOT_UI_CONFIG=/tmp/x.json PYTHONPATH=. .venv/bin/python tests/test_run_modes.py``
"""
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import ml.candle_model as cm
from core.engine import SimulationSession
from core.run_mode import RunMode, predict_mode_for
from exchange.live_client import ExchangeMode

_VERDICTS = {
    "🔴 PAS RENTABLE",
    "🟠 rentable mais sous le buy & hold",
    "✅ RENTABLE (et bat le buy & hold)",
}


@contextlib.contextmanager
def _isolated_models():
    """Isolate persisted weights so the test never touches the real models/."""
    orig = cm.MODELS_DIR
    cm.MODELS_DIR = Path(tempfile.mkdtemp())
    try:
        yield
    finally:
        cm.MODELS_DIR = orig


def _fresh_session() -> SimulationSession:
    # Empty models dir → no saved model → boots into TRAIN deterministically.
    return SimulationSession.create(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)


def _run_paper_to_end(s: SimulationSession, cap: int = 5000) -> None:
    guard = 0
    while not s._paper_done and guard < cap:
        s.tick()
        guard += 1


def test_run_mode_cycle_and_predict_flag():
    assert RunMode.TRAIN.next() is RunMode.PAPER
    assert RunMode.PAPER.next() is RunMode.LIVE
    assert RunMode.LIVE.next() is RunMode.TRAIN
    assert predict_mode_for(RunMode.LIVE)
    assert not predict_mode_for(RunMode.TRAIN)
    assert not predict_mode_for(RunMode.PAPER)


def test_train_mode_advances_via_step_path():
    with _isolated_models():
        s = _fresh_session()
        assert s.run_mode is RunMode.TRAIN
        assert s.bot.predict_mode is False and s.bot.use_model is False
        step0 = s.bot.market.step
        for _ in range(20):
            s.tick()
        assert s.bot.market.step > step0
        assert s.bot.metrics.total_steps >= 20


def test_paper_emits_verdict_and_never_touches_home_book():
    with _isolated_models():
        s = _fresh_session()
        home_bot = s.bot
        home_risk = home_bot.risk
        trades0 = len(home_bot.portfolio.trades)
        cash0 = home_bot.portfolio.cash

        s.set_run_mode(RunMode.PAPER)
        assert s.run_mode is RunMode.PAPER
        assert s.bot is not home_bot, "paper runs on a COPY of the bot"
        assert s.bot.risk is not home_risk, "paper gets its own risk-policy copy"
        assert s.bot.risk.sigma_target_ann == home_risk.sigma_target_ann
        assert s.bot.portfolio.initial_cash == 10_000.0, "fresh book for the verdict"

        _run_paper_to_end(s)
        assert s._paper_done, "cached paper backtest must terminate with a verdict"
        v = s.paper_verdict
        assert v is not None and v.final and v.bars > 0
        assert v.verdict in _VERDICTS
        assert isinstance(v.profitable, bool)
        # buy&hold benchmark is wired (non-trivial over a real OOS window).
        assert v.buy_hold_pct != 0.0

        # Leaving paper restores the real bot, book completely untouched.
        s.set_run_mode(RunMode.TRAIN)
        assert s.bot is home_bot
        assert len(s.bot.portfolio.trades) == trades0, "paper polluted the home book!"
        assert s.bot.portfolio.cash == cash0


def test_paper_backtest_is_deterministic():
    # Deterministic risk policy + seeded engine ⇒ repeatable verdict (what the
    # measure harness relies on for before/after comparisons).
    with _isolated_models():
        runs = []
        for _ in range(2):
            s = _fresh_session()
            s.set_run_mode(RunMode.PAPER)
            _run_paper_to_end(s)
            runs.append((round(s.paper_verdict.ret_pct, 6), s.paper_verdict.n_trades))
        assert runs[0] == runs[1], f"paper backtest not deterministic: {runs}"


def test_live_mode_predicts_without_trading():
    with _isolated_models():
        s = _fresh_session()
        s.train_model(epochs=60, lr=0.5)  # trains + switches to LIVE
        assert s.run_mode is RunMode.LIVE
        assert s.bot.predict_mode is True and s.bot.use_model is True
        n_trades = len(s.bot.portfolio.trades)
        for _ in range(40):
            s.tick()
        assert len(s.bot.portfolio.trades) == n_trades, "Live must not trade"
        assert s.bot.live_eval.started, "Live must run a walk-forward eval"
        # La trace de risque reste visible en Live (le raisonnement s'affiche
        # même sans exécution).
        assert s.bot.risk.last_trace is not None


def test_cycle_skips_live_without_trained_model():
    with _isolated_models():
        s = _fresh_session()
        assert s.run_mode is RunMode.TRAIN
        assert s.cycle_run_mode() is RunMode.PAPER
        # No trained model → cannot go LIVE; cycle falls back to TRAIN.
        assert s.cycle_run_mode() is RunMode.TRAIN


if __name__ == "__main__":
    test_run_mode_cycle_and_predict_flag()
    test_train_mode_advances_via_step_path()
    test_paper_emits_verdict_and_never_touches_home_book()
    test_paper_backtest_is_deterministic()
    test_live_mode_predicts_without_trading()
    test_cycle_skips_live_without_trained_model()
    print("✓ run-mode tests OK")
