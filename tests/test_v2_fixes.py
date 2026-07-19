"""Non-régression des correctifs « aspects non fonctionnels » (2026-07-02).

Chaque test verrouille un défaut reproduit par l'audit runtime post-V2 :
  1.  clip du RATIO avant les portes (la turbulence mord aussi en vol basse)
      + la formule affichée reproduit exactement e* ;
  2.  reset de l'état EWMA de RiskPolicy à chaque ré-ancrage (switch actif/
      timeframe, entrée paper) ;
  3.  le bot paper reçoit le modèle SUR SES POIDS D'ANCRE (pas la dérive du
      renforcement en ligne) et une policy à l'état vierge ;
  4.  step_live_forward exécute au close de la bougie FRAÎCHEMENT clôturée
      (plus de fill rétroactif au close vieux d'une barre) ;
  5.  predict_only (mode LIVE) rend une trace explicitement hypothétique ;
  6.  la prédiction scorée par LiveEvaluator = l'argmax des masses CALIBRÉES
      (la même lecture que le bandeau consensus et que le Brier) ;
  7.  UISettings.load champ par champ (un champ corrompu ne détruit plus tout,
      tick_speed négatif refusé) + timeframe persisté ;
  8.  migration du cache SQLite : les timestamps en SECONDES (bug « 1970 »)
      sont convertis en ms puis purgés, et refusés à l'écriture ;
  9.  l'historique d'equity est segmenté par époque de bot (pas de splice
      TRAIN/PAPER) ;
  10. l'avertissement « Live ignoré » de cycle_run_mode atterrit dans le
      journal du bot RESTAURÉ (pas celui du bot paper jeté) ;
  11. le HMM de régime reçoit les opens → real_ohlc peut authentifier le range
      (Parkinson) au lieu de retomber en silence sur r² ;
  12. le verdict paper voyage toujours avec son qualificatif de
      significativité (sig_tag) ;
  13. un .npz au schéma périmé/corrompu est signalé, plus ignoré en silence ;
  14. CLI : --timeframe transmis (--cli), --timeframe invalide → erreur claire,
      --train sur données synthétiques → refus (exit 2) ;
  15. touches TUI : s avance d'un pas PENDANT la pause, r conserve
      actif/timeframe/source.

Run:  PYTHONPATH=. .venv/bin/python tests/test_v2_fixes.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("BOT_UI_CONFIG", tempfile.mktemp(suffix="_v2fix_ui.json"))
os.environ.setdefault("BOT_MODELS_DIR", tempfile.mkdtemp(prefix="bot-models-v2fix-"))

from core.decision import RiskPolicy  # noqa: E402
from core.engine import SimulationSession  # noqa: E402
from core.run_mode import RunMode  # noqa: E402
from exchange.live_client import ExchangeMode  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent


def _fresh_session(n_scenarios: int = 100) -> SimulationSession:
    return SimulationSession.create(
        n_scenarios=n_scenarios, exchange_mode=ExchangeMode.SIMULATION
    )


# --------------------------------------------------------------------------- #
# 1. clip du ratio + formule fidèle                                            #
# --------------------------------------------------------------------------- #
def test_regime_gate_bites_in_low_vol() -> None:
    pol = RiskPolicy(sigma_smooth=0.0)  # pas de lissage : test direct
    # σ̂ minuscule → ratio >> 1. Turbulence certaine → la porte DOIT diviser
    # par 2 l'exposition effectivement prise (l'ancien clip-du-produit rendait
    # e* = 100 % : porte neutralisée).
    tr = pol.target_exposure(sigma_bar=1e-5, timeframe="1h", p_turbulent=1.0)
    assert abs(tr.target_exposure - 0.5) < 1e-9, tr.target_exposure
    # La formule instanciée reproduit le calcul terme à terme.
    recomputed = min(tr.raw_exposure, 1.0) * tr.regime_mult * (1.0 + tr.dir_tilt)
    assert abs(min(recomputed, 1.0) - tr.target_exposure) < 1e-9
    assert f"{tr.target_exposure:.0%}" in tr.formula_line()
    print("✓ porte de régime opérante en vol basse + formule fidèle")


# --------------------------------------------------------------------------- #
# 2. reset EWMA aux ré-ancrages                                                #
# --------------------------------------------------------------------------- #
def test_ewma_state_reset_on_reanchor() -> None:
    s = _fresh_session()
    for _ in range(3):
        s.tick()
    assert s.bot.risk._sigma_state is not None, "l'EWMA doit s'amorcer en TRAIN"
    s.switch_asset("ETH/USDT")
    assert s.bot.risk._sigma_state is None, "switch actif doit reset l'état EWMA"
    for _ in range(3):
        s.tick()
    assert s.bot.risk._sigma_state is not None
    s.switch_timeframe("4h")
    assert s.bot.risk._sigma_state is None, "switch timeframe doit reset l'état EWMA"
    print("✓ état EWMA remis à zéro aux ré-ancrages (actif / timeframe)")


# --------------------------------------------------------------------------- #
# 3. paper : modèle ancré + policy vierge                                      #
# --------------------------------------------------------------------------- #
def test_paper_bot_gets_anchored_model_and_clean_state() -> None:
    s = _fresh_session()
    s.train_model(epochs=30, lr=0.5)  # persiste dans BOT_MODELS_DIR isolé
    home_model = s.bot.candle_model
    assert home_model is not None and home_model.trained
    home_model.ensure_anchor()
    rng = np.random.default_rng(0)
    for _ in range(40):  # dérive online artificielle (comme une marche LIVE)
        home_model.online_update(rng.normal(size=home_model.weights.shape[0]), 1)
    assert home_model.n_online == 40
    assert not np.allclose(home_model.weights, home_model.anchor_weights)

    for _ in range(3):
        s.tick()  # amorce l'EWMA du bot maison
    s.set_run_mode(RunMode.PAPER)
    pm = s.bot.candle_model
    assert pm is not None and pm is not home_model, "paper doit recevoir une copie"
    assert np.allclose(pm.weights, home_model.anchor_weights), (
        "le bot paper doit trader les poids d'ANCRE (batch validés)"
    )
    assert pm.n_online == 0
    assert not np.allclose(home_model.weights, home_model.anchor_weights), (
        "la dérive du bot maison doit survivre à l'aller-retour paper"
    )
    assert s.bot.risk._sigma_state is None or s.bot.risk._sigma_state != \
        s._home_bot.risk._sigma_state, "la policy paper ne doit pas hériter l'état EWMA"
    s.set_run_mode(RunMode.TRAIN)
    assert s.bot.candle_model is home_model
    print("✓ paper = modèle ancré (sans dérive online) + policy à l'état vierge")


# --------------------------------------------------------------------------- #
# 4. step_live_forward : fill au close frais                                   #
# --------------------------------------------------------------------------- #
def test_step_live_forward_fills_at_fresh_close() -> None:
    from core.bot import TradingBot
    from core.live_feed import LiveMarketFeed
    from tests.test_live_feed import _FakeClient

    client = _FakeClient(n=160)
    feed = LiveMarketFeed(symbol="BTC/USDT", timeframe="1h", client=client)
    feed.refresh_interval_s = 0.0
    engine = TradingBot(simulator=feed, n_scenarios=100)
    bot = engine.new_episode(episode=1, market_sim=feed)
    bot.market = feed.state_at(feed.n_bars - 1)  # ancré en tête de feed

    assert engine.step_live_forward(bot) is False, "tête de feed → attendre"
    assert not bot.portfolio.trades, "aucun fill ne doit arriver en attente"

    stale_close = float(feed.prices[-1])
    client.advance_one_candle()  # une bougie clôture
    assert engine.step_live_forward(bot) is True
    assert bot.market.step == feed.n_bars - 1, "le pas doit avancer sur la bougie close"
    fresh_close = float(feed.prices[-1])
    assert bot.portfolio.trades, "première décision → rebalancement attendu"
    px = bot.portfolio.trades[-1].price
    assert abs(px - fresh_close) / fresh_close < 0.002, (
        f"fill {px} doit coller au close FRAIS {fresh_close}"
    )
    assert abs(px - stale_close) / stale_close > 0.003, (
        f"fill {px} ne doit PAS être au close périmé {stale_close}"
    )
    print("✓ marche avant live : exécution au close fraîchement clôturé")


# --------------------------------------------------------------------------- #
# 5. predict_only : trace hypothétique                                         #
# --------------------------------------------------------------------------- #
def test_predict_only_trace_is_hypothetical() -> None:
    s = _fresh_session()
    s.bot_engine.predict_only(s.bot)
    tr = s.bot.risk.last_trace
    assert tr is not None
    assert tr.rebalanced is False, "Live ne trade jamais — la trace non plus"
    assert tr.reason.startswith("Live (aucun trade)"), tr.reason
    print("✓ trace de décision LIVE marquée hypothétique")


# --------------------------------------------------------------------------- #
# 6. la prédiction scorée = argmax des masses calibrées                        #
# --------------------------------------------------------------------------- #
def test_live_eval_scores_calibrated_masses() -> None:
    from core.live_eval import LiveEvaluator
    from core.next_candle import CandleBubble, NextCandleForecast

    bubble_up = CandleBubble(id=0, direction="up", return_pct=0.01,
                             probability=0.30, predicted_price=101.0, source="blend")
    fc = NextCandleForecast(
        timeframe="1h", current_price=100.0, bubbles=[bubble_up],
        most_probable=bubble_up,          # la bulle ★ dit HAUSSE…
        prob_up=0.25, prob_down=0.15, prob_flat=0.60,  # …les masses disent NEUTRE
        expected_return=0.0004, markov_from="flat", markov_to="flat",
        markov_transition_prob=0.4, backtest_hit_rate=0.5, gbm_drift=0.0,
        gbm_vol=0.005, n_backtest_bars=100, model_driven=True,
        target_step=10, target_ts=None, base_step=9,
    )
    ev = LiveEvaluator()
    ev.reset()
    pend = ev.open_from_forecast(fc)
    assert pend.direction == "flat", "scorer l'argmax des masses calibrées, pas la bulle ★"
    assert abs(pend.probability - 0.60) < 1e-9
    assert abs(pend.return_pct - 0.0004) < 1e-12
    print("✓ LiveEvaluator score la même lecture que le consensus calibré")


# --------------------------------------------------------------------------- #
# 7. UISettings robustes + timeframe persisté                                  #
# --------------------------------------------------------------------------- #
def test_ui_settings_per_field_and_timeframe() -> None:
    from config.ui_settings import UISettings

    path = Path(tempfile.mktemp(suffix="_uisettings.json"))
    old = os.environ.get("BOT_UI_CONFIG")
    os.environ["BOT_UI_CONFIG"] = str(path)
    try:
        path.write_text(json.dumps({
            "theme": "cute",
            "asset": "ETH/USDT",
            "timeframe": "4h",
            "tick_speed": "vite",          # corrompu → défaut, SANS tout effacer
            "boogie_size_idx": 2,
        }))
        ui = UISettings.load()
        assert ui.theme == "cute" and ui.asset == "ETH/USDT"
        assert ui.timeframe == "4h", "timeframe doit être persisté/chargé"
        assert ui.tick_speed == 0.35, "champ corrompu → défaut de CE champ"
        assert ui.boogie_size_idx == 2

        path.write_text(json.dumps({"tick_speed": -3.0, "timeframe": "banana"}))
        ui = UISettings.load()
        assert ui.tick_speed == 0.35, "tick_speed négatif → refusé (app figée sinon)"
        assert ui.timeframe == "1h", "timeframe inconnu → défaut"

        ui.timeframe = "15m"
        ui.save()
        assert UISettings.load().timeframe == "15m", "round-trip save→load"
    finally:
        if old is not None:
            os.environ["BOT_UI_CONFIG"] = old
        path.unlink(missing_ok=True)
    print("✓ UISettings champ-par-champ + timeframe persisté")


# --------------------------------------------------------------------------- #
# 8. migration cache « 1970 »                                                  #
# --------------------------------------------------------------------------- #
def test_loader_migrates_seconds_timestamps() -> None:
    from data.loader import DataLoader

    db = Path(tempfile.mktemp(suffix="_cache.sqlite"))
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE ohlcv (
                exchange TEXT, symbol TEXT, timeframe TEXT,
                ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (exchange, symbol, timeframe, ts))"""
        )
        conn.execute("INSERT INTO ohlcv VALUES ('binance','BTC/USDT','4h',1688227200,1,2,0.5,1.5,10)")
        conn.execute("INSERT INTO ohlcv VALUES ('binance','BTC/USDT','4h',1688241600000,1,2,0.5,1.6,11)")
    DataLoader(db_path=db)  # _init_db → migration
    with sqlite3.connect(db) as conn:
        bad = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE ts < 1000000000000").fetchone()[0]
        migrated = conn.execute(
            "SELECT close FROM ohlcv WHERE ts = 1688227200000"
        ).fetchone()
    assert bad == 0, "plus aucune ligne en secondes (dates 1970)"
    assert migrated is not None and abs(migrated[0] - 1.5) < 1e-9, "ligne convertie en ms"
    db.unlink(missing_ok=True)
    print("✓ cache : timestamps en secondes migrés en ms puis purgés")


# --------------------------------------------------------------------------- #
# 9. segmentation d'equity par époque                                          #
# --------------------------------------------------------------------------- #
def test_equity_history_is_segmented_by_bot_epoch() -> None:
    s = _fresh_session()
    for _ in range(3):
        s.tick()
    e_train = s.history[-1]["epoch"]
    s.set_run_mode(RunMode.PAPER)
    s.tick()
    e_paper = s.history[-1]["epoch"]
    assert e_paper > e_train, "le bot paper ouvre un nouveau segment d'equity"
    s.set_run_mode(RunMode.TRAIN)
    s.tick()
    e_back = s.history[-1]["epoch"]
    assert e_back > e_paper, "le retour TRAIN ouvre encore un segment"
    print("✓ historique d'equity segmenté par époque de bot (pas de splice)")


# --------------------------------------------------------------------------- #
# 10. avertissement « Live ignoré » visible                                    #
# --------------------------------------------------------------------------- #
def test_cycle_skip_live_warning_lands_in_restored_journal() -> None:
    s = _fresh_session()
    assert s.bot.candle_model is None or not s.bot.candle_model.trained or True
    s.bot.candle_model = None  # aucun modèle → LIVE doit être sauté
    s.set_run_mode(RunMode.PAPER)
    mode = s.cycle_run_mode()  # PAPER → (LIVE sauté) → TRAIN
    assert mode == RunMode.TRAIN
    assert any("Live ignoré" in m for m in s.bot.log), (
        "l'explication doit être dans le journal du bot AFFICHÉ"
    )
    print("✓ avertissement « Live ignoré » visible après PAPER→TRAIN")


# --------------------------------------------------------------------------- #
# 11. HMM : les opens atteignent real_ohlc (Parkinson possible)                #
# --------------------------------------------------------------------------- #
def test_hmm_observations_use_parkinson_with_real_ohlc() -> None:
    from core.hmm import _vol_observations
    from ml.vol_model import real_ohlc

    rng = np.random.default_rng(3)
    n = 300
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    body = np.abs(closes - opens)
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.004, n))) + 0.2 * body
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.004, n))) - 0.2 * body
    assert real_ohlc(opens, highs, lows, closes), "harnais : l'OHLC doit être 'réel'"

    with_range = _vol_observations(closes, highs, lows, opens=opens)
    without = _vol_observations(closes, highs, lows)  # ancien appel (opens absents)
    assert not np.allclose(with_range, without), (
        "avec opens, l'observation doit venir de Parkinson (≠ proxy r²)"
    )
    print("✓ HMM : opens threadés → Parkinson au lieu du repli r² silencieux")


# --------------------------------------------------------------------------- #
# 12. verdict paper toujours qualifié                                          #
# --------------------------------------------------------------------------- #
def test_paper_verdict_carries_significance() -> None:
    from core.paper_eval import PaperVerdict
    from ui.panels import format_status_strip

    v = PaperVerdict(ret_pct=2.0, buy_hold_pct=1.0, n_trades=8, n_round=4,
                     win_rate=0.5, turnover=0.03, bars=250, forward=False,
                     final=True, t_stat=0.8)
    assert v.sig_tag and "n.s." in v.sig_tag, v.sig_tag

    s = _fresh_session()
    s.tick()
    s.bot.mode = "paper"
    strip = format_status_strip(s.bot, auto=True, styles=None, paper_verdict=v)
    assert "t=+0.8" in strip.plain, strip.plain
    print("✓ verdict paper affiché avec sa significativité (bandeau inclus)")


# --------------------------------------------------------------------------- #
# 13. .npz périmé signalé                                                      #
# --------------------------------------------------------------------------- #
def test_stale_model_is_reported_not_silent() -> None:
    from ml.candle_model import load_model_with_note, model_path

    p = model_path("ZZZ/USDT", "1h")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, weights=np.zeros((2, 3)), meta=np.array(["ZZZ/USDT", "1h"]))
    model, note = load_model_with_note("ZZZ/USDT", "1h")
    assert model is None and note, "un artefact illisible doit produire une note"
    assert "ZZZ_USDT_1h" in note
    p.unlink(missing_ok=True)
    print("✓ modèle .npz périmé/corrompu signalé au journal (plus de silence)")


# --------------------------------------------------------------------------- #
# 14. CLI : timeframe transmis, erreurs claires, refus synthétique             #
# --------------------------------------------------------------------------- #
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["BOT_MODELS_DIR"] = tempfile.mkdtemp(prefix="bot-models-cli-")
    return subprocess.run(
        [sys.executable, "main.py", *args],
        capture_output=True, text=True, cwd=_REPO, env=env, timeout=240,
    )


def test_cli_paths() -> None:
    r = _run_cli("--cli", "--episodes", "1", "--steps", "30", "--timeframe", "4h")
    assert r.returncode == 0, r.stderr[-800:]
    assert "4h" in r.stdout, "le timeframe demandé doit apparaître (et être utilisé)"

    r = _run_cli("--cli", "--timeframe", "banana", "--episodes", "1", "--steps", "5")
    assert r.returncode == 2, (r.returncode, r.stderr[-300:])
    assert "timeframe invalide" in (r.stderr + r.stdout)

    r = _run_cli("--train", "--asset", "FOO/BAR", "--epochs", "10")
    assert r.returncode == 2, "entraîner sur du synthétique doit être refusé"
    out = r.stdout + r.stderr
    assert "SYNTH" in out.upper(), "le refus doit nommer les données synthétiques"
    print("✓ CLI : --timeframe honoré, erreurs claires, --train synthétique refusé")


# --------------------------------------------------------------------------- #
# 14b. échec de source (x → LIVE sans clés) : état de mode cohérent            #
# --------------------------------------------------------------------------- #
def test_exchange_mode_unchanged_when_connection_fails() -> None:
    assert not os.environ.get("BINANCE_API_KEY"), "test suppose l'absence de clés"
    s = _fresh_session()
    prev = s.exchange_mode
    try:
        s.set_exchange_mode(ExchangeMode.LIVE)  # sans clés API → doit lever
        raised = False
    except Exception:
        raised = True
    assert raised, "LIVE sans clés doit échouer"
    assert s.exchange_mode == prev, (
        "l'échec ne doit PAS muter exchange_mode (badge/cycle x désynchronisés)"
    )
    print("✓ échec de connexion LIVE → session cohérente sur son mode courant")


# --------------------------------------------------------------------------- #
# 15. touches TUI : s pendant la pause, r conserve la config                   #
# --------------------------------------------------------------------------- #
async def _tui_keys_check() -> None:
    from ui.app import BotSimulatorApp

    app = BotSimulatorApp(n_scenarios=100, exchange_mode=ExchangeMode.SIMULATION)
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.pause()
        app.session.auto_train = False
        app.session.bot.paused = True
        step0 = app.session.bot.market.step
        app.action_step_once()
        assert app.session.bot.market.step == step0 + 1, (
            "s doit avancer d'une barre même en pause"
        )
        assert app.session.bot.paused is True, "s ne doit pas lever la pause"

        app.session.switch_asset("ETH/USDT")
        app.session.switch_timeframe("4h")
        app.action_reset()
        assert app.session.asset == "ETH/USDT", "r doit conserver l'actif"
        assert app.session.timeframe == "4h", "r doit conserver le timeframe"


def test_tui_step_and_reset_keys() -> None:
    asyncio.run(_tui_keys_check())
    print("✓ touches : s avance en pause · r conserve actif/timeframe")


if __name__ == "__main__":
    test_regime_gate_bites_in_low_vol()
    test_ewma_state_reset_on_reanchor()
    test_paper_bot_gets_anchored_model_and_clean_state()
    test_step_live_forward_fills_at_fresh_close()
    test_predict_only_trace_is_hypothetical()
    test_live_eval_scores_calibrated_masses()
    test_ui_settings_per_field_and_timeframe()
    test_loader_migrates_seconds_timestamps()
    test_equity_history_is_segmented_by_bot_epoch()
    test_cycle_skip_live_warning_lands_in_restored_journal()
    test_hmm_observations_use_parkinson_with_real_ohlc()
    test_paper_verdict_carries_significance()
    test_stale_model_is_reported_not_silent()
    test_cli_paths()
    test_exchange_mode_unchanged_when_connection_fails()
    test_tui_step_and_reset_keys()
    print("✓ v2-fixes regression tests OK")
