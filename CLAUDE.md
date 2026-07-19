# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A terminal (Textual TUI) simulator of a probabilistic trading bot. At each market
bar it generates 100–10,000 Monte Carlo scenarios (rough-vol Student-t cone,
martingale-centred + HMM regime + optional Hawkes jumps), reads its conformal
intervals, and a **deterministic risk-budget policy** (`core/decision.py`)
decides the exposure: `e* = clip(σ_target/σ̂, 0, 1) × (1 − 0.5·P(turbulent)) ×
(1 + tilt·1[Wilson edge])`. Real BTC/ETH/SOL data comes from Binance via `ccxt`,
cached in SQLite (freshness-checked, forming candle excluded).

**Read `README.md` first** — it documents the feature set, keybindings, and
module map (mostly in French). `RAPPORT_REFONTE_V2.md` documents the 2026-07
overhaul (diagnosis → fixes → measured validation); `RAPPORT_REFONTE.md` and
`ANALYSE_CRITIQUE_MODELE.md` document the earlier quantitative research — read
them before touching `ml/rough_vol.py`, `core/hmm.py`, `core/conformal.py`,
`core/hawkes.py` or `core/decision.py`.

**Critical honesty constraint**: this project's own research concluded that
1-bar-ahead direction on real crypto data is ≈50/50 — no model here fabricates
directional edge. The real, measured signals are magnitude (rough-vol), regime
(HMM) and calibration (conformal); the bot acts on those. The directional tilt
in the risk policy only opens behind a Wilson significance gate (structurally
almost never). Don't reintroduce claims of directional edge or inflated backtest
numbers; every new model must beat its baseline in `tests/measure_model.py`
(walk-forward, no leak) or be reported honestly and left OFF by default.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the interactive TUI. Boots in TRAIN; the default DATA SOURCE is the
# persisted preference (paper Binance feed out of the box — real prices, no
# orders), falling back to cached/simulated history when the network is down.
python main.py
python main.py --asset ETH/USDT
python main.py --paper          # data source = live Binance prices (paper fills)

# Headless walk-forward replay of the risk policy (deterministic, cached data)
python main.py --cli --episodes 5 --steps 400

# Train the supervised next-candle model (first 85% of history; the last 15%
# stays untouched for the Live/Paper out-of-sample walk)
python main.py --train --asset BTC/USDT --timeframe 1h --epochs 400

# Tests — no pytest in the venv; every tests/*.py file is a standalone script.
PYTHONPATH=. .venv/bin/python tests/smoke.py
PYTHONPATH=. .venv/bin/python tests/test_decision.py    # risk policy + portfolio v2
PYTHONPATH=. .venv/bin/python tests/measure_model.py BTC/USDT 1h  # flagship harness
PYTHONPATH=. .venv/bin/python tests/data_checks.py      # Hurst + regime stability
```

There is no linter/formatter config in this repo — don't assume one exists.
UI tests must set `BOT_UI_CONFIG=/tmp/…json` (they do) so key presses never
clobber `config/ui_settings.json`. Tests that train models must wrap in the
`_isolated_models()` pattern **or** set `BOT_MODELS_DIR=/tmp/…` (env override
honoured by `ml/candle_model.py::models_dir`) so they never overwrite the
user's `models/*.npz` — a forgotten isolation has already replaced the real
BTC model with a val-100% toy twice.

## Architecture

### Two independent axes: run mode vs. exchange mode

- **`RunMode`** (`core/run_mode.py`) — *what the bot does*: `TRAIN` (replay
  history with the risk policy; train the candle model with `g`), `PAPER`
  (frozen deterministic bot, fresh book, net of fees+slippage — verdict carries
  a Student-t significance qualifier), `LIVE` (predict the next candle, Wilson
  self-evaluation, **no trading**).
- **`ExchangeMode`** (`exchange/live_client.py`) — *where prices come from*.
  Real orders are double-guarded: a LIVE client is never attached to the
  portfolio (`core/engine.py::_attach_exchange`), and `place_market_order`
  additionally requires `BINANCE_ALLOW_REAL_ORDERS=1`. No run mode places real
  orders.

The app boots into TRAIN even when a saved model exists (a hint in the journal
says `m` switches to LIVE) — auto-booting into the do-nothing LIVE replay was
the worst first impression (UX audit).

### Pipeline (core/, ml/)

```
data/loader.py          cache→ccxt→synthetic; staleness check; forming candle dropped
core/market.py          market state, regime heuristic (legacy display only)
ml/rough_vol.py         rough vol (RFSV); per-horizon ŝ²ₖ measured at fit
ml/vol_model.py         range estimators, GARCH/HAR — shoot-out contenders
core/cone.py            Student-t martingale cone ladder (A/B-testable rungs)
core/hmm.py             3-state vol HMM; regime_probs() = causal filtered dist
core/conformal.py       split conformal + ACI; update() scores the EMITTED interval
core/hawkes.py          self-exciting jumps (OFF by default; σ̂-relative sizes)
core/scenarios.py       assembles the cone; terminal-scale likelihood; no Markov sampling
core/next_candle.py     next-candle forecast; headline masses = calibrated source
ml/candle_model.py      calibrated softmax candle model; 15% OOS tail; anchored online SGD
core/decision.py        RiskPolicy + DecisionTrace — THE decision layer
core/portfolio.py       fractional rebalancing, fees + σ̂-scaled slippage, MtM equity
core/bot.py             TradingBot.step: generate → decide → rebalance; RunMetrics
core/engine.py          SimulationSession — run modes, Live/Paper sessions, snapshots
core/live_eval.py       Wilson eval + reformed reliability note (coverage/calibration/edge)
core/paper_eval.py      qualified profitability verdict (t-stat, minimum bars/trades)
ui/app.py               Textual app; heavy widgets refresh only on their active tab
ui/panels.py            decision panel, pipeline, honesty, metrics, bubbles…
```

`SimulationSession` (`core/engine.py`) is the entry point every runner
constructs first. `TradingBot.step` is the per-bar loop: scenario bundle →
`RiskPolicy.target_exposure`/`decide` → `Portfolio.rebalance`.

### Known gotchas (see project memory for fuller writeups)

- Real market log-realized-vol is measured rough (Hurst H≈0.06–0.08, **but
  downward-biased by RV-proxy noise** — treat as a bound, not a constant);
  don't swap the σ head without re-running `tests/measure_model.py`.
- The scenario likelihood must stay at the TERMINAL scale (Σ var_path): scoring
  h-bar returns with 1-bar σ² collapses the softmax onto flat paths (old bug).
- The sizing σ̂ and P(turbulent) are EWMA-smoothed inside `RiskPolicy` — feeding
  the raw jumpy heads directly re-creates the churn (90 trades/400 bars) the
  smoothing was measured to fix. `RiskPolicy` carries that state; call
  `reset_state()` when re-anchoring a session.
- The flat/neutral band is quantile-based (`core/thresholds.py`) with per-bar
  local labels — a fixed `k·σ` band collapses the model to ~85% NEUTRE.
- `LiveMarketFeed.advance` must build states from the SIM's fresh arrays and
  return None at the head (no new closed candle) — the engine waits instead of
  re-processing the same bar.
- `plotext` figures with subplots: address the specific subplot or you paint
  into every pane.
- `pandas`/`ccxt` ms timestamps: use `.as_unit("ms").astype(int64)`
  (`core/timeutil.py`); a plain `astype(int64)//10**6` lands in 1970.
- Empirical-superiority claims pinned as hard test asserts must use the
  harness-sized window + tolerance (a 70-bar window flipped with one cache
  refresh — see `tests/test_vol_model.py`).
