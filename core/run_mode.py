"""The bot's three operating modes — a single *behaviour* axis.

This is the **primary** mode the user selects. It is orthogonal to the data
**source** (:class:`exchange.live_client.ExchangeMode`: cached history / Binance
paper / Binance live): the source says *where prices come from and whether
orders are real*, the run mode says *what the bot does with them*.

* ``TRAIN`` — load history, replay it with the risk-budget policy and train the
  supervised next-candle model (key ``g``). The replay PnL here is in-sample and
  is **not** a profitability claim.
* ``PAPER`` — answer *"rentable ou non ?"*: run the bot frozen (deterministic
  risk policy, fresh book), net of fees + slippage, and report a verdict. On
  cached history this is an out-of-sample backtest; on a live feed it is
  forward paper-trading.
* ``LIVE`` — apply the trained model to the **next candle** in real time:
  predict and self-evaluate, **without trading**.
"""
from __future__ import annotations

from enum import Enum


class RunMode(str, Enum):
    TRAIN = "train"
    PAPER = "paper"
    LIVE = "live"

    @property
    def label(self) -> str:
        """Human label for the status strip / metrics row."""
        return {
            "train": "Entraînement",
            "paper": "Paper (rentabilité)",
            "live": "Live (prédiction)",
        }[self.value]

    @property
    def icon(self) -> str:
        # Distinct from the SOURCE chip glyphs (📊/📝/🔴) on purpose — the two
        # axes must never look like the same control.
        return {"train": "🎓", "paper": "🧪", "live": "🔮"}[self.value]

    @property
    def tagline(self) -> str:
        return {
            "train": "charge l'historique et apprend",
            "paper": "rentable ou non — politique gelée, net de frais",
            "live": "prédit la prochaine bougie en temps réel",
        }[self.value]

    def next(self) -> "RunMode":
        order = (RunMode.TRAIN, RunMode.PAPER, RunMode.LIVE)
        return order[(order.index(self) + 1) % len(order)]


# Back-compat helper: the old boolean ``predict_mode`` meant "Live". Keep a
# single mapping so call sites that still think in that boolean stay correct.
def predict_mode_for(mode: RunMode) -> bool:
    return mode == RunMode.LIVE
