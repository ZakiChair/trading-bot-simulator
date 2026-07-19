"""Binance connectivity via ccxt — paper trading and optional live orders."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from config.market_config import DEFAULT_EXCHANGE
from data.loader import DataLoader, timeframe_minutes


class ExchangeMode(str, Enum):
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    cost: float
    fee: float
    status: str
    mode: ExchangeMode
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeClient:
    """Unified client for market data and order execution."""

    exchange_id: str = DEFAULT_EXCHANGE
    mode: ExchangeMode = ExchangeMode.PAPER
    _exchange: Any = field(default=None, repr=False)
    _paper_orders: list[OrderResult] = field(default_factory=list)
    _order_counter: int = 0

    def __post_init__(self) -> None:
        if self.mode == ExchangeMode.SIMULATION:
            return
        self._init_exchange()

    def _init_exchange(self) -> None:
        import ccxt

        opts: dict[str, Any] = {"enableRateLimit": True, "timeout": 20000}
        if self.mode == ExchangeMode.LIVE:
            key = os.environ.get("BINANCE_API_KEY", "")
            secret = os.environ.get("BINANCE_API_SECRET", "")
            if not key or not secret:
                raise RuntimeError(
                    "Mode LIVE requiert BINANCE_API_KEY et BINANCE_API_SECRET"
                )
            opts["apiKey"] = key
            opts["secret"] = secret

        klass = getattr(ccxt, self.exchange_id)
        self._exchange = klass(opts)

    @property
    def is_connected(self) -> bool:
        return self._exchange is not None or self.mode == ExchangeMode.SIMULATION

    def ping(self) -> dict[str, Any]:
        if self.mode == ExchangeMode.SIMULATION:
            return {"status": "simulation", "exchange": self.exchange_id}
        assert self._exchange is not None
        self._exchange.load_markets()
        ticker = self._exchange.fetch_ticker("BTC/USDT")
        return {
            "status": "ok",
            "exchange": self.exchange_id,
            "mode": self.mode.value,
            "btc_price": ticker.get("last"),
        }

    def now_ms(self) -> int:
        """Current UTC time in ms — exchange clock when connected, else local."""
        if self._exchange is not None:
            return int(self._exchange.milliseconds())
        return int(time.time() * 1000)

    def fetch_ticker(self, symbol: str) -> float:
        if self.mode == ExchangeMode.SIMULATION:
            loader = DataLoader(self.exchange_id)
            df, _ = loader.load(symbol, "1h", 5)
            return float(df["close"].iloc[-1])
        assert self._exchange is not None
        t = self._exchange.fetch_ticker(symbol)
        return float(t["last"])

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 200
    ) -> pd.DataFrame:
        if self.mode == ExchangeMode.SIMULATION:
            loader = DataLoader(self.exchange_id)
            df, _ = loader.load(symbol, timeframe, limit)
            return df

        assert self._exchange is not None
        tf_ms = timeframe_minutes(timeframe) * 60_000
        since = self._exchange.milliseconds() - limit * tf_ms
        collected: list = []
        while len(collected) < limit:
            batch = self._exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=min(1000, limit)
            )
            if not batch:
                break
            collected.extend(batch)
            since = batch[-1][0] + tf_ms
            if len(batch) < 1000:
                break
            time.sleep((self._exchange.rateLimit or 200) / 1000.0)

        rows = [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in collected]
        return DataLoader._rows_to_df(rows).tail(limit)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price_hint: float | None = None,
    ) -> OrderResult:
        side = side.lower()
        if amount <= 0:
            raise ValueError("amount must be positive")

        if self.mode == ExchangeMode.PAPER:
            price = price_hint or self.fetch_ticker(symbol)
            cost = amount * price
            fee = cost * 0.001
            self._order_counter += 1
            result = OrderResult(
                order_id=f"paper-{self._order_counter}",
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                cost=cost,
                fee=fee,
                status="filled",
                mode=ExchangeMode.PAPER,
            )
            self._paper_orders.append(result)
            if len(self._paper_orders) > 500:
                self._paper_orders = self._paper_orders[-500:]
            return result

        if self.mode == ExchangeMode.LIVE:
            # Garde-fou : aucun chemin du simulateur ne doit passer un ordre
            # RÉEL sans opt-in explicite. Le mode d'exécution LIVE du bot est
            # « prédiction seule » ; TRAIN/PAPER rejouent ou simulent — un ordre
            # réel depuis ces boucles serait toujours une erreur (audit 2026-07).
            if os.environ.get("BINANCE_ALLOW_REAL_ORDERS") != "1":
                raise RuntimeError(
                    "Ordre RÉEL bloqué — exporte BINANCE_ALLOW_REAL_ORDERS=1 "
                    "pour armer l'exécution réelle (aucun mode du simulateur "
                    "ne la requiert)."
                )
            assert self._exchange is not None
            raw = self._exchange.create_order(symbol, "market", side, amount)
            price = float(raw.get("average") or raw.get("price") or 0)
            if price <= 0 and price_hint:
                # Les fills market peuvent revenir sans prix moyen immédiat —
                # un prix 0 polluerait entry_price/PnL en aval.
                price = float(price_hint)
            return OrderResult(
                order_id=str(raw.get("id", "")),
                symbol=symbol,
                side=side,
                amount=float(raw.get("amount", amount)),
                price=price,
                cost=float(raw.get("cost") or 0),
                fee=float((raw.get("fee") or {}).get("cost") or 0),
                status=str(raw.get("status", "unknown")),
                mode=ExchangeMode.LIVE,
                raw=raw,
            )

        raise RuntimeError(f"Orders not supported in mode {self.mode}")

    def recent_orders(self, n: int = 10) -> list[OrderResult]:
        return self._paper_orders[-n:]