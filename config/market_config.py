"""Market assets and data settings."""
from __future__ import annotations

from pathlib import Path

ASSETS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

DEFAULT_ASSET = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
TIMEFRAMES: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_EXCHANGE = "binance"
DEFAULT_LIMIT = 1500

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
DB_PATH = CACHE_DIR / "market_data.sqlite"

WARMUP_BARS = 100
CHART_WINDOW = 72