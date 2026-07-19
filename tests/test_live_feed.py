"""Live feed: the forming candle must be excluded; indices must stay stable.

Uses a fake exchange client so the test is deterministic and offline. The bug
these guard against: ccxt returns the *currently forming* candle as the last
row, and its moving close poisons predictions; and a rolling window shifts bar
indices so settlements misalign.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.live_feed import LiveMarketFeed
from core.timeutil import timeframe_ms


class _FakeClient:
    """Minimal ExchangeClient stand-in returning controllable OHLCV."""

    exchange_id = "binance"

    def __init__(self, n: int, tf: str = "1h") -> None:
        self.tf = tf
        self.tf_ms = timeframe_ms(tf)
        # Anchor so the last bar's open is mid-interval => it is "forming".
        self._now = 1_700_000_000_000
        self._last_open = self._now - (self._now % self.tf_ms)
        self._n = n

    def now_ms(self) -> int:
        return self._now

    def ping(self):
        return {"status": "ok"}

    def _frame(self, n: int) -> pd.DataFrame:
        # n closed bars + 1 forming bar (open == self._last_open, still open now)
        opens = [self._last_open - (n - i) * self.tf_ms for i in range(n)]
        opens.append(self._last_open)  # the forming candle
        ts = np.array(opens, dtype=np.int64)
        closes = 100.0 + np.arange(len(ts), dtype=float)
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes + 1,
                "low": closes - 1,
                "close": closes,
                "volume": np.ones(len(ts)),
            },
            index=pd.to_datetime(ts, unit="ms", utc=True),
        )
        return df

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=500):
        return self._frame(self._n)

    def advance_one_candle(self) -> None:
        """Simulate time passing: forming candle closes, a new one opens."""
        self._now += self.tf_ms
        self._last_open += self.tf_ms
        self._n += 1


def test_forming_candle_is_excluded():
    client = _FakeClient(n=60)
    feed = LiveMarketFeed(symbol="BTC/USDT", timeframe="1h", client=client)

    # The forming candle (last row) must NOT be in the closed-bar series.
    assert feed.forming_ts == client._last_open
    assert feed.timestamps[-1] < client._last_open
    # Last closed bar = exactly one timeframe before the forming candle's open.
    assert feed.timestamps[-1] == client._last_open - client.tf_ms
    closed_n = feed.n_bars


def test_indices_are_stable_across_refresh():
    client = _FakeClient(n=60)
    feed = LiveMarketFeed(symbol="BTC/USDT", timeframe="1h", client=client)
    feed.refresh_interval_s = 0.0  # always allow a refresh

    first_ts = feed.timestamps.copy()
    n0 = feed.n_bars

    # A candle closes; refresh must APPEND exactly one bar, keeping old indices.
    client.advance_one_candle()
    appended = feed.maybe_refresh()
    assert appended is True
    assert feed.n_bars == n0 + 1
    # Every previously-held timestamp keeps its index (no window roll).
    assert np.array_equal(feed.timestamps[:n0], first_ts)
    # The newly-appended closed bar is the one that just finished forming.
    assert feed.timestamps[n0 - 1] == first_ts[-1]


def test_seconds_to_next_close_positive():
    client = _FakeClient(n=60)
    feed = LiveMarketFeed(symbol="BTC/USDT", timeframe="1h", client=client)
    secs = feed.seconds_to_next_close()
    assert secs is not None and 0.0 < secs <= 3600.0


if __name__ == "__main__":
    test_forming_candle_is_excluded()
    test_indices_are_stable_across_refresh()
    test_seconds_to_next_close_positive()
    print("✓ live-feed forming-candle + stable-index tests OK")
