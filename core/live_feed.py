"""Live market feed — real-time OHLCV from Binance.

Two subtleties make a naïve live feed produce *incoherent* predictions, both
fixed here:

* **The forming candle.** ccxt returns the candle that is *currently forming*
  as the last row of ``fetch_ohlcv``. Its "close" is just the live price and
  keeps moving until the bar closes, so using it as a closed bar poisons both
  the prediction's base price and the realised price it is scored against. We
  **drop** it: ``prices[-1]`` is always the last *closed* candle, and the bar we
  predict is the one now forming (not yet in the array).

* **Index drift.** Tracking a pending prediction by integer bar index breaks if
  the window rolls (old bars fall off the front on each refresh). We
  **accumulate** closed candles keyed by open-timestamp, only ever *appending*
  newly-closed bars, so a given candle keeps the same index for the whole
  session and settlements line up.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from core.market import MarketSimulator, MarketState, _detect_regimes
from core.timeutil import timeframe_ms
from exchange.live_client import ExchangeClient, ExchangeMode


@dataclass
class LiveMarketFeed(MarketSimulator):
    """Market simulator that refreshes from a live exchange connection.

    Only **closed** candles ever enter the price arrays; the forming candle is
    tracked separately (``forming_price`` / ``forming_ts``) for display.
    """

    client: ExchangeClient = field(
        default_factory=lambda: ExchangeClient(mode=ExchangeMode.PAPER)
    )
    refresh_interval_s: float = 20.0
    _last_refresh: float = field(default=0.0, repr=False)
    # Live forming candle (the one we are predicting) — not part of `prices`.
    forming_price: float = field(default=0.0, repr=False)
    forming_ts: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._pull_live_data()

    @property
    def is_live(self) -> bool:
        return True

    def _pull_live_data(self) -> bool:
        """Pull OHLCV, drop the forming candle, append only new closed bars.

        Returns True when at least one freshly-closed candle was appended.
        """
        df = self.client.fetch_ohlcv(self.symbol, self.timeframe, limit=500)
        if len(df) < 50:
            raise RuntimeError(f"Données insuffisantes pour {self.symbol}")

        closes = df["close"].to_numpy(dtype=np.float64)
        vols = df["volume"].to_numpy(dtype=np.float64)
        # Real intrabar OHLC — the rough-vol/HMM heads run on range-based RV
        # (Parkinson) when the high/low are genuine. Dropping them (the old
        # behaviour) silently degraded σ̂ to the noisy squared-close proxy for
        # every live/paper-live session, inflating the forecast variance.
        opens_a = df["open"].to_numpy(dtype=np.float64) if "open" in df else None
        highs_a = df["high"].to_numpy(dtype=np.float64) if "high" in df else None
        lows_a = df["low"].to_numpy(dtype=np.float64) if "low" in df else None
        ts = df.index.as_unit("ms").astype(np.int64).to_numpy()

        # Identify the forming candle: its open + one timeframe is still in the
        # future relative to the exchange clock. Drop it from the closed series.
        tf_ms = timeframe_ms(self.timeframe)
        now = self.client.now_ms()
        if len(ts) and ts[-1] + tf_ms > now:
            self.forming_price = float(closes[-1])
            self.forming_ts = int(ts[-1])
            closes, vols, ts = closes[:-1], vols[:-1], ts[:-1]
            if opens_a is not None:
                opens_a, highs_a, lows_a = opens_a[:-1], highs_a[:-1], lows_a[:-1]
        else:
            # No candle currently forming (rare/at boundary): the next one opens
            # one timeframe after the last closed bar.
            self.forming_price = float(closes[-1]) if len(closes) else 0.0
            self.forming_ts = int(ts[-1]) + tf_ms if len(ts) else None

        if self.prices is None or len(self.prices) == 0:
            # First pull — seed the whole closed-bar history.
            self.prices = closes
            self.volumes = vols
            self.timestamps = ts
            self.opens, self.highs, self.lows = opens_a, highs_a, lows_a
            self.regimes = _detect_regimes(self.prices)
            self.n_bars = len(self.prices)
            self.source = f"live:{self.client.exchange_id}:{self.symbol}"
            self._last_refresh = time.monotonic()
            return False

        # Subsequent pulls — append only candles strictly newer than the last
        # one we already hold (stable indices, no window roll).
        last_ts = int(self.timestamps[-1]) if self.timestamps is not None else -1
        mask = ts > last_ts
        appended = int(mask.sum())
        if appended:
            self.prices = np.concatenate([self.prices, closes[mask]])
            self.volumes = np.concatenate([self.volumes, vols[mask]])
            self.timestamps = np.concatenate([self.timestamps, ts[mask]])
            if opens_a is not None and self.opens is not None:
                self.opens = np.concatenate([self.opens, opens_a[mask]])
                self.highs = np.concatenate([self.highs, highs_a[mask]])
                self.lows = np.concatenate([self.lows, lows_a[mask]])
            self.regimes = _detect_regimes(self.prices)
            self.n_bars = len(self.prices)
        self._last_refresh = time.monotonic()
        return appended > 0

    def maybe_refresh(self) -> bool:
        if time.monotonic() - self._last_refresh < self.refresh_interval_s:
            return False
        return self._pull_live_data()

    def force_refresh(self) -> bool:
        """Pull the latest bars now, bypassing the refresh interval.

        Used when entering Live mode so the bot predicts the candle currently
        forming on the freshest data, not a bar that may be up to one refresh
        interval stale.
        """
        return self._pull_live_data()

    def seconds_to_next_close(self) -> float | None:
        """Seconds until the forming candle closes (None if unknown)."""
        if self.forming_ts is None:
            return None
        close_ms = self.forming_ts + timeframe_ms(self.timeframe)
        return max(0.0, (close_ms - self.client.now_ms()) / 1000.0)

    def advance(self, state: MarketState) -> MarketState | None:
        nxt = state.step + 1
        if nxt >= self.n_bars:
            self.maybe_refresh()
        if nxt >= self.n_bars:
            # Pas de nouvelle bougie clôturée : on N'AVANCE PAS. L'ancien code
            # rembobinait state.step à n_bars-2 et re-traitait la même barre à
            # chaque tick (métriques/verdicts corrompus) — et le parent
            # construisait l'état suivant sur les tableaux FIGÉS du state
            # (copiés à l'ouverture d'épisode), qui après un refresh du feed ne
            # couvrent plus les nouveaux indices (IndexError reproductible).
            return None
        # État reconstruit sur les tableaux FRAIS du feed (state_at copie les
        # tableaux du simulateur, qui sont rebindés à chaque refresh).
        return self.state_at(nxt)

    @classmethod
    def create(
        cls,
        symbol: str,
        timeframe: str = "1h",
        mode: ExchangeMode = ExchangeMode.PAPER,
    ) -> LiveMarketFeed:
        client = ExchangeClient(mode=mode)
        client.ping()
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            client=client,
            source=f"live:{client.exchange_id}:{symbol}",
        )
