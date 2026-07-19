"""Market feed: synthetic or real OHLCV-backed price paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from config.market_config import WARMUP_BARS


class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"
    VOLATILE = "volatile"


@dataclass
class MarketState:
    prices: np.ndarray
    volumes: np.ndarray
    step: int = 0
    regime: Regime = Regime.RANGE
    symbol: str = "BTC/USDT"
    source: str = "synthetic"
    timestamps: np.ndarray | None = None
    opens: np.ndarray | None = None
    highs: np.ndarray | None = None
    lows: np.ndarray | None = None

    @property
    def price(self) -> float:
        return float(self.prices[self.step])

    @property
    def history(self) -> np.ndarray:
        return self.prices[: self.step + 1]

    def chart_window(self, window: int = 72) -> np.ndarray:
        """Bounded price window for charts — never grows past ``window`` bars."""
        hist = self.history
        return hist[-window:] if len(hist) > window else hist

    def ts_window(self, window: int = 72) -> np.ndarray | None:
        """Timestamps (UTC ms) aligned to ``chart_window``/``ohlc_window``.

        Same right-aligned slice the close/OHLC windows use, so a label at
        bucket *i* names the same bar. ``None`` when the feed carries no
        timestamps (synthetic data) — the chart then falls back to bar indices.
        """
        if self.timestamps is None:
            return None
        end = self.step + 1
        start = max(0, end - window)
        return self.timestamps[start:end]

    def ohlc_window(self, window: int = 72) -> dict[str, list[float]] | None:
        """Bounded OHLC window for candlestick charts (None if no OHLC).

        Synthesises open/high/low from the close series when the feed only
        carries closes, so the chart can always draw candles.
        """
        end = self.step + 1
        start = max(0, end - window)
        close = self.prices[start:end]
        if len(close) < 2:
            return None
        if self.opens is not None and self.highs is not None and self.lows is not None:
            o = self.opens[start:end]
            h = self.highs[start:end]
            low = self.lows[start:end]
        else:
            # Derive plausible OHLC from closes: open = previous close, and the
            # wick extends a fraction of the bar's move beyond the body.
            o = np.empty_like(close)
            o[0] = close[0]
            o[1:] = close[:-1]
            body_hi = np.maximum(o, close)
            body_lo = np.minimum(o, close)
            span = np.abs(close - o)
            h = body_hi + span * 0.35
            low = body_lo - span * 0.35
        return {
            "Open": [float(x) for x in o],
            "High": [float(x) for x in h],
            "Low": [float(x) for x in low],
            "Close": [float(x) for x in close],
        }

    def returns(self, window: int = 20) -> np.ndarray:
        hist = self.history
        if len(hist) < 2:
            return np.array([0.0])
        r = np.diff(np.log(hist))
        return r[-window:] if len(r) >= window else r

    def volatility(self, window: int = 20) -> float:
        r = self.returns(window)
        return float(np.std(r)) if len(r) else 0.01

    def ewma_volatility(self, lam: float = 0.94, lookback: int = 150) -> float:
        """RiskMetrics EWMA volatility: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}.

        Weights recent returns more, so it tracks **volatility clustering** —
        the genuinely predictable part of a candle's *magnitude*. On real data
        this is a sharper one-step-ahead σ forecast than the flat 20-bar window
        (lower Gaussian NLL of the realised next return), which is what the GBM
        layer needs to size the bubble spread / scenario paths. Falls back to the
        flat estimate when history is too short.
        """
        hist = self.history
        if len(hist) < 6:
            return self.volatility()
        r = np.diff(np.log(hist[-(lookback + 1):]))
        if len(r) < 4:
            return self.volatility()
        var = float(r[0] ** 2)
        for x in r[1:]:
            var = lam * var + (1.0 - lam) * float(x) * float(x)
        sig = float(np.sqrt(max(var, 1e-12)))
        return sig if np.isfinite(sig) and sig > 0.0 else self.volatility()

    def momentum(self, window: int = 10) -> float:
        hist = self.history
        if len(hist) <= window:
            return 0.0
        return float((hist[-1] - hist[-window - 1]) / hist[-window - 1])


@dataclass
class MarketSimulator:
    """Price path from synthetic generation or real OHLCV."""

    n_bars: int = 2000
    seed: int = 42
    symbol: str = "SYNTH"
    source: str = "synthetic"
    timeframe: str = "1h"
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    regimes: list[Regime] = field(default_factory=list)
    timestamps: np.ndarray | None = None
    opens: np.ndarray | None = None
    highs: np.ndarray | None = None
    lows: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.prices) == 0:
            rng = np.random.default_rng(self.seed)
            self.prices, self.volumes, self.regimes = self._generate_synthetic(rng)
            self.n_bars = len(self.prices)
        if self.opens is None and len(self.prices) > 1:
            # Synthesise OHLC from the close path so candlesticks always work.
            close = self.prices
            o = np.empty_like(close)
            o[0] = close[0]
            o[1:] = close[:-1]
            span = np.abs(close - o)
            self.opens = o
            self.highs = np.maximum(o, close) + span * 0.35
            self.lows = np.minimum(o, close) - span * 0.35

    @classmethod
    def from_ohlcv(
        cls,
        df: pd.DataFrame,
        symbol: str,
        source: str,
        timeframe: str = "1h",
    ) -> MarketSimulator:
        prices = df["close"].to_numpy(dtype=np.float64)
        volumes = df["volume"].to_numpy(dtype=np.float64)
        opens = df["open"].to_numpy(dtype=np.float64) if "open" in df else None
        highs = df["high"].to_numpy(dtype=np.float64) if "high" in df else None
        lows = df["low"].to_numpy(dtype=np.float64) if "low" in df else None
        # df.index is a tz-aware DatetimeIndex; its resolution may be ms or ns
        # depending on pandas. Normalise to ms explicitly (a plain astype(int64)
        # of an already-ms index, divided by 1e6, lands back in 1970).
        ts = df.index.as_unit("ms").astype(np.int64)  # UTC ms timestamps
        regimes = _detect_regimes(prices)
        return cls(
            n_bars=len(prices),
            symbol=symbol,
            source=source,
            timeframe=timeframe,
            prices=prices,
            volumes=volumes,
            regimes=regimes,
            timestamps=ts.to_numpy(),
            opens=opens,
            highs=highs,
            lows=lows,
        )

    def _generate_synthetic(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, list[Regime]]:
        prices = np.zeros(self.n_bars)
        volumes = np.zeros(self.n_bars)
        regimes: list[Regime] = []
        price = 50_000.0
        regime = Regime.RANGE
        regime_left = 0

        for i in range(self.n_bars):
            if regime_left <= 0:
                regime = Regime(rng.choice([r.value for r in Regime]))
                regime_left = int(rng.integers(80, 250))
            regime_left -= 1
            regimes.append(regime)

            params = {
                Regime.BULL: (0.0004, 0.008),
                Regime.BEAR: (-0.00035, 0.009),
                Regime.RANGE: (0.0, 0.006),
                Regime.VOLATILE: (0.0, 0.018),
            }[regime]
            drift, vol = params
            shock = rng.normal(drift, vol)
            price *= np.exp(shock)
            prices[i] = price
            volumes[i] = float(rng.lognormal(10, 0.5))

        return prices, volumes, regimes

    def state_at(self, step: int) -> MarketState:
        step = max(0, min(step, self.n_bars - 1))
        return MarketState(
            prices=self.prices.copy(),
            volumes=self.volumes.copy(),
            step=step,
            regime=self.regimes[step] if step < len(self.regimes) else Regime.RANGE,
            symbol=self.symbol,
            source=self.source,
            timestamps=self.timestamps.copy() if self.timestamps is not None else None,
            opens=self.opens.copy() if self.opens is not None else None,
            highs=self.highs.copy() if self.highs is not None else None,
            lows=self.lows.copy() if self.lows is not None else None,
        )

    def advance(self, state: MarketState) -> MarketState | None:
        if state.step >= self.n_bars - 1:
            return None
        nxt = state.step + 1
        return MarketState(
            prices=state.prices,
            volumes=state.volumes,
            step=nxt,
            regime=self.regimes[nxt] if nxt < len(self.regimes) else Regime.RANGE,
            symbol=self.symbol,
            source=self.source,
            timestamps=state.timestamps,
            opens=state.opens,
            highs=state.highs,
            lows=state.lows,
        )

    def reset_step(self, step: int = WARMUP_BARS) -> MarketState:
        """Rewind to warmup step for a new episode on the same data."""
        return self.state_at(min(step, self.n_bars - 2))


def _detect_regimes(prices: np.ndarray, window: int = 20) -> list[Regime]:
    """Classify each bar from rolling momentum and volatility."""
    n = len(prices)
    regimes: list[Regime] = [Regime.RANGE] * n
    if n < window + 2:
        return regimes

    log_ret = np.diff(np.log(prices))
    for i in range(window, n):
        r = log_ret[i - window : i]
        vol = float(np.std(r))
        mom = float((prices[i] - prices[i - window]) / prices[i - window])
        if vol > 0.025:
            regimes[i] = Regime.VOLATILE
        elif mom > 0.03:
            regimes[i] = Regime.BULL
        elif mom < -0.03:
            regimes[i] = Regime.BEAR
        else:
            regimes[i] = Regime.RANGE
    return regimes


def load_market_simulator(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 1500,
) -> MarketSimulator:
    """Build a market simulator from real (or fallback) OHLCV data."""
    from data.loader import DataLoader

    loader = DataLoader()
    shared_cache = os.environ.get("SHARED_OHLCV_CACHE")
    if shared_cache and Path(shared_cache).exists():
        DataLoader.import_shared_cache(Path(shared_cache))

    df, source = loader.load(symbol, timeframe, limit)
    return MarketSimulator.from_ohlcv(df, symbol=symbol, source=source, timeframe=timeframe)