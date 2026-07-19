"""Timeframe + timestamp helpers shared by the forecast and live evaluator.

Exchange bars carry millisecond UTC timestamps (Binance/ccxt). The Live mode
needs to (a) name the *date/time* of the candle it is predicting and (b) stamp
each log line with the wall-clock moment the prediction was made.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Minutes per timeframe unit — covers every entry in config.TIMEFRAMES.
_UNIT_MIN = {"m": 1, "h": 60, "d": 60 * 24, "w": 60 * 24 * 7}


def timeframe_minutes(timeframe: str) -> int:
    """Duration of one candle in minutes (``"1h"`` → 60, ``"15m"`` → 15)."""
    tf = (timeframe or "1h").strip().lower()
    num = "".join(c for c in tf if c.isdigit()) or "1"
    unit = "".join(c for c in tf if c.isalpha()) or "h"
    return int(num) * _UNIT_MIN.get(unit[0], 60)


def timeframe_ms(timeframe: str) -> int:
    """Duration of one candle in milliseconds."""
    return timeframe_minutes(timeframe) * 60_000


def format_ts(ms: int | float | None, *, with_date: bool = True) -> str:
    """Render a millisecond UTC timestamp as ``JJ/MM HH:MM`` (or ``HH:MM``).

    Returns ``"—"`` for a missing timestamp so callers can drop it inline.
    """
    if ms is None:
        return "—"
    try:
        dt = datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return "—"
    return dt.strftime("%d/%m %H:%M") if with_date else dt.strftime("%H:%M")


def now_hms() -> str:
    """Local wall-clock ``HH:MM:SS`` — the moment a prediction is logged."""
    return datetime.now().strftime("%H:%M:%S")


def format_axis_ts(ms: int | float | None, timeframe: str = "1h") -> str:
    """Compact chart-axis label for a bar timestamp, scaled to the timeframe.

    Intraday bars (minutes/hours) need the time of day; daily-or-longer bars
    only need the date. Kept short so a handful fit across the X-axis.
    """
    if ms is None:
        return ""
    try:
        dt = datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return ""
    return dt.strftime("%d/%m %Hh") if timeframe_minutes(timeframe) < 1440 else dt.strftime("%d/%m/%y")

