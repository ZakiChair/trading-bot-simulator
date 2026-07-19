"""OHLCV loader: cache → ccxt → per-symbol synthetic fallback."""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.market_config import CACHE_DIR, DB_PATH, DEFAULT_EXCHANGE, DEFAULT_LIMIT

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def timeframe_minutes(tf: str) -> int:
    import re

    m = re.fullmatch(r"(\d+)([mhdw])", str(tf))
    if not m:
        # Message net plutôt qu'un ValueError d'int() illisible — ce chemin est
        # aussi celui du repli synthétique, qui re-crashait en cascade.
        raise ValueError(
            f"timeframe invalide: {tf!r} (attendu <nombre><m|h|d|w>, ex. 15m, 1h, 4h)"
        )
    return int(m.group(1)) * {"m": 1, "h": 60, "d": 1440, "w": 10080}[m.group(2)]


class DataLoader:
    def __init__(self, exchange_id: str = DEFAULT_EXCHANGE, db_path: Path = DB_PATH):
        self.exchange_id = exchange_id
        self.db_path = db_path
        self._exchanges: dict = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # Toute date OHLCV plausible en ms est > 1e12 (2001) ; une valeur en dessous
    # est un timestamp en SECONDES entré par un chemin d'import (3296 lignes
    # « 1970 » trouvées dans BTC/USDT 4h — audit 2026-07-02).
    _MS_FLOOR = 1_000_000_000_000

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    exchange TEXT, symbol TEXT, timeframe TEXT,
                    ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (exchange, symbol, timeframe, ts)
                )
                """
            )
            self._migrate_seconds_rows(conn)

    def _migrate_seconds_rows(self, conn) -> None:
        """Convertit en ms les lignes stockées en secondes, puis les purge.

        Ces lignes triaient en tête de ``ORDER BY ts`` et se rendaient sous des
        dates 1970 dans le graphe ; ``INSERT OR IGNORE`` préserve les vraies
        lignes ms déjà présentes sur le même timestamp.
        """
        n_bad = conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE ts < ?", (self._MS_FLOOR,)
        ).fetchone()[0]
        if not n_bad:
            return
        conn.execute(
            """
            INSERT OR IGNORE INTO ohlcv
            SELECT exchange, symbol, timeframe, ts * 1000,
                   open, high, low, close, volume
            FROM ohlcv WHERE ts < ? AND ts > 1000000000
            """,
            (self._MS_FLOOR,),
        )
        conn.execute("DELETE FROM ohlcv WHERE ts < ?", (self._MS_FLOOR,))

    # Fraîcheur du cache : au-delà de N barres de retard sur l'horloge, on tente
    # un rafraîchissement réseau (le cache restait sinon figé À VIE dès qu'il
    # comptait 200 lignes — des semaines de retard étiquetées « cache: » sans
    # que rien ne le signale). En cas d'échec réseau, le cache périmé reste la
    # meilleure donnée disponible et est servi tel quel, étiqueté « stale ».
    _STALE_BARS = 6

    def load(self, symbol: str, timeframe: str = "1h", limit: int = DEFAULT_LIMIT) -> tuple[pd.DataFrame, str]:
        """Return (ohlcv_df, source_label)."""
        cached = self._cache_get(symbol, timeframe, limit)
        if len(cached) >= 200:
            age_ms = time.time() * 1000 - cached.index[-1].timestamp() * 1000
            stale = age_ms > self._STALE_BARS * timeframe_minutes(timeframe) * 60_000
            if not stale:
                return cached, f"cache:{symbol}"
            try:
                df, label = self._fetch_ccxt(symbol, timeframe, limit)
                self._cache_put(symbol, timeframe, df)
                return df, label
            except Exception:
                return cached, f"cache-stale:{symbol}"

        try:
            df, label = self._fetch_ccxt(symbol, timeframe, limit)
            self._cache_put(symbol, timeframe, df)
            return df, label
        except Exception as exc:
            fb = self._synthetic(symbol, timeframe, limit)
            return fb, f"synthetic:{symbol} ({exc.__class__.__name__})"

    def _cache_get(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume FROM ohlcv
                WHERE exchange=? AND symbol=? AND timeframe=?
                ORDER BY ts DESC LIMIT ?
                """,
                (self.exchange_id, symbol, timeframe, limit),
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=OHLCV_COLS)
        rows.reverse()
        return self._rows_to_df(rows)

    def _cache_put(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        rows = [
            (self.exchange_id, symbol, timeframe, int(ts.timestamp() * 1000),
             float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume))
            for ts, r in df.iterrows()
        ]
        # Garde d'échelle : jamais de timestamp non-ms dans le cache.
        rows = [r for r in rows if r[3] >= self._MS_FLOOR]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", rows
            )

    def _fetch_ccxt(self, symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame, str]:
        import ccxt

        ex = self._get_exchange()
        tf_ms = timeframe_minutes(timeframe) * 60_000
        since = ex.milliseconds() - limit * tf_ms
        collected: list = []
        while len(collected) < limit:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            if not batch:
                break
            collected.extend(batch)
            since = batch[-1][0] + tf_ms
            if len(batch) < 1000:
                break
            time.sleep((ex.rateLimit or 200) / 1000.0)

        if len(collected) < 50:
            raise RuntimeError(f"insufficient data for {symbol}")

        # La bougie EN FORMATION revient en dernière ligne de fetch_ohlcv : son
        # « close » est le prix live et bouge jusqu'à la clôture. La garder
        # comme barre close (l'ancien comportement) mutait la dernière barre à
        # chaque re-fetch — verdicts paper/backtests instables à code identique.
        tf_ms = timeframe_minutes(timeframe) * 60_000
        now_ms = int(time.time() * 1000)
        if collected and collected[-1][0] + tf_ms > now_ms:
            collected = collected[:-1]

        rows = [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in collected]
        df = self._rows_to_df(rows).tail(limit)
        return df, f"ccxt:{self.exchange_id}:{symbol}"

    def _get_exchange(self):
        if self.exchange_id not in self._exchanges:
            import ccxt

            klass = getattr(ccxt, self.exchange_id)
            self._exchanges[self.exchange_id] = klass({"enableRateLimit": True, "timeout": 20000})
        return self._exchanges[self.exchange_id]

    @staticmethod
    def _rows_to_df(rows: list) -> pd.DataFrame:
        arr = np.asarray(rows, dtype="float64")
        idx = pd.to_datetime(arr[:, 0], unit="ms", utc=True)
        df = pd.DataFrame(arr[:, 1:], columns=OHLCV_COLS, index=idx)
        return df.dropna().sort_index()

    @staticmethod
    def _synthetic(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        seed = int.from_bytes(hashlib.sha256(symbol.encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        tf_min = timeframe_minutes(timeframe)
        bars_per_year = (365 * 24 * 60) / tf_min
        mu = 0.20 / bars_per_year
        sigma = 0.65 / np.sqrt(bars_per_year)

        base = {"BTC/USDT": 50_000, "ETH/USDT": 3_000, "SOL/USDT": 120}.get(symbol, 1000)
        close = base
        rows = []
        t0 = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=tf_min * limit)
        for i in range(limit):
            shock = rng.normal(mu, sigma)
            o = close
            c = o * np.exp(shock)
            h = max(o, c) * (1 + abs(rng.normal(0, sigma * 0.3)))
            l = min(o, c) * (1 - abs(rng.normal(0, sigma * 0.3)))
            v = float(rng.lognormal(10, 0.5))
            ts = int((t0 + pd.Timedelta(minutes=tf_min * i)).timestamp() * 1000)
            rows.append((ts, o, h, l, c, v))
            close = c
        return DataLoader._rows_to_df(rows)

    @staticmethod
    def import_shared_cache(other_db: Path) -> int:
        """Copy rows from crypto-backtest-terminal cache if present."""
        if not other_db.exists():
            return 0
        copied = 0
        with sqlite3.connect(other_db) as src, sqlite3.connect(DB_PATH) as dst:
            dst.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    exchange TEXT, symbol TEXT, timeframe TEXT,
                    ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (exchange, symbol, timeframe, ts)
                )
                """
            )
            # Écarter les timestamps en secondes (la source du bug « 1970 »).
            rows = src.execute(
                "SELECT * FROM ohlcv WHERE ts >= ?", (DataLoader._MS_FLOOR,)
            ).fetchall()
            if rows:
                dst.executemany(
                    "INSERT OR IGNORE INTO ohlcv VALUES (?,?,?,?,?,?,?,?,?)", rows
                )
                copied = len(rows)
        return copied