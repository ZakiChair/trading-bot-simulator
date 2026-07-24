"""Market assets and data settings."""
from __future__ import annotations

from pathlib import Path

ASSETS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

DEFAULT_ASSET = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
TIMEFRAMES: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_EXCHANGE = "binance"
DEFAULT_LIMIT = 1500
# Profondeur d'historique pour l'ENTRAÎNEMENT du modèle de bougie. La session
# n'affiche/rejoue que DEFAULT_LIMIT barres (coût TUI), mais le cache en détient
# bien plus (13k+ en 1h) — entraîner sur la fenêtre d'affichage laissait ~90 %
# des données au tiroir (~1 270 échantillons au lieu de ~5 000). L'ancrage OOS
# de la marche Live/Paper se fait par TIMESTAMP (train_end_ts), pas par index,
# pour rester correct quand l'historique d'entraînement est plus long que la
# fenêtre de session.
TRAIN_HISTORY_BARS = 6000

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
DB_PATH = CACHE_DIR / "market_data.sqlite"

WARMUP_BARS = 100
CHART_WINDOW = 72