"""Fixed-size Plotext charts — bounded window, no growth."""
from __future__ import annotations

import datetime as _dt
import math

import numpy as np
from textual.widgets import Static
from textual_plotext import PlotextPlot

from config.market_config import CHART_WINDOW
from core.bot import BotState
from core.timeutil import format_axis_ts

# Fixed plot dimensions (decoupled from widget layout growth)
_PLOT_ROWS = 10
_PLOT_COLS = 58

# A candlestick needs a few terminal cells (body + a gap) to be legible. Cram
# more candles than ``cols / _MIN_CELLS_PER_CANDLE`` and plotext renders them as
# adjacent slivers that blur into a solid wall — the "bougies garbled when you
# enlarge the graph" bug. 3 cells/candle keeps a clear gap between bodies.
_MIN_CELLS_PER_CANDLE = 3

# Price-chart render styles.
PRICE_STYLE_CANDLES = "candles"
PRICE_STYLE_LINE = "line"


def _aggregate_ohlc(
    ohlc: dict[str, list[float]], max_candles: int
) -> tuple[dict[str, list[float]], int]:
    """Down-sample an OHLC window to at most ``max_candles`` candles.

    When a window holds more bars than can be drawn legibly, group consecutive
    bars into composite candles (open = first, high = max, low = min, close =
    last) — exactly how a real chart behaves when you zoom out. Buckets are
    **right-aligned**: the most recent bar always lands in the last bucket and
    the *oldest* remainder is dropped, so the latest candle never shifts.

    Returns the (possibly aggregated) OHLC dict and the aggregation factor
    (1 = untouched).
    """
    n = len(ohlc["Close"])
    if n <= max_candles or max_candles < 1:
        return ohlc, 1
    k = max(1, math.ceil(n / max_candles))
    usable = (n // k) * k
    start = n - usable  # drop the oldest remainder so the newest bar is kept
    o = np.asarray(ohlc["Open"][start:], dtype=float).reshape(-1, k)
    h = np.asarray(ohlc["High"][start:], dtype=float).reshape(-1, k)
    low = np.asarray(ohlc["Low"][start:], dtype=float).reshape(-1, k)
    c = np.asarray(ohlc["Close"][start:], dtype=float).reshape(-1, k)
    return (
        {
            "Open": [float(x) for x in o[:, 0]],
            "High": [float(x) for x in h.max(axis=1)],
            "Low": [float(x) for x in low.min(axis=1)],
            "Close": [float(x) for x in c[:, -1]],
        },
        k,
    )


def _fmt_price(v: float) -> str:
    """Compact, fixed-width price/equity label (keeps left margin constant)."""
    a = abs(v)
    if a >= 1_000_000:
        return f"{v / 1_000_000:>6.2f}M"
    if a >= 1_000:
        return f"{v / 1_000:>6.2f}k"
    return f"{v:>7.1f}"


def _fmt_pct(v: float) -> str:
    return f"{v:>5.0f}%"


class _FixedChart(PlotextPlot):
    """Base chart whose plot canvas follows the widget box (so a chart that is
    the sole content of a tab fills it instead of floating as a small strip with
    dead space around it). Falls back to fixed dims before the first layout pass
    (``size`` is 0 then). ``PriceChart`` overrides these to keep its bounded,
    candle-aggregating behaviour."""

    can_focus = False

    def _plot_cols(self) -> int:
        w = self.size.width
        return max(40, w - 2) if w > 0 else _PLOT_COLS

    def _plot_rows(self) -> int:
        h = self.size.height
        return max(8, h - 2) if h > 0 else _PLOT_ROWS

    def _reset(self) -> None:
        self.plt.clear_figure()
        try:
            self.plt.clear_data()
        except Exception:
            pass
        self.plt.plotsize(self._plot_cols(), self._plot_rows())

    def _stable_axes(
        self,
        values: list[float],
        *,
        yfmt=_fmt_price,
        n_yticks: int = 5,
        n_xticks: int = 5,
        ts_window=None,
        timeframe: str = "1h",
    ) -> None:
        """Pin axis ticks so abscissa/ordinate stay constant tick-to-tick.

        Fixed tick count + fixed-width labels keep the plot frame from
        shifting horizontally as the underlying data range changes. When
        ``ts_window`` is supplied the X labels are real dates/times; otherwise
        they fall back to bar indices.
        """
        if not values:
            return
        lo, hi = float(min(values)), float(max(values))
        if hi - lo < 1e-9:
            lo -= 1.0
            hi += 1.0
        pad = (hi - lo) * 0.06
        lo -= pad
        hi += pad
        self.plt.ylim(lo, hi)
        yt = [lo + (hi - lo) * i / (n_yticks - 1) for i in range(n_yticks)]
        self.plt.yticks(yt, [yfmt(v) for v in yt])

        n = len(values)
        self.plt.xlim(1, max(n, 2))
        step = max(n - 1, 1) / (n_xticks - 1)
        xt = [1 + step * i for i in range(n_xticks)]
        # Plot indices are 1..n; map each tick back to the matching timestamp
        # (the window's tail aligns to ``values`` 1:1) for a real date label.
        if ts_window is not None and len(ts_window) >= 2:
            m = len(ts_window)
            labels = []
            for t in xt:
                ti = min(m - 1, max(0, int(round(t)) - 1 + (m - n)))
                labels.append(format_axis_ts(int(ts_window[ti]), timeframe))
            self.plt.xticks(xt, labels)
        else:
            self.plt.xticks(xt, [f"{int(round(t)):>3d}" for t in xt])

    def on_resize(self, _event=None) -> None:
        self.plt.plotsize(self._plot_cols(), self._plot_rows())
        self.refresh()


class PriceChart(_FixedChart):
    def __init__(self, **kwargs) -> None:
        super().__init__(id="price-chart", **kwargs)
        self.border_title = "Prix"
        self._style = PRICE_STYLE_CANDLES
        self._window = CHART_WINDOW   # how many bars are shown (zoom)
        self._rows = _PLOT_ROWS       # plot height in cells
        self._last_candle_count = 0   # candles actually drawn (for tests/debug)

    def set_style(self, style: str) -> None:
        # Migrate the legacy "boogie" style → candlesticks.
        self._style = PRICE_STYLE_LINE if style == PRICE_STYLE_LINE else PRICE_STYLE_CANDLES

    def set_size(self, window: int, rows: int) -> None:
        """Adjust how much history is shown (window) and the plot height."""
        self._window = max(12, int(window))
        self._rows = max(6, int(rows))
        # CSS still caps the widget box; keep the plot height within it.
        self.styles.height = self._rows + 2
        self.styles.min_height = self._rows + 2
        self.styles.max_height = self._rows + 2

    def _plot_rows(self) -> int:
        return self._rows

    def _plot_cols(self) -> int:
        """Plot width in cells, following the widget so candles use the room.

        Bounded by the *current* widget width (never grows unbounded — the
        "fixed dimensions" guarantee still holds), and falls back to the
        default before the first layout pass, when ``size.width`` is 0.
        """
        w = self.size.width
        if w <= 0:
            return _PLOT_COLS
        return max(40, w - 2)

    def toggle_style(self) -> str:
        self._style = (
            PRICE_STYLE_LINE if self._style == PRICE_STYLE_CANDLES else PRICE_STYLE_CANDLES
        )
        return self._style

    @staticmethod
    def _index_dates(n: int) -> list[str]:
        """Synthetic, evenly-spaced date strings for plotext's candlestick X.

        plotext requires date-form strings; we only care about ordering, so map
        bar index → a daily sequence. Axis labels are overridden to bar numbers.
        """
        base = _dt.datetime(2000, 1, 1)
        return [(base + _dt.timedelta(days=i)).strftime("%d/%m/%Y") for i in range(n)]

    # plotext positions candlesticks by parsing these date strings to numbers, so
    # they must be unique & monotonic; minute resolution satisfies that for every
    # timeframe (>= 1m). The X labels are overridden with a sparse, compact subset.
    _PLOT_DATE_FORM = "d/m/Y H:M"

    @staticmethod
    def _bucket_ts(ts_window, n_drawn: int, agg: int):
        """Representative timestamp (ms) per drawn candle after aggregation.

        Mirrors :func:`_aggregate_ohlc`: right-aligned buckets of width ``agg``,
        oldest remainder dropped. Bucket *j*'s label is its **last** raw bar (the
        candle's close time). Returns ``None`` if no timestamps are available.
        """
        if ts_window is None or len(ts_window) == 0 or n_drawn <= 0:
            return None
        total = len(ts_window)
        usable = n_drawn * agg
        start = max(0, total - usable)
        out = []
        for j in range(n_drawn):
            last = min(total - 1, start + (j + 1) * agg - 1)
            out.append(int(ts_window[last]))
        return out

    def _apply_date_xticks(self, plot_dates: list[str], ts_ms, timeframe: str, n_ticks: int = 4) -> None:
        """Override the X-axis with a sparse set of real, compact date labels."""
        n = len(plot_dates)
        if n == 0 or ts_ms is None:
            return
        n_ticks = max(2, min(n_ticks, n))
        idx = [round(i * (n - 1) / (n_ticks - 1)) for i in range(n_ticks)]
        # De-dup while preserving order (short windows can repeat indices).
        seen, sel = set(), []
        for i in idx:
            if i not in seen:
                seen.add(i)
                sel.append(i)
        self.plt.xticks(
            [plot_dates[i] for i in sel],
            [format_axis_ts(ts_ms[i], timeframe) for i in sel],
        )

    def update_data(self, bot: BotState) -> None:
        market = bot.market
        window = market.chart_window(self._window)
        if len(window) < 3:
            self.refresh()
            return

        cols = self._plot_cols()
        self._reset()
        self.plt.plotsize(cols, self._rows)
        sym = market.symbol.replace("/USDT", "")
        values = window.tolist()
        timeframe = getattr(bot.scenario_engine, "timeframe", "1h")
        ts_window = market.ts_window(self._window)  # UTC ms aligned to the window
        has_ts = ts_window is not None and len(ts_window) >= 2

        if self._style == PRICE_STYLE_CANDLES:
            ohlc = market.ohlc_window(self._window)
            if ohlc is not None and len(ohlc["Close"]) >= 2:
                # Keep every candle legible: cap the drawn count to what the plot
                # width can show at >= _MIN_CELLS_PER_CANDLE cells each, and
                # aggregate the window into that many composite candles (zoom-out
                # behaviour) instead of cramming slivers.
                max_candles = max(8, cols // _MIN_CELLS_PER_CANDLE)
                ohlc, agg = _aggregate_ohlc(ohlc, max_candles)
                n = len(ohlc["Close"])
                self._last_candle_count = n
                bucket_ts = self._bucket_ts(ts_window, n, agg) if has_ts else None
                if bucket_ts is not None:
                    # Position candles at their REAL close times (minute-resolution
                    # date strings keep them unique & monotonic for plotext).
                    self.plt.date_form(self._PLOT_DATE_FORM)
                    dates = [
                        _dt.datetime.utcfromtimestamp(t / 1000.0).strftime("%d/%m/%Y %H:%M")
                        for t in bucket_ts
                    ]
                    self.plt.candlestick(dates, ohlc, colors=["green", "red"])
                    self._stable_axes_ohlc(ohlc["High"], ohlc["Low"])
                    self._apply_date_xticks(dates, bucket_ts, timeframe)
                else:
                    # Synthetic data (no timestamps): keep the index date strings.
                    self.plt.date_form("d/m/Y")
                    dates = self._index_dates(n)
                    self.plt.candlestick(dates, ohlc, colors=["green", "red"])
                    self._stable_axes_ohlc(ohlc["High"], ohlc["Low"])
                mode_label = f"{n} bougies" + (f" ×{agg}" if agg > 1 else "")
            else:  # not enough OHLC yet — fall back to a line
                self.plt.plot(values, marker="braille", label=sym)
                self._stable_axes(values, ts_window=ts_window, timeframe=timeframe)
                mode_label = "ligne"
        else:
            self.plt.plot(values, marker="braille", label=sym)
            self._stable_axes(values, ts_window=ts_window, timeframe=timeframe)
            mode_label = "ligne"

        self.plt.title(f"{market.symbol}  [{market.source}]")
        # Report the bars actually shown (bounded by available history), not the
        # requested window — otherwise an early session reads "300 barres" with
        # only 160 loaded.
        self.border_title = (
            f"{sym} ${market.price:,.2f}  ·  {mode_label}  ·  {len(values)} barres"
        )
        self.refresh()

    def _stable_axes_ohlc(self, highs: list[float], lows: list[float]) -> None:
        """Y range from the candle wicks; X ticks suppressed (date strings are
        synthetic, so showing them would mislead)."""
        lo, hi = float(min(lows)), float(max(highs))
        if hi - lo < 1e-9:
            lo -= 1.0
            hi += 1.0
        pad = (hi - lo) * 0.06
        lo -= pad
        hi += pad
        self.plt.ylim(lo, hi)
        yt = [lo + (hi - lo) * i / 4 for i in range(5)]
        self.plt.yticks(yt, [_fmt_price(v) for v in yt])
        self.plt.xticks([], [])

    def on_resize(self, _event=None) -> None:
        # Keep the user-chosen plot height, and let the width follow the box so
        # candlesticks spread across the available room instead of bunching up.
        self.plt.plotsize(self._plot_cols(), self._rows)
        self.refresh()


class EquityChart(_FixedChart):
    def __init__(self, **kwargs) -> None:
        super().__init__(id="equity-chart", **kwargs)
        self.border_title = "Equity"

    def update_data(self, bot: BotState, history: list[dict]) -> None:
        # Ne tracer que le SEGMENT courant (même bot/portefeuille) : après un
        # aller-retour PAPER→TRAIN, l'historique contenait bout à bout deux
        # livres différents — la courbe splicée (et son buy & hold calculé sur
        # la fenêtre mélangée) ne voulait rien dire.
        if history:
            cur_epoch = history[-1].get("epoch")
            history = [s for s in history if s.get("epoch") == cur_epoch]
        snaps = history[-CHART_WINDOW:]
        curve = [bot.portfolio.initial_cash] + [s["equity"] for s in snaps]
        if len(curve) < 2:
            curve.append(bot.portfolio.mark_to_market(bot.market.price))

        self._reset()
        curve = curve[-CHART_WINDOW:]
        self.plt.plot(curve, marker="braille", label="equity (bot)")
        # Benchmark honnête sur la même échelle $ : buy & hold démarré au même
        # point que la fenêtre affichée — l'œil compare tout de suite.
        prices = [s.get("price", 0.0) for s in snaps]
        if len(prices) >= 2 and prices[0] > 0:
            base = curve[0]
            bench = [base * p / prices[0] for p in prices]
            bench = ([base] + bench)[-len(curve):]
            self.plt.plot(bench, marker="braille", label="buy & hold")
        self._stable_axes(curve)
        expo = snaps[-1].get("exposure", 0.0) if snaps else 0.0
        self.plt.title(f"Equity vs buy & hold · exposition {expo:.0%}")
        self.refresh()


class LearningChart(_FixedChart):
    def __init__(self, **kwargs) -> None:
        super().__init__(id="learning-chart", **kwargs)
        self.border_title = "Apprentissage"

    def update_data(self, history: list[dict], train_report=None, model=None) -> None:
        # Priority for the "apprentissage" headline:
        #   1. a fresh training run this session (train_report), else
        #   2. the loss curve persisted on a loaded model, else
        #   3. the exposure-vs-target risk tracking history, else
        #   4. an explicit empty-state hint (so the tab is never blank).
        rep = train_report
        loss = list(rep.loss_history) if rep is not None and getattr(rep, "loss_history", None) else None
        val_acc = rep.val_accuracy if rep is not None else None
        val_maj = rep.val_majority if rep is not None else None
        val_brier = getattr(rep, "val_brier", None) if rep is not None else None
        temperature = None
        if loss is None and model is not None and getattr(model, "loss_history", None):
            loss = list(model.loss_history)
            val_acc = getattr(model, "val_accuracy", None)
            temperature = getattr(model, "temperature", None)

        if loss:
            self._reset()
            self.plt.plot(loss, marker="braille", label="perte (cross-entropy)")
            lo, hi = min(loss), max(loss)
            if hi - lo < 1e-9:
                lo -= 0.01
                hi += 0.01
            pad = (hi - lo) * 0.08
            ylo, yhi = lo - pad, hi + pad
            self.plt.ylim(ylo, yhi)
            yt = [ylo + (yhi - ylo) * i / 4 for i in range(5)]
            self.plt.yticks(yt, [f"{v:.3f}" for v in yt])
            n = len(loss)
            self.plt.xlim(1, max(n, 2))
            step = max(n - 1, 1) / 4
            xt = [1 + step * i for i in range(5)]
            # Recorded every 5 epochs → x label = epoch number.
            self.plt.xticks(xt, [f"{int(round(t - 1)) * 5:>4d}" for t in xt])
            maj_txt = f" (classe maj. {val_maj:.0%})" if val_maj is not None else ""
            acc_txt = f"  ·  val {val_acc:.0%}{maj_txt}" if val_acc is not None else ""
            brier_txt = f"  ·  Brier {val_brier:.3f}" if val_brier else ""
            self.plt.title(
                f"Descente de gradient — perte {loss[0]:.3f}→{loss[-1]:.3f}{acc_txt}{brier_txt}"
            )
            cal = f" · calibré T={temperature:.2f}" if temperature else ""
            self.border_title = f"Apprentissage modèle — perte / epoch{cal}"
            self.refresh()
            return

        if len(history) < 2:
            # Never leave the tab blank — tell the user how to populate it.
            self._reset()
            self.plt.text(
                "Entraine le modele (g) ou lance un episode (e)",
                x=1, y=1,
            )
            self.plt.xlim(0, 2)
            self.plt.ylim(0, 2)
            self.plt.title("Apprentissage — en attente d'entraînement (g / e)")
            self.border_title = "Apprentissage — touche g pour entraîner"
            self.refresh()
            return

        # Pas de modèle entraîné : montrer le suivi de risque (exposition réelle
        # vs cible) — une mesure honnête du travail du bot, à la place des
        # anciennes courbes « précision/autonomie » qui scoraient du bruit.
        expo = [h.get("exposure", 0) * 100 for h in history[-CHART_WINDOW:]]
        target = [h.get("target_exposure", 0) * 100 for h in history[-CHART_WINDOW:]]

        self._reset()
        self.plt.plot(expo, label="exposition %")
        self.plt.plot(target, label="cible e* %")
        self.plt.ylim(0, 100)
        self.plt.yticks([0, 25, 50, 75, 100], [_fmt_pct(v) for v in (0, 25, 50, 75, 100)])
        n = max(len(expo), len(target))
        self.plt.xlim(1, max(n, 2))
        step = max(n - 1, 1) / 4
        xt = [1 + step * i for i in range(5)]
        self.plt.xticks(xt, [f"{int(round(t)):>3d}" for t in xt])
        self.plt.title("Suivi du budget de risque (g pour entraîner le modèle)")
        self.border_title = "Exposition vs cible"
        self.refresh()


def _grid_dims(n: int) -> tuple[int, int]:
    """Pick a near-square grid for n scenarios (capped at 100×100)."""
    if n <= 100:
        return 10, 10
    side = int(math.ceil(math.sqrt(n)))
    side = min(side, 100)
    return side, side


class ScenarioHeatmap(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__(id="scenario-heatmap", **kwargs)
        self._grid: str = "[dim]Heatmap scénarios…[/]"

    def update_data(self, bot: BotState) -> None:
        bundle = bot.current_bundle
        if not bundle:
            self._grid = "[dim]Aucun scénario[/]"
            self.refresh()
            return

        returns = np.array([s.terminal_return for s in bundle.scenarios])
        probs = np.array([s.probability for s in bundle.scenarios])
        n = len(returns)
        rows, cols = _grid_dims(n)

        chars = " ░▒▓█"
        lines: list[str] = []

        # Top section: probability-weighted return histogram
        lo, hi = float(returns.min()), float(returns.max())
        if hi - lo < 1e-8:
            lo -= 0.01
            hi += 0.01
        n_bins = 16
        bins = np.linspace(lo, hi, n_bins + 1)
        hist = np.zeros(n_bins)
        for v, p in zip(returns, probs):
            idx = int(np.clip(np.searchsorted(bins, v, side="right") - 1, 0, n_bins - 1))
            hist[idx] += p
        mx = hist.max() or 1.0
        lines.append("[bold]Densité probabiliste des rendements[/]")
        for i, h in enumerate(hist):
            bar_len = int(h / mx * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            color = "green" if bins[i] >= 0 else "red"
            lines.append(f"[{color}]{bar}[/] [{bins[i]:+.2%},{bins[i+1]:+.2%}] {h:.1%}")
        lines.append("")

        # Grid section (sample or full depending on n)
        if n <= 400:
            lines.append(f"[bold]Grille {rows}×{cols}[/]  (intensité = |ret|)")
            for r in range(rows):
                cells = []
                for c in range(cols):
                    i = r * cols + c
                    if i >= n:
                        cells.append(" ")
                        continue
                    ret = returns[i]
                    intensity = int(min(4, abs(ret) * 200))
                    if bundle.most_probable_id == i:
                        ch = "★"
                    elif bundle.selected_id == i:
                        ch = "◆"
                    else:
                        ch = chars[intensity]
                    if ret > 0.003:
                        cells.append(f"[green]{ch}[/]")
                    elif ret < -0.003:
                        cells.append(f"[red]{ch}[/]")
                    else:
                        cells.append(f"[yellow]{ch}[/]")
                lines.append("".join(cells))
        else:
            # For large N, show top-probability cells only
            top_k = min(50, n)
            top_idx = np.argsort(probs)[-top_k:][::-1]
            lines.append(f"[bold]Top {top_k} scénarios[/]  (★=scénario d'action, pondéré risque)")
            for rank, i in enumerate(top_idx[:12]):
                ret = returns[i]
                p = probs[i]
                star = "★" if bundle.most_probable_id == i else " "
                color = "green" if ret > 0 else "red" if ret < 0 else "yellow"
                lines.append(
                    f" {star}#{i:4d} [{color}]{ret:+.2%}[/] P={p:.2%}"
                )

        mp = bundle.most_probable
        mp_txt = f"#{mp.id} P={mp.probability:.1%}" if mp else "—"
        lines.append(
            f"[dim]{n:,} scénarios  max P={probs.max():.1%}  "
            f"action(risque)={mp_txt}  sel=#{bundle.selected_id}[/]"
        )
        self._grid = "\n".join(lines)
        self.refresh()

    def render(self) -> str:
        return self._grid