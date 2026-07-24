"""Rich rendering helpers for terminal panels.

Refonte UI : l'écran raconte désormais ce que le bot FAIT réellement — tenir un
budget de risque à partir des têtes validées (σ̂ rough-vol, régime HMM,
calibration conforme) — au lieu d'un « score d'autonomie » cosmétique et de
poids de policy qui apprenaient du bruit. Chaque panneau montre soit une mesure
honnête, soit le raisonnement complet d'une décision (jamais un chiffre
flatteur non mesuré).
"""
from __future__ import annotations

import numpy as np
from rich.console import RenderableType
from rich.table import Table
from rich.text import Text

from core.bot import BotState
from core.portfolio import Action
from core.next_candle import NextCandleForecast
from core.run_mode import RunMode
from core.scenarios import ScenarioBundle
from ui.colors import ThemeStyles

_BUBBLE_GLYPHS = ("•", "●", "◉")  # small · medium · large — bold & widely supported
_DIR_ARROW = {"up": "▲", "down": "▼", "flat": "●"}  # direction marker for the consensus strip
_BUBBLE_PLOT_W = 44
_BUBBLE_PLOT_H = 11
# Frozen bubble-chart axes — keep the abscissa (rendement) and ordinate
# (probabilité) constant tick-to-tick so a given bubble always lands in the
# same cell instead of jumping as the per-frame data range shifts.
_BUBBLE_PROB_CEIL = 0.5      # Y ceiling: top bubble sits high, the rest spread truthfully
_BUBBLE_RETURN_SPAN = 0.04   # X fallback half-range (±4%) when the app supplies none

# Couleur par régime de volatilité HMM.
_REGIME_STYLE = {"calm": "green", "normal": "cyan", "turbulent": "red"}


def _dir_colors(styles: ThemeStyles | None) -> dict[str, str]:
    if styles is None:
        return {"up": "green", "down": "red", "flat": "yellow"}
    return {
        "up": styles.positive,
        "down": styles.negative,
        "flat": styles.rich("warning", "#D29922"),
    }


def _ascii_bar(value: float, total: float, width: int = 28, char: str = "█") -> str:
    if total <= 0:
        return "░" * width
    filled = int(value / total * width)
    filled = max(0, min(width, filled))
    return char * filled + "░" * (width - filled)


def _exposure_gauge(current: float, target: float, width: int = 24) -> Text:
    """Jauge d'exposition 0..100 % : ▓ = exposition actuelle, ▲ = cible."""
    cur = max(0.0, min(1.0, current))
    tgt = max(0.0, min(1.0, target))
    cells = ["░"] * width
    for i in range(int(round(cur * (width - 1))) + 1):
        cells[i] = "▓"
    tpos = int(round(tgt * (width - 1)))
    marker = "▲" if abs(tgt - cur) > 1.0 / width else "◆"
    t = Text()
    t.append("0% ", style="dim")
    t.append("".join(cells[:tpos]))
    t.append(marker, style="bold")
    t.append("".join(cells[tpos + 1:]))
    t.append(" 100%", style="dim")
    return t


def _sparkline(path, width: int = 18) -> str:
    arr = np.array(path, dtype=float)
    if len(arr) < 2:
        return ""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return "─" * min(width, len(arr))
    norm = (arr - mn) / (mx - mn)
    chars = "▁▂▃▄▅▆▇█"
    indices = (norm * (len(chars) - 1)).astype(int)
    if len(indices) > width:
        step = len(indices) / width
        indices = np.array([indices[int(i * step)] for i in range(width)])
    return "".join(chars[i] for i in indices)


def _histogram_bars(
    values: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 12,
    width: int = 24,
) -> list[str]:
    """Weighted histogram as ASCII bars."""
    if len(values) == 0:
        return ["[dim]—[/]"]
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-8:
        lo -= 0.01
        hi += 0.01
    bins = np.linspace(lo, hi, n_bins + 1)
    counts = np.zeros(n_bins)
    for v, w in zip(values, weights):
        idx = int(np.clip(np.searchsorted(bins, v, side="right") - 1, 0, n_bins - 1))
        counts[idx] += w
    mx = counts.max() or 1.0
    lines = []
    for i, c in enumerate(counts):
        left = bins[i]
        right = bins[i + 1]
        bar = _ascii_bar(c, mx, width)
        lines.append(f"  [{left:+.2%},{right:+.2%}] {bar} {c:.1%}")
    return lines


# --------------------------------------------------------------------------- #
# Bandeau statut                                                               #
# --------------------------------------------------------------------------- #
def format_status_strip(
    bot: BotState, auto: bool, styles: ThemeStyles | None = None, paper_verdict=None
) -> Text:
    p = bot.portfolio
    price = bot.market.price
    pnl = p.total_pnl(price)
    pos_style = styles.positive if styles else "green"
    neg_style = styles.negative if styles else "red"
    info_style = styles.info if styles else "cyan"

    t = Text()
    t.append(" ● ", style=f"bold {pos_style}" if auto and not bot.paused else "bold yellow")
    t.append("AUTO" if auto and not bot.paused else "PAUSE", style="bold")
    sym = bot.market.symbol.replace("/USDT", "")
    tf = getattr(bot.scenario_engine, "timeframe", "1h")
    t.append(f"  {sym}·{tf}  s{bot.market.step}")
    t.append(f"  ${price:,.0f}  PnL ")
    t.append(f"{pnl:+,.0f}", style=pos_style if pnl >= 0 else neg_style)

    # L'état de risque — le cœur de la refonte : exposition vs cible, σ̂, régime.
    trace = bot.risk.last_trace
    if trace is not None:
        rstyle = _REGIME_STYLE.get(trace.regime, info_style)
        t.append("  │  exp ")
        t.append(f"{trace.current_exposure:.0%}", style="bold")
        t.append("→")
        t.append(f"{trace.target_exposure:.0%}", style=f"bold {info_style}")
        t.append(f"  σ̂ {trace.sigma_ann:.0%}")
        t.append(f"  {trace.regime}", style=f"bold {rstyle}")

    bundle = bot.current_bundle
    if bundle and bundle.next_candle:
        # Lecture CALIBRÉE (argmax des masses) — la même source que le bandeau
        # consensus et que la fiabilité Live. L'ancienne bulle ★ (pick pondéré
        # risque) pouvait afficher l'inverse du consensus dans le même écran.
        fc = bundle.next_candle
        masses = {"up": fc.prob_up, "flat": fc.prob_flat, "down": fc.prob_down}
        top = max(masses, key=masses.get)  # type: ignore[arg-type]
        labels = {"up": "HAUSSE", "flat": "NEUTRE", "down": "BAISSE"}
        colors = _dir_colors(styles)
        t.append("  │  🔮 ")
        t.append(f"{labels[top][:3]} ", style=f"bold {colors[top]}")
        t.append(f"{fc.expected_return:+.1%} ({masses[top]:.0%})")

    # Honest data-source badge: real-time Binance feed vs replay of stored bars.
    sim = getattr(bot, "market_sim", None)
    is_live_feed = bool(getattr(sim, "is_live", False))
    t.append("  │  ")
    if is_live_feed:
        secs = sim.seconds_to_next_close() if hasattr(sim, "seconds_to_next_close") else None
        fp = getattr(sim, "forming_price", 0.0) or 0.0
        t.append("🔴 BINANCE LIVE", style="bold red")
        if fp:
            t.append(f" ${fp:,.0f}", style="bold")
        if secs is not None:
            mm, ss = divmod(int(secs), 60)
            t.append(f" · clôture dans {mm:02d}:{ss:02d}", style="dim")
    else:
        src = bot.market.source
        kind = "cache" if src.startswith("cache") else "synthétique" if src.startswith("synthetic") else "historique"
        t.append(f"⏪ REPLAY {kind}", style="bold yellow")

    # Behaviour axis (run mode): the single "mode" the user controls. Shown
    # distinctly from the data SOURCE badge above so they never read as one knob.
    cm = getattr(bot, "candle_model", None)
    mode = getattr(bot, "mode", "train")
    t.append("  │  ")
    if mode == "live" and cm is not None:
        ev = getattr(bot, "live_eval", None)
        fc = bundle.next_candle if bundle else None
        tgt = f"#{fc.target_step} {fc.target_dt}" if fc and fc.target_step >= 0 else ""
        t.append(f"🔮 LIVE · estime bougie {tgt} · ", style="bold green")
        if ev is not None and ev.n_eval > 0:
            label, gstyle, _ = ev.grade()
            t.append(
                f"fiabilité {ev.reliability_note:.0f}/100 {label} ({ev.n_eval})",
                style=f"bold {gstyle}",
            )
            n_on = int(getattr(cm, "n_online", 0))
            if n_on > 0:
                t.append(f" · 🔄{n_on}", style="dim")  # online self-reinforcement
        else:
            t.append("attend la 1ʳᵉ clôture", style="green")
    elif mode == "paper":
        kind = "forward" if (paper_verdict is not None and paper_verdict.forward) else "backtest"
        t.append(f"🧪 PAPER {kind} · ", style="bold magenta")
        if paper_verdict is not None:
            vstyle = pos_style if paper_verdict.profitable else neg_style
            t.append(paper_verdict.verdict, style=f"bold {vstyle}")
            if paper_verdict.sig_tag:
                # La significativité voyage AVEC le verdict (rapport V2 §3).
                t.append(f" · {paper_verdict.sig_tag}", style="dim")
        else:
            t.append("évalue la rentabilité…", style="magenta")
    else:
        t.append("🎓 Entraînement (replay + budget de risque)", style="dim")
    return t


def _format_position(portfolio, price: float) -> str:
    """Compact open-position cell: quantity (adaptive precision) + live $ value."""
    qty = portfolio.position
    if abs(qty) >= 100:
        qty_s = f"{qty:,.1f}"
    elif abs(qty) >= 1:
        qty_s = f"{qty:.2f}"
    else:
        qty_s = f"{qty:.4f}"
    value = qty * price
    if abs(value) >= 1000:
        val_s = f"${value / 1000:,.1f}k"
    else:
        val_s = f"${value:,.0f}"
    return f"{qty_s} ≈ {val_s}"


# --------------------------------------------------------------------------- #
# Métriques (sidebar)                                                          #
# --------------------------------------------------------------------------- #
def format_metrics(bot: BotState, styles: ThemeStyles | None = None) -> Table:
    p = bot.portfolio
    price = bot.market.price
    equity = p.mark_to_market(price)
    pnl = p.total_pnl(price)
    pos_style = styles.positive if styles else "green"
    neg_style = styles.negative if styles else "red"
    info_style = styles.info if styles else "cyan"
    tf = getattr(bot.scenario_engine, "timeframe", "1h")

    t = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    t.add_column("k", style="dim", width=11, no_wrap=True)
    t.add_column("v", width=16, no_wrap=True, overflow="ellipsis")

    t.add_row("Actif", f"[bold]{bot.market.symbol}[/]")
    src = bot.market.source
    src_short = (
        "🔴 binance live" if src.startswith("live")
        else "ccxt live" if src.startswith("ccxt")
        else "cache" if src.startswith("cache")
        else "synthetic"
    )
    t.add_row("Source", f"[{info_style}]{src_short}[/]")
    try:
        _rm = RunMode(bot.mode)
        mode_cell = Text(f"{_rm.icon} {bot.mode.upper()}", style="bold")
    except ValueError:
        mode_cell = Text(bot.mode.upper())
    t.add_row("Mode", mode_cell)
    t.add_row("Step", str(bot.market.step))
    t.add_row("Prix", f"${price:,.2f}")

    # --- État de risque (le job du bot) --------------------------------- #
    trace = bot.risk.last_trace
    if trace is not None:
        rstyle = _REGIME_STYLE.get(trace.regime, info_style)
        t.add_row("Régime vol", f"[bold {rstyle}]{trace.regime}[/] [dim](HMM)[/]")
        t.add_row("σ̂ annuel", f"{trace.sigma_ann:.0%}")
        t.add_row("Vol cible", f"{trace.sigma_target_ann:.0%}")
        t.add_row(
            "Exposition",
            f"[bold]{trace.current_exposure:.0%}[/] → [{info_style}]{trace.target_exposure:.0%}[/]",
        )
    rv = bot.metrics.realized_vol_ann(tf)
    if rv > 0:
        gap = rv - bot.risk.sigma_target_ann
        track_style = pos_style if abs(gap) <= 0.10 else neg_style
        t.add_row("Vol réalisée", Text(f"{rv:.0%} ({gap:+.0%})", style=track_style))

    t.add_row("Equity", f"${equity:,.2f}")
    t.add_row("PnL", f"[{pos_style if pnl >= 0 else neg_style}]{pnl:+,.2f}[/]")
    t.add_row("Position", _format_position(p, price) if p.position else "—")
    t.add_row("Trades", str(len(p.trades)))
    t.add_row("Rotation", f"{bot.metrics.turnover:.2f}/barre")
    costs = p.fees_paid + p.slippage_paid
    if costs > 0:
        t.add_row("Coûts payés", f"[{neg_style}]{costs:,.2f}$[/]")

    cm = getattr(bot, "candle_model", None)
    if cm is not None and getattr(cm, "trained", False):
        n_samp = int(getattr(cm, "n_samples", 0))
        caveat = " ⚠" if n_samp and n_samp < 300 else ""
        t.add_row("Modèle", f"val {cm.val_accuracy:.0%} (n={n_samp}{caveat})")
        t.add_row(
            "Mode prév.",
            Text("LIVE", style=f"bold {pos_style}") if bot.predict_mode else Text("entraîn.", style="dim"),
        )
        if bot.predict_mode and getattr(cm, "n_online", 0):
            t.add_row(
                "En ligne",
                Text(f"🔄 {cm.n_online}× · {cm.online_drift():.2f}", style=info_style),
            )
    else:
        t.add_row("Modèle", Text("non entraîné (g)", style="dim"))

    # Live self-evaluation — the reliability note that tells whether the bot
    # needs more training.
    ev = getattr(bot, "live_eval", None)
    if getattr(bot, "predict_mode", False) and ev is not None and ev.n_eval > 0:
        label, gstyle, _needs = ev.grade()
        t.add_row("Fiabilité", Text(f"{ev.reliability_note:.0f}/100", style=f"bold {gstyle}"))
        t.add_row("Dir. juste", f"{ev.accuracy:.0%} / {ev.n_eval}")
        t.add_row("Edge", f"{ev.edge:+.0%}")
        t.add_row("Verdict", Text(label, style=gstyle))
        cov = ev.conformal.empirical_coverage() if ev.conformal.n_seen >= 20 else None
        if cov is not None:
            t.add_row("Couv. 90%", f"{cov:.0%} / {ev.conformal.n_seen}")
    return t


# --------------------------------------------------------------------------- #
# Décision de risque (remplace le « cerveau » à poids appris sur du bruit)     #
# --------------------------------------------------------------------------- #
def format_decision(
    bot: BotState,
    styles: ThemeStyles | None = None,
    history: list[dict] | None = None,
) -> RenderableType:
    """La trace complète de la dernière décision : chaque terme de la formule
    d'exposition, la bande de coûts, et le POURQUOI en une phrase. Remplace
    l'ancien panneau de poids de policy (qui apprenait du bruit par
    construction — cf. ANALYSE_CRITIQUE §1-2 et l'audit de refonte)."""
    pos = styles.positive if styles else "green"
    warn = styles.warning if styles else "yellow"
    info = styles.info if styles else "cyan"
    muted = styles.muted if styles else "dim"

    trace = bot.risk.last_trace
    if trace is None:
        return Text("En attente de la première décision…", style="dim")

    rstyle = _REGIME_STYLE.get(trace.regime, info)
    lines: list[RenderableType] = []
    t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    t.add_column("k", style="bold", width=14, no_wrap=True)
    t.add_column("v", ratio=1)

    t.add_row(
        "σ̂ (rough-vol)",
        f"{trace.sigma_bar:.3%}/barre → [bold]{trace.sigma_ann:.0%}[/] annualisé",
    )
    raw_cell = (
        f"cible {trace.sigma_target_ann:.0%} → e_brut = "
        f"{trace.sigma_target_ann:.0%}/{trace.sigma_ann:.0%} = "
        f"[bold]{min(trace.raw_exposure, 9.99):.0%}[/]"
    )
    if trace.raw_exposure > 1.0:
        # Le ratio est plafonné AVANT les portes (long-only, pas de levier) —
        # sinon la porte de régime serait neutralisée en vol basse.
        raw_cell += " → plafonné [bold]100%[/]"
    t.add_row("Budget de vol", raw_cell)
    t.add_row(
        "Régime (HMM)",
        Text.from_markup(
            f"[bold {rstyle}]{trace.regime}[/] × {trace.regime_mult:.2f}"
            + ("  [dim](dé-risquage turbulence)[/]" if trace.regime_mult < 0.995 else ""),
        ),
    )
    if trace.edge_significant:
        t.add_row("Porte Wilson", f"[{pos}]OUVERTE[/] → tilt {trace.dir_tilt:+.2f}")
    else:
        t.add_row(
            "Porte Wilson",
            f"[{warn}]fermée[/] [{muted}](pas d'edge directionnel significatif → tilt 0)[/]",
        )
    t.add_row(
        "Exposition",
        f"e* = [bold {info}]{trace.target_exposure:.0%}[/]  ·  "
        f"actuelle {trace.current_exposure:.0%}  ·  bande ±{trace.band:.0%}",
    )
    verdict_style = pos if trace.rebalanced else muted
    t.add_row(
        "Décision",
        Text(("⚖ " if trace.rebalanced else "✋ ") + trace.reason, style=verdict_style),
    )
    lines.append(t)
    lines.append(_exposure_gauge(trace.current_exposure, trace.target_exposure, width=30))

    if history:
        expos = [h.get("exposure", 0.0) for h in history[-60:]]
        if len(expos) >= 3:
            spark = _sparkline(expos, width=30)
            foot = Text()
            foot.append("expo 60 barres ", style="dim")
            foot.append(spark, style=info)
            lines.append(foot)

    from rich.console import Group
    return Group(*lines)


# --------------------------------------------------------------------------- #
# Pipeline probabiliste                                                       #
# --------------------------------------------------------------------------- #
def format_probabilistic_models(bot: BotState, styles: ThemeStyles | None = None) -> RenderableType:
    """Le pipeline réel de la refonte, étape par étape, avec les formules et les
    paramètres du tick courant. Plus de couche Markov décorative : la matrice de
    transition n'apparaît que comme statistique descriptive du forecast 1 barre,
    avec sa faiblesse dite (V de Cramér ≈ 0,09 — quasi bruit)."""
    bundle = bot.current_bundle
    head = styles.heading if styles else "bold cyan"
    pos = styles.positive if styles else "green"
    warn = styles.warning if styles else "yellow"
    muted = styles.muted if styles else "dim"
    if bundle is None or bundle.model is None:
        return Text("En attente de la première génération…", style="dim")

    m = bundle.model
    mk = bundle.markov

    t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    t.add_column("Étape", style=head, width=22)
    t.add_column("Détail", ratio=1)

    t.add_row(
        "[bold]Pipeline[/]",
        "σ̂ rough-vol → cône Student-t (martingale) → régime HMM → conformal → décision",
    )

    t.add_row("1. Vol rough (RFSV)", m.rough_formula)
    t.add_row(
        "   Paramètres",
        f"σ₁={m.estimated_vol:.4f}  ν_t={m.cone_nu:.1f}  H≈{m.rough_hurst:.2f} "
        f"[{muted}](H borné vers le bas par le bruit du proxy RV)[/]",
    )
    t.add_row("2. Cône fat-tailed", m.cone_formula)
    qs = bundle.terminal_quantiles((0.05, 0.5, 0.95))
    t.add_row(
        "   Ensemble MC",
        f"{m.n_scenarios:,} trajectoires × {m.horizon} barres · centre=martingale · "
        f"q05={qs[0.05]:+.1%}  q50={qs[0.5]:+.1%}  q95={qs[0.95]:+.1%}",
    )
    rstyle = _REGIME_STYLE.get(m.vol_regime, "cyan")
    t.add_row(
        "3. Régime HMM",
        f"3 états de vol (Baum-Welch, log-RV lissé) → [bold {rstyle}]{m.vol_regime}[/] "
        f"[{muted}](descriptif — pas un forecast directionnel)[/]",
    )
    ev = getattr(bot, "live_eval", None)
    if ev is not None and ev.conformal.n_seen >= 20:
        cov = ev.conformal.empirical_coverage()
        t.add_row(
            "4. Conformal + ACI",
            f"couverture réalisée [bold]{cov:.0%}[/] (cible 90%) sur {ev.conformal.n_seen} bougies · "
            f"α_t={ev.conformal.alpha_t:.2f}",
        )
    else:
        t.add_row(
            "4. Conformal + ACI",
            f"[{muted}]intervalles next-bar à couverture garantie — s'alimente en mode Live[/]",
        )
    cm = getattr(bot, "candle_model", None)
    if cm is not None and getattr(cm, "trained", False):
        t.add_row(
            "5. Modèle bougie",
            f"softmax calibré (T={getattr(cm, 'temperature', 1.0):.2f}) · val {cm.val_accuracy:.0%} · "
            f"[{warn}]calibré, pas d'edge directionnel (structurel)[/]",
        )
    else:
        t.add_row("5. Modèle bougie", f"[{muted}]non entraîné — touche g[/]")

    t.add_row("6. Décision", m.decision_formula)
    trace = bot.risk.last_trace
    if trace is not None:
        t.add_row("   Trace", f"[bold]{trace.formula_line()}[/]")
        t.add_row("   Verdict", trace.reason)

    t.add_row("", "")
    t.add_row("[bold]Consensus cône[/]", bundle.verdict())

    if mk is not None and mk.direction_matrix is not None:
        t.add_row("", "")
        t.add_row(
            "[bold]Transition 1 barre[/]",
            f"[{muted}]fréquences conditionnelles (500 barres) — dépendance mesurée "
            f"très faible (V≈0.09) : descriptif, pas un signal[/]",
        )
        for line in mk.direction_matrix_text():
            t.add_row("", line)

    return t


# --------------------------------------------------------------------------- #
# Bulles prochaine bougie                                                     #
# --------------------------------------------------------------------------- #
def _bubble_glyph_index(prob: float, max_prob: float) -> int:
    ratio = prob / max(max_prob, 1e-9)
    if ratio >= 0.40:
        return 2
    if ratio >= 0.08:
        return 1
    return 0


def _stamp_bubble(
    cells: list[list[tuple[str, str]]],
    cx: int,
    cy: int,
    glyph_idx: int,
    color: str,
    *,
    is_star: bool = False,
) -> None:
    """Bubble with a size-proportional halo so probability mass is visible."""
    glyph = _BUBBLE_GLYPHS[glyph_idx]
    h = len(cells)
    w = len(cells[0]) if cells else 0
    ch = "★" if is_star else glyph
    if 0 <= cx < w and 0 <= cy < h:
        cells[cy][cx] = (ch, f"bold {color}")

    if glyph_idx == 0:
        halo = [(-1, 0), (1, 0)]
        halo_glyph = _BUBBLE_GLYPHS[0]
    elif glyph_idx == 1:
        halo = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        halo_glyph = _BUBBLE_GLYPHS[0]
    else:
        halo = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0)]
        halo_glyph = _BUBBLE_GLYPHS[1]
    for dx, dy in halo:
        x, y = cx + dx, cy + dy
        if 0 <= x < w and 0 <= y < h and cells[y][x][0] == " ":
            cells[y][x] = (halo_glyph, color)


def _render_canvas(cells: list[tuple[str, str]], y_label: str, axis_char: str) -> Text:
    """One scatter row: right-aligned Y label + framed left axis + cells."""
    line = Text()
    line.append(f"{y_label:>4} ", style="dim")
    line.append(axis_char, style="dim")
    for ch, style in cells:
        line.append(ch, style=style or "dim")
    return line


def _consensus_block(
    forecast: NextCandleForecast,
    colors: dict[str, str],
    width: int = 22,
) -> Text:
    """Reinforced consensus bars — the headline read of the next candle.

    Les masses affichées ici sont désormais la source CALIBRÉE (modèle appris en
    Live, ligne de transition en mode stat) — plus la moyenne du nuage de bulles,
    qui diluait la calibration avec du bruit d'échantillonnage.
    """
    rows = (
        ("HAUSSE", forecast.prob_up, "up"),
        ("NEUTRE", forecast.prob_flat, "flat"),
        ("BAISSE", forecast.prob_down, "down"),
    )
    top_dir = max(rows, key=lambda r: r[1])[2]
    block = Text()
    block.append("  Consensus prochaine bougie", style="bold")
    tag = " 🧠 calibré" if forecast.model_driven else " 📊 stat"
    block.append(tag + "\n", style="dim")
    for label, mass, direction in rows:
        color = colors[direction]
        filled = int(round(mass * width))
        filled = max(0, min(width, filled))
        bar = "█" * filled + "░" * (width - filled)
        # Une courte tête (marge top-2 < 5 pts) n'est pas un appel assumé : le
        # marqueur le dit, au lieu de présenter un quasi-tirage comme « probable ».
        if direction != top_dir:
            marker = ""
        elif forecast.low_confidence:
            marker = f"◀ courte tête (Δ{forecast.consensus_margin:.0%})"
        else:
            marker = "◀ probable"
        block.append(f"  {_DIR_ARROW[direction]} {label} ", style=f"bold {color}")
        block.append(bar, style=color)
        block.append(f" {mass:4.0%} ", style=f"bold {color}")
        if marker:
            block.append(marker, style="bold" if marker == "◀ probable" else "dim")
        block.append("\n")
    return block


def format_bubble_chart(
    forecast: NextCandleForecast,
    *,
    styles: ThemeStyles | None = None,
    return_range: tuple[float, float] | None = None,
    live_eval=None,
) -> RenderableType:
    """Markov bubble scatter — theme colours, compact glyphs, frozen axes."""
    colors = _dir_colors(styles)
    accent = styles.accent if styles else "cyan"
    muted = styles.muted if styles else "dim"

    if not forecast.bubbles:
        return Text("Aucune bulle", style="dim")

    plot_w, plot_h = _BUBBLE_PLOT_W, _BUBBLE_PLOT_H
    cells: list[list[tuple[str, str]]] = [[(" ", "") for _ in range(plot_w)] for _ in range(plot_h)]

    if return_range is not None and return_range[1] > return_range[0]:
        lo, hi = return_range
    else:
        lo, hi = -_BUBBLE_RETURN_SPAN, _BUBBLE_RETURN_SPAN

    prob_ceil = _BUBBLE_PROB_CEIL                       # frozen Y domain
    max_p = max(b.probability for b in forecast.bubbles)  # glyph sizing only
    mp_id = forecast.most_probable.id

    zero_x = int((0.0 - lo) / (hi - lo) * (plot_w - 1))
    zero_x = max(0, min(plot_w - 1, zero_x))
    for y in range(plot_h):
        cells[y][zero_x] = ("┊", muted)

    top = forecast.bubbles[:12]
    coords: list[list[int]] = []
    for bubble in top:
        x = int((bubble.return_pct - lo) / (hi - lo) * (plot_w - 1))
        y = int((1.0 - min(bubble.probability, prob_ceil) / prob_ceil) * (plot_h - 2))
        coords.append([max(0, min(plot_w - 1, x)), max(0, min(plot_h - 2, y))])

    for _ in range(12):
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist2 = dx * dx + dy * dy
                if dist2 >= 9 or dist2 == 0:
                    continue
                if dx == 0 and dy == 0:
                    dx, dy = 1, 0
                push = 1
                sx = push if dx > 0 else -push if dx < 0 else 0
                sy = push if dy > 0 else -push if dy < 0 else push
                coords[i][0] = max(0, min(plot_w - 1, coords[i][0] + sx))
                coords[i][1] = max(0, min(plot_h - 2, coords[i][1] + sy))
                coords[j][0] = max(0, min(plot_w - 1, coords[j][0] - sx))
                coords[j][1] = max(0, min(plot_h - 2, coords[j][1] - sy))

    ordered = sorted(range(len(top)), key=lambda i: top[i].probability)
    for i in ordered:
        bubble = top[i]
        x, y = coords[i]
        gidx = _bubble_glyph_index(bubble.probability, max_p)
        _stamp_bubble(
            cells,
            x,
            y,
            gidx,
            colors[bubble.direction],
            is_star=bubble.id == mp_id,
        )

    for x in range(plot_w):
        cells[plot_h - 1][x] = ("─", "dim")

    # --- centred chart block (consensus block + scatter canvas + axes) ------
    chart = Text()
    chart.append(_consensus_block(forecast, colors))
    chart.append("\n")

    for row_idx in range(plot_h - 1):
        frac = 1.0 - row_idx / max(plot_h - 2, 1)
        labelled = row_idx % 3 == 0
        label = f"{frac * prob_ceil * 100:>3.0f}%" if labelled else ""
        axis_char = "┤" if labelled else "│"
        chart.append(_render_canvas(cells[row_idx], label, axis_char))
        chart.append("\n")

    base = Text()
    base.append(f"{'':>4} ", style="dim")
    base.append("┼", style="dim")
    baseline = ["─"] * plot_w
    if 0 <= zero_x < plot_w:
        baseline[zero_x] = "┴"
    base.append("".join(baseline), style="dim")
    chart.append(base)
    chart.append("\n")

    tick_vals = [lo, 0.0, hi] if lo < 0 < hi else [lo, (lo + hi) / 2, hi]
    tick_parts = []
    for tv in tick_vals:
        col = int((tv - lo) / (hi - lo) * (plot_w - 1))
        tick_parts.append((col, f"{tv:+.2%}"))
    tick_parts.sort(key=lambda t: t[0])
    ticks = Text()
    ticks.append("      ", style="dim")  # align under the framed plot area
    cursor = 0
    for col, lbl in tick_parts:
        col = max(cursor, min(plot_w - len(lbl), col))
        if col > cursor:
            ticks.append(" " * (col - cursor), style="dim")
        ticks.append(lbl, style=accent)
        cursor = col + len(lbl) + 1
    chart.append(ticks)
    chart.append("\n")
    chart.append("      ↞ baisse        rendement attendu        hausse ↠", style="dim")

    # --- details column (rendered to the RIGHT of the chart) ---------------
    details = Text()
    mp = forecast.most_probable
    details.append("★ ", style="bold")
    # « Scénario d'action », PAS « plus probable » : la bulle ★ est le pick
    # pondéré vraisemblance × risque — le « plus probable » directionnel est le
    # consensus calibré affiché au-dessus. Les deux doivent rester distincts.
    details.append(
        f"Scénario d'action  {mp.label_fr}  {mp.return_pct:+.2%}  ",
        style=f"bold {colors[mp.direction]}",
    )
    details.append(f"P={mp.probability:.1%}  →  ${mp.predicted_price:,.0f}  ")
    details.append("(pondéré risque)\n", style="dim")
    details.append(
        f"Markov {forecast.markov_from}→{forecast.markov_to} ({forecast.markov_transition_prob:.1%})  ·  "
        f"E[r]={forecast.expected_return:+.3%}  ·  backtest {forecast.backtest_hit_rate:.0%}\n",
        style="dim",
    )

    details.append("\n")
    details.append("Légende  ", style="bold")
    for direction, label in (("up", "hausse"), ("flat", "neutre"), ("down", "baisse")):
        details.append(f"{_BUBBLE_GLYPHS[1]} {label}  ", style=colors[direction])

    details.append("\n\n")
    details.append("Top scénarios\n", style="bold")
    for b in forecast.bubbles[:5]:
        g = _BUBBLE_GLYPHS[_bubble_glyph_index(b.probability, max_p)]
        star = "★" if b.id == mp_id else " "
        details.append(f" {star} ", style="dim")
        details.append(f"{g} ", style=f"bold {colors[b.direction]}")
        details.append(
            f"#{b.id} {b.return_pct:+.2%}  P={b.probability:.1%}  ${b.predicted_price:,.0f}\n"
        )

    # --- Live self-evaluation: running reliability of the chained predictions
    if live_eval is not None and getattr(live_eval, "n_eval", 0) > 0:
        details.append("\n")
        details.append("Fiabilité Live  ", style="bold")
        label, gstyle, _needs = live_eval.grade()
        details.append(
            f"{live_eval.reliability_note:.0f}/100  ", style=f"bold {gstyle}"
        )
        details.append(f"{label}\n", style=gstyle)
        details.append(
            f"  {live_eval.n_correct}/{live_eval.n_eval} bougies justes "
            f"({live_eval.accuracy:.0%}) · baseline {live_eval.baseline:.0%} · "
            f"erreur moy. {live_eval.mean_abs_error:.2%}\n",
            style="dim",
        )

    # Distribution-free conformal next-bar interval (shown as soon as a forecast
    # is pending, even during warm-up). Its realised coverage proves the stated
    # confidence is honest — the calibration layer's visible payoff.
    if live_eval is not None:
        cline = getattr(live_eval, "conformal_line", lambda: "")()
        if cline:
            details.append("\n")
            details.append("📐 ", style="bold")
            details.append(f"{cline}\n", style="dim")

    layout = Table.grid(padding=(0, 2), expand=True)
    layout.add_column(justify="left", vertical="top")   # bubble scatter
    layout.add_column(justify="left", vertical="top", ratio=1)  # details
    layout.add_row(chart, details)
    return layout


def bubble_chart_title(forecast: NextCandleForecast) -> str:
    """Border title for the bubble panel (carries the dynamic Live/candle tags)."""
    if not forecast.bubbles:
        return "🔮 Prochaine bougie"
    live = "  ·  🧠 LIVE" if forecast.model_driven else ""
    candle = ""
    if forecast.target_step >= 0:
        candle = f"  ·  🎯 #{forecast.target_step} {forecast.target_dt} UTC"
    return (
        f"🔮 Prochaine bougie — {forecast.timeframe}  ·  "
        f"${forecast.current_price:,.0f}{candle}{live}"
    )


# --------------------------------------------------------------------------- #
# Distribution des scénarios                                                  #
# --------------------------------------------------------------------------- #
def format_scenario_distribution(bundle: ScenarioBundle, styles: ThemeStyles | None = None) -> RenderableType:
    counts = bundle.direction_counts()
    masses = bundle.direction_probs()
    total = sum(counts.values()) or 1
    n = bundle.model.n_scenarios if bundle.model else len(bundle.scenarios)
    pos = styles.positive if styles else "green"
    neg = styles.negative if styles else "red"
    warn = styles.warning if styles else "yellow"

    body = Table.grid(padding=(0, 1))
    body.add_column()

    mp = bundle.most_probable
    if mp:
        # Action pick = risk-weighted SELECTION, not a directional forecast — so
        # its direction is shown dim/neutral, never colored as an up/down call
        # (the colored directional view is the equal-weight density below). §8.4
        body.add_row(
            f"[bold]★ Scénario d'action #{mp.id}[/]  [dim](pondéré risque)[/]  "
            f"[dim]{mp.direction} {mp.terminal_return:+.2%}  "
            f"DD={mp.max_drawdown:.1%}  P={mp.probability:.1%}[/]"
        )
        body.add_row(f"[dim]{bundle.verdict()}[/]")
        body.add_row("")

    body.add_row("[bold]Forecast directionnel (densité équipondérée)[/]")
    body.add_row(
        f"[{pos}]▲ Hausse[/]  {counts['up']:4d} ({counts['up']/total:.0%})  "
        f"[{pos}]{_ascii_bar(counts['up'], total)}[/]"
    )
    body.add_row(
        f"[{warn}]● Neutre[/] {counts['flat']:4d} ({counts['flat']/total:.0%})  "
        f"[{warn}]{_ascii_bar(counts['flat'], total)}[/]"
    )
    body.add_row(
        f"[{neg}]▼ Baisse[/] {counts['down']:4d} ({counts['down']/total:.0%})  "
        f"[{neg}]{_ascii_bar(counts['down'], total)}[/]"
    )
    qs = bundle.terminal_quantiles((0.05, 0.25, 0.5, 0.75, 0.95))
    body.add_row("")
    body.add_row(
        f"[bold]Quantiles terminal ({bundle.horizon} barres)[/]  "
        f"q05 [{neg}]{qs[0.05]:+.1%}[/] · q25 {qs[0.25]:+.1%} · "
        f"q50 {qs[0.5]:+.1%} · q75 {qs[0.75]:+.1%} · q95 [{pos}]{qs[0.95]:+.1%}[/]"
    )
    body.add_row(
        f"E[r]={bundle.expected_return:+.3%} [dim](log, martingale ⇒ ≈0)[/]  "
        f"n={n:,} scénarios"
    )

    returns = np.array([s.terminal_return for s in bundle.scenarios])
    probs = np.array([s.probability for s in bundle.scenarios])
    body.add_row("")
    body.add_row("[bold]Histogramme rendements[/]")
    for line in _histogram_bars(returns, probs, n_bins=6, width=16):
        body.add_row(line)

    return body


def format_top_scenarios(bundle: ScenarioBundle, n: int = 10, styles: ThemeStyles | None = None) -> Table:
    pos = styles.positive if styles else "green"
    neg = styles.negative if styles else "red"
    warn = styles.warning if styles else "yellow"
    t = Table(show_header=True, expand=True, box=None, header_style="bold")
    t.add_column("#", style="dim", width=6)
    t.add_column("Poids", justify="right", width=7)
    t.add_column("Ret.", justify="right", width=8)
    t.add_column("MaxDD", justify="right", width=8)
    t.add_column("Dir", width=6)
    t.add_column("Trajectoire", width=18)

    t.caption = "Poids = vraisemblance × pénalité de drawdown (pick de risque, pas un forecast)"
    t.caption_style = "dim italic"

    for s in bundle.top_n(n):
        style = {"up": pos, "down": neg, "flat": warn}[s.direction]
        spark = _sparkline(s.path, width=16)
        marker = ""
        if bundle.most_probable_id == s.id:
            marker += "★"
        if bundle.selected_id == s.id and bundle.selected_id != bundle.most_probable_id:
            marker += "◀"
        t.add_row(
            str(s.id) + marker,
            f"{s.probability:.1%}",
            f"[{style}]{s.terminal_return:+.2%}[/]",
            f"{s.max_drawdown:.2%}",
            f"[{style}]{s.direction}[/]",
            spark,
        )
    return t


# --------------------------------------------------------------------------- #
# Trades (jambes de rebalancement)                                            #
# --------------------------------------------------------------------------- #
def format_trades(bot: BotState, n: int = 8, styles: ThemeStyles | None = None) -> Table:
    pos = styles.positive if styles else "green"
    neg = styles.negative if styles else "red"
    t = Table(show_header=True, expand=False, box=None, header_style="bold dim", padding=(0, 1))
    t.add_column("Act", width=3)
    t.add_column("Prix", justify="right", width=7)
    t.add_column("→Expo", justify="right", width=6)
    t.add_column("Rdt", justify="right", width=7)

    t.caption_style = "dim italic"
    if getattr(bot, "predict_mode", False):
        t.caption = "Mode LIVE : le bot prédit — aucun trade"
    else:
        t.caption = "Trade = rebalancement vers l'exposition cible (budget de risque)"

    trades = bot.portfolio.trades[-n:]
    if not trades:
        t.add_row("—", "—", "—", "[dim]∅[/]")
        return t

    for tr in reversed(trades):
        # Le rendement réalisé n'existe que sur la fraction vendue (jambe SELL).
        if tr.action == Action.SELL and tr.pnl_pct:
            rdt_s = f"{tr.pnl_pct:+.1%}"
            rdt_style = pos if tr.pnl > 0 else neg if tr.pnl < 0 else "dim"
        else:
            rdt_s, rdt_style = "—", "dim"
        act_style = {"BUY": pos, "SELL": neg, "HOLD": "dim"}.get(tr.action.value, "")
        t.add_row(
            f"[{act_style}]{tr.action.value[:1]}[/]",
            f"{tr.price:,.0f}",
            f"{tr.exposure_after:.0%}",
            f"[{rdt_style}]{rdt_s}[/]",
        )
    return t


# --------------------------------------------------------------------------- #
# Honnêteté                                                                   #
# --------------------------------------------------------------------------- #
def format_honesty(
    bot: BotState,
    train_report=None,
    styles: ThemeStyles | None = None,
    online: dict | None = None,
    paper_verdict=None,
) -> RenderableType:
    """Honest 'what this model can and cannot do' panel.

    Surfaces the structural truth (1-bar direction ≈ 50/50; magnitude/régime/
    calibration are the real, measured signals), the trained model's out-of-
    sample calibration/edge, and — in Live mode — the running Wilson verdict.
    The bottom line states plainly what the bot's job is (risk budget) and what
    it is not (directional alpha). See ANALYSE_CRITIQUE_MODELE.md §6–8.
    """
    head = styles.heading if styles else "bold cyan"
    pos = styles.positive if styles else "green"
    neg = styles.negative if styles else "red"
    warn = styles.warning if styles else "yellow"
    muted = styles.muted if styles else "dim"

    t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    t.add_column("k", style=head, width=22, no_wrap=True)
    t.add_column("v", ratio=1)

    # -- Paper verdict (when a paper run has produced one) ---------------- #
    if paper_verdict is not None:
        vstyle = pos if paper_verdict.profitable else neg
        kind = "paper-trading live" if paper_verdict.forward else "backtest hors-éch."
        head_txt = Text(paper_verdict.verdict, style=f"bold {vstyle}")
        if paper_verdict.sig_tag:
            head_txt.append(f"  {paper_verdict.sig_tag}", style="dim")
        t.add_row("[bold]🧪 Rentabilité[/]", head_txt)
        sig = f" · {paper_verdict.significance_note}" if paper_verdict.significance_note else ""
        t.add_row(
            f"[{muted}]{kind}[/]",
            f"net [{vstyle}]{paper_verdict.ret_pct:+.2f}%[/] vs buy&hold "
            f"{paper_verdict.buy_hold_pct:+.2f}% · {paper_verdict.n_round} ventes · "
            f"réussite {paper_verdict.win_rate:.0%}{sig}",
        )

    # -- Layer 1: structural truth (always shown) ------------------------- #
    t.add_row("[bold]Job du bot[/]", "tenir la vol réalisée ≈ cible (budget de risque)")
    t.add_row(
        "Direction 1 barre",
        f"[{warn}]≈ 50/50 — pas d'edge exploitable[/] [{muted}](structurel, mesuré)[/]",
    )
    t.add_row(
        "Signaux réels",
        f"[{pos}]magnitude (rough-vol) · régime (HMM) · calibration (conforme)[/]",
    )

    # -- Layer 2: measured out-of-sample metrics -------------------------- #
    cm = getattr(bot, "candle_model", None)
    rep = train_report
    trained = bool(cm is not None and getattr(cm, "trained", False))
    if rep is not None and getattr(rep, "n_val", 0):
        val_acc = rep.val_accuracy
        val_maj = rep.val_majority
        edge = val_acc - val_maj
        gap = rep.train_accuracy - rep.val_accuracy
        edge_style = pos if edge > 0 else neg if edge < 0 else warn
        t.add_row("", "")
        t.add_row("[bold]Mesuré (holdout 20%)[/]", f"[{muted}]{rep.n_val} barres hors échantillon[/]")
        t.add_row(
            "Direction juste",
            f"{val_acc:.1%} vs classe maj. {val_maj:.1%} "
            f"([{edge_style}]edge {edge:+.1%}[/])",
        )
        gap_style = pos if abs(gap) <= 0.02 else warn
        t.add_row("Sur-apprentissage", f"[{gap_style}]train−val {gap:+.1%}[/]")
        if getattr(rep, "val_brier", 0.0):
            t.add_row(
                "Calibration (Brier)",
                f"{rep.val_brier:.3f} [{muted}]· T={rep.temperature:.2f}[/]",
            )
    elif trained:
        t.add_row("", "")
        t.add_row(
            "[bold]Modèle chargé[/]",
            f"val {cm.val_accuracy:.0%} [{muted}]· T={getattr(cm, 'temperature', 1.0):.2f} "
            f"· g pour ré-entraîner & mesurer[/]",
        )
    else:
        t.add_row("", "")
        t.add_row("[bold]Modèle[/]", f"[{muted}]non entraîné — touche g pour entraîner & mesurer[/]")

    _honesty_online(t, bot, online, pos, warn, muted)
    _honesty_live_and_verdict(t, bot, trained, rep, pos, neg, warn, muted)
    return t


def _honesty_online(t, bot, online, pos, warn, muted) -> None:
    """Append the Live self-reinforcement (online SGD) state, framed honestly:
    it tracks regime + keeps calibration, it does NOT create a directional edge."""
    if online is None or not online.get("trained") or not getattr(bot, "predict_mode", False):
        return
    cm = getattr(bot, "candle_model", None)
    cap = float(getattr(cm, "online_max_drift", 0.75))
    t.add_row("", "")
    if online.get("online_learn"):
        n_on = int(online.get("n_online", 0))
        t.add_row("[bold]Renforcement en ligne[/]", f"[{pos}]🔄 actif[/] [{muted}]· {n_on} maj (touche o)[/]")
        if n_on > 0:
            t.add_row(
                "Adaptation",
                f"NLL≈{online.get('online_nll', 0.0):.2f} · "
                f"dérive {online.get('online_drift', 0.0):.2f}/{cap:.2f} "
                f"[{muted}](région de confiance)[/]",
            )
        t.add_row(
            "Portée",
            f"[{muted}]suit le régime + calibration — [/][{warn}]ne crée pas d'edge directionnel[/]",
        )
    else:
        t.add_row("[bold]Renforcement en ligne[/]", f"[{muted}]inactif — poids figés (touche o)[/]")


def _honesty_live_and_verdict(t, bot, trained, rep, pos, neg, warn, muted) -> None:
    """Append the Live Wilson verdict (if running) and the honest bottom line."""
    ev = getattr(bot, "live_eval", None)
    if getattr(bot, "predict_mode", False) and ev is not None and ev.n_eval > 0:
        label, gstyle, _needs = ev.grade()
        t.add_row("", "")
        t.add_row("[bold]Live (Wilson)[/]", f"[{muted}]{ev.n_eval} bougies évaluées[/]")
        t.add_row(
            "Direction juste",
            f"{ev.accuracy:.0%} (IC95 ≥ {ev.acc_lower_bound:.0%}) "
            f"vs baseline {ev.baseline:.0%}",
        )
        t.add_row("Edge significatif", f"[{gstyle}]{label}[/]")
        if ev.brier_sum:
            t.add_row("Calibration (Brier)", f"{ev.mean_brier:.3f}")
        if ev.conformal.n_seen >= 20:
            t.add_row(
                "Couverture 90%",
                f"{ev.conformal.empirical_coverage():.0%} réalisée sur {ev.conformal.n_seen}",
            )

    # -- Bottom line: the one sentence the whole panel exists to make plain. #
    t.add_row("", "")
    if trained:
        verdict = (
            f"[{warn}]Calibré, pas rentable.[/] Les probabilités sont fiables "
            f"(« je ne sais pas » bien quantifié) mais l'edge directionnel 1 barre "
            f"reste ≤ 0. Le bot agit là où le signal existe : budget de risque "
            f"(σ̂, régime, coûts) — l'inclinaison directionnelle ne s'ouvre que "
            f"sur un edge Wilson prouvé."
        )
    else:
        verdict = (
            f"[{muted}]Entraîne le modèle (g) pour mesurer son edge réel. "
            f"Attends-toi à une direction calibrée mais sans edge à 1 barre — "
            f"c'est structurel, pas un bug. Le bot, lui, gère l'exposition "
            f"au risque (σ̂ rough-vol + régime HMM), net de coûts.[/]"
        )
    t.add_row("[bold]Verdict[/]", verdict)
