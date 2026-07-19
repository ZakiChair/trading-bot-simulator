#!/usr/bin/env python3
"""Probabilistic Trading Bot Simulator — entry point.

Launch the interactive terminal UI::

    python main.py

Paper trading on live Binance prices::

    python main.py --paper

Adjust scenario count (100–10 000)::

    python main.py --scenarios 10000 --paper
"""
from __future__ import annotations

import argparse
import sys


def _resolve_exchange_mode(args: argparse.Namespace):
    from exchange.live_client import ExchangeMode

    if args.live:
        return ExchangeMode.LIVE
    if args.paper:
        return ExchangeMode.PAPER
    return ExchangeMode.SIMULATION


def _warn_if_synthetic(console, data_source: str) -> bool:
    """Bannière impossible à rater quand la session tourne sur du SYNTHÉTIQUE.

    Un actif inconnu (ou un réseau coupé sans cache) retombe sur des prix
    fabriqués — les résultats ont l'air réels (PnL, trades, vol) mais ne
    mesurent RIEN. L'ancien chemin ne l'affichait nulle part (exit 0, tableau
    normal). Retourne True si la source est synthétique.
    """
    if not data_source.startswith("synthetic"):
        return False
    from rich.markup import escape

    console.print(
        f"\n[bold red]⚠⚠ DONNÉES SYNTHÉTIQUES ⚠⚠[/] [red]source = "
        f"{escape(data_source)}[/]\n"
        "[red]Aucune donnée réelle pour cet actif/timeframe (actif inconnu ou "
        "réseau indisponible sans cache) : les prix sont générés aléatoirement. "
        "Tout PnL/verdict ci-dessous est dénué de sens sur un vrai marché.[/]\n"
    )
    return True


def run_cli(
    episodes: int,
    steps: int,
    n_scenarios: int,
    asset: str,
    exchange_mode,
    timeframe: str = "1h",
) -> int:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    from core.engine import SimulationSession

    console = Console()
    session = SimulationSession.create(
        n_scenarios=n_scenarios,
        asset=asset or "BTC/USDT",
        timeframe=timeframe,
        exchange_mode=exchange_mode,
    )

    console.print(
        f"\n[bold cyan]🤖 Training {episodes} épisodes × {steps} steps "
        f"({n_scenarios} scénarios · {session.asset} {session.timeframe} · "
        f"{exchange_mode.value} · source {escape(session.data_source)})[/]\n"
    )
    _warn_if_synthetic(console, session.data_source)

    results = []
    for ep in range(1, episodes + 1):
        session.bot.episode = ep
        stats = session.train_episode(steps=steps)
        results.append(
            (stats["episode"], stats["pnl"], stats["realized_vol"],
             stats["avg_exposure"], stats["trades"], stats["fees"])
        )
        console.print(
            f"  Épisode {stats['episode']:2d} | PnL {stats['pnl']:+8,.0f} | "
            f"vol réalisée {stats['realized_vol']:5.0%} (cible {stats['target_vol']:.0%}) | "
            f"expo moy. {stats['avg_exposure']:4.0%} | Trades {stats['trades']} | "
            f"coûts {stats['fees']:,.0f}$"
        )

    table = Table(title="Résumé replay (politique de budget de risque)")
    table.add_column("Épisode")
    table.add_column("PnL", justify="right")
    table.add_column("Vol réalisée", justify="right")
    table.add_column("Expo moy.", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Coûts $", justify="right")
    for row in results:
        table.add_row(
            str(row[0]),
            f"{row[1]:+,.0f}",
            f"{row[2]:.0%}",
            f"{row[3]:.0%}",
            str(row[4]),
            f"{row[5]:,.0f}",
        )
    console.print(table)

    console.print(
        "\n[dim]Rappel honnête : la direction 1 barre est ≈50/50 — le job du bot est de "
        "tenir le budget de risque (vol réalisée ≈ cible), pas de deviner le signe.[/]"
    )
    return 0


def run_train(
    n_scenarios: int,
    asset: str | None,
    timeframe: str,
    epochs: int,
) -> int:
    """Headless: train the next-candle model by gradient descent on history."""
    from rich.console import Console

    from core.engine import SimulationSession
    from exchange.live_client import ExchangeMode
    from ml.candle_model import model_path

    console = Console()
    session = SimulationSession.create(
        n_scenarios=n_scenarios,
        asset=asset or "BTC/USDT",
        timeframe=timeframe,
        exchange_mode=ExchangeMode.SIMULATION,  # train on historical data
    )
    from rich.markup import escape

    # ``[{data_source}]`` était avalé par le markup Rich (interprété comme un
    # tag de style) — précisément la seule ligne qui devait annoncer la source.
    console.print(
        f"\n[bold cyan]🧠 Entraînement modèle next-candle — {session.asset} "
        f"{session.timeframe} \\[{escape(session.data_source)}] · descente de gradient[/]\n"
    )
    if _warn_if_synthetic(console, session.data_source):
        # Refus net : entraîner et PERSISTER un .npz sur du bruit fabriqué
        # pollue models/ avec un artefact d'apparence légitime.
        console.print(
            "[bold red]Entraînement refusé sur données synthétiques.[/] "
            "Vérifie l'actif/timeframe (ou le réseau/cache) et relance."
        )
        return 2

    def progress(epoch: int, loss: float, val: float) -> None:
        if epoch % 50 == 0:
            console.print(f"  epoch {epoch:4d}  ·  perte {loss:.4f}  ·  val {val:.1%}")

    rep = session.train_model(epochs=epochs, lr=0.5, progress=progress)
    if not rep.n_train:
        console.print("[yellow]Historique insuffisant pour entraîner le modèle.[/]")
        return 1

    edge = rep.val_accuracy - rep.val_majority
    edge_style = "green" if edge > 0 else "yellow"
    console.print(
        f"\n[bold]Modèle entraîné par descente de gradient[/] — "
        f"{rep.n_train} barres train / {rep.n_val} val\n"
        f"  perte {rep.loss_history[0]:.3f} → {rep.final_loss:.3f}  "
        f"(la perte diminue = le modèle apprend)\n"
        f"  précision val [bold]{rep.val_accuracy:.1%}[/]  vs classe majoritaire "
        f"{rep.val_majority:.1%}  ([{edge_style}]edge {edge:+.1%}[/])\n"
        f"  calibration Brier {rep.val_brier:.3f}  ·  T={rep.temperature:.2f}\n"
        f"  classes (hausse/neutre/baisse): {rep.class_counts}\n"
        f"  poids → {model_path(session.asset, session.timeframe)}"
    )
    if edge <= 0:
        console.print(
            "[yellow]Verdict honnête : calibré, sans edge directionnel — c'est la borne "
            "structurelle de la direction à 1 barre (≈50/50), pas un défaut d'entraînement. "
            "La valeur du modèle est la CALIBRATION de ses probabilités (mode LIVE).[/]"
        )
    else:
        console.print(
            "[dim]Edge positif sur CE holdout — à confirmer en Live (Wilson) avant d'y croire.[/]"
        )
    return 0


def run_ui(n_scenarios: int, asset: str | None, exchange_mode, run_mode=None) -> int:
    from core.engine import SimulationSession
    from ui.app import BotSimulatorApp

    app = BotSimulatorApp(n_scenarios=n_scenarios, exchange_mode=exchange_mode)
    if asset and asset != app.ui_settings.asset:
        app.ui_settings.asset = asset
        # ``exchange_mode`` peut être None (= « préférence persistée ») : la
        # session reconstruite doit utiliser le mode RÉSOLU par l'app, pas None
        # (qui partait en LiveMarketFeed et crashait avant l'ouverture de l'UI).
        app.session = SimulationSession.create(
            n_scenarios=n_scenarios,
            asset=asset,
            timeframe=app.ui_settings.timeframe,
            exchange_mode=app.session.exchange_mode,
        )
    if run_mode is not None:
        # Behaviour axis requested explicitly — apply over the auto-boot default.
        try:
            app.session.set_run_mode(run_mode)
        except Exception:
            pass
    app.run()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probabilistic Trading Bot Simulator")
    parser.add_argument("--cli", action="store_true", help="Mode CLI sans interface")
    parser.add_argument("--episodes", type=int, default=5, help="Épisodes d'entraînement (CLI)")
    parser.add_argument("--steps", type=int, default=400, help="Steps par épisode (CLI)")
    parser.add_argument(
        "--scenarios",
        type=int,
        default=100,
        help="Nombre de scénarios Monte Carlo + Markov (100–10 000)",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Actif: BTC/USDT, ETH/USDT, SOL/USDT",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper trading sur prix Binance live (sans ordres réels)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Trading LIVE — requiert BINANCE_API_KEY et BINANCE_API_SECRET",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Entraîner le modèle next-candle par descente de gradient sur l'historique, puis quitter",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Epochs d'entraînement du modèle (avec --train)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe (1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d) — pour --train et --cli",
    )
    args = parser.parse_args()

    # Validation tôt et lisible : un timeframe malformé partait en double
    # traceback brut (le repli synthétique re-crashait dans timeframe_minutes).
    import re as _re

    if not _re.fullmatch(r"\d+[mhdw]", args.timeframe):
        parser.error(
            f"--timeframe invalide: {args.timeframe!r} (attendu: nombre + unité "
            f"m/h/d/w, ex. 15m, 1h, 4h, 1d)"
        )

    if args.paper and args.live:
        print("⚠ --paper et --live sont mutuellement exclusifs — mode LIVE retenu")
        args.paper = False

    n_scenarios = max(100, min(10_000, args.scenarios))
    if n_scenarios != args.scenarios:
        print(f"⚠ Scénarios bornés à {n_scenarios} (plage 100–10 000)")

    explicit_mode = args.paper or args.live
    exchange_mode = _resolve_exchange_mode(args) if explicit_mode else None

    if args.train:
        return run_train(n_scenarios, args.asset, args.timeframe, args.epochs)

    if args.cli:
        from exchange.live_client import ExchangeMode

        # Headless replay: cache local par défaut (déterministe, pas de réseau) ;
        # --paper/--live restent disponibles explicitement. Le timeframe est
        # transmis (il était silencieusement ignoré — replay toujours en 1h).
        mode = exchange_mode or ExchangeMode.SIMULATION
        return run_cli(
            args.episodes, args.steps, n_scenarios, args.asset, mode,
            timeframe=args.timeframe,
        )
    return run_ui(n_scenarios, args.asset, exchange_mode)


if __name__ == "__main__":
    sys.exit(main())