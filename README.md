# Probabilistic Trading Bot Simulator

Simulateur de bot de trading en terminal (Textual). À chaque barre, le bot
génère un **cône Monte-Carlo rough-vol Student-t** (100–10 000 trajectoires,
centré martingale), lit son **régime de volatilité HMM** et ses **intervalles
conformes**, puis une **politique de budget de risque** décide de l'exposition :

```
e* = clip(σ_cible / σ̂, 0, 1) × (1 − 0.5·P(turbulent)) × (1 + tilt·1[edge Wilson])
```

> **Honnêteté (l'ADN du projet).** La direction d'une bougie est ≈ 50/50 sur
> données réelles (démontré — `ANALYSE_CRITIQUE_MODELE.md`) : aucun modèle ici
> ne fabrique d'edge directionnel. Les signaux réels et mesurés sont la
> **magnitude** (rough-vol), le **régime** (HMM) et la **calibration**
> (conforme) — c'est là-dessus que le bot agit et se note. L'inclinaison
> directionnelle n'existe que derrière une porte Wilson (edge significatif hors
> échantillon), fermée par défaut. Voir `RAPPORT_REFONTE_V2.md`.

## Modèles (validés en walk-forward, `tests/measure_model.py`)

- **Volatilité rough** (RFSV, Gatheral-Jaisson-Rosenbaum 2018) — bat
  GARCH/HAR/EWMA en NLL+QLIKE sur BTC/ETH/SOL ; structure par terme avec
  correction lognormale **mesurée par horizon**
- **Cône Student-t** (ν fit par actif) martingale — couverture 95/99 % au
  nominal, quantiles terminaux affichés
- **Régime de volatilité HMM** (Baum-Welch, 3 états calm/normal/turbulent) —
  probabilités filtrées causales, consommées en continu par la décision
- **Calibration conforme + ACI** (Gibbs-Candès 2021) — intervalles next-bar à
  couverture garantie, couverture réalisée affichée
- **Modèle de bougie** (softmax calibré, température) — probabilités
  directionnelles *calibrées* (pas d'edge, et l'UI le dit) + renforcement en
  ligne borné (mode Live)
- **Politique de budget de risque** (`core/decision.py`) — vol-targeting lissé,
  dé-risquage de régime continu, bande de non-trade, frais + slippage comptés
- **Sauts de Hawkes** auto-excitants (clustering de queue) — OFF par défaut

## Installation & lancement

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                      # TUI (boot en TRAIN sur le cache)
python main.py --asset ETH/USDT
python main.py --paper              # source = prix Binance live (paper)
python main.py --cli --episodes 5 --steps 400   # replay headless (segments walk-forward)
python main.py --train --asset BTC/USDT --timeframe 1h  # entraîne le modèle de bougie
```

## Modes (touche `m`) × sources (touche `x`)

Deux axes **orthogonaux** :

| Axe | Valeurs | Sens |
|---|---|---|
| **Run mode** (`m`) | 🎓 TRAIN · 🧪 PAPER · 🔮 LIVE | ce que le bot FAIT : replay+budget de risque · verdict de rentabilité (gelé, net de coûts, significativité testée) · prédiction pure + auto-évaluation Wilson |
| **Source** (`x`) | 📊 cache · 📝 Binance paper · 🔴 Binance live | d'où viennent les prix. Aucun ordre réel n'est jamais passé (garde-fou `BINANCE_ALLOW_REAL_ORDERS`) |

## Interface

- **5 onglets** : Marché (prix + 🎯 Décision de risque + bulles + distribution) ·
  Scénarios · Modèles (pipeline complet avec formules) · Equity (vs buy & hold) ·
  Apprentissage (courbe de perte + panneau Honnêteté)
- **Bandeau statut** : exposition e→e*, σ̂ annualisé, régime coloré, source de
  données honnête (REPLAY vs 🔴 LIVE), verdict du mode courant
- **16 thèmes**, presets de layout, panneaux configurables (`p`)

## Raccourcis clavier (`h` pour la liste dans l'app)

| Touche | Action | Touche | Action |
|---|---|---|---|
| `espace` | pause/reprise | `g` | entraîner le modèle de bougie |
| `s` | avancer d'un pas | `m` | mode TRAIN → PAPER → LIVE |
| `a` | auto (boucle) | `o` | renforcement en ligne ON/OFF |
| `e` | replay d'un épisode (segment) | `c`/`f`/`x` | actif / timeframe / source |
| `r` | reset session | `n`/`d` | nb scénarios / vitesse |
| `1-5` | onglets | `b`/`l`/`p`/`t`/`v` | taille graph / layout / panneaux / thème / bougies-ligne |
| `q` | quitter | | |

## Architecture

```
core/market.py        données de marché (cache SQLite frais / synthétique)
data/loader.py        cache → ccxt → synthétique ; fraîcheur contrôlée, bougie en formation exclue
ml/rough_vol.py       σ̂ rough (RFSV) + structure par terme (ŝ²ₖ mesuré par horizon)
ml/vol_model.py       estimateurs de range, GARCH/HAR (le shoot-out du harnais)
core/cone.py          trajectoires Student-t martingale (échelle A/B-testable)
core/hmm.py           HMM gaussien de régime de vol (probabilités filtrées causales)
core/conformal.py     split-conformal + ACI (couverture garantie)
core/hawkes.py        sauts auto-excitants (OFF par défaut)
core/scenarios.py     assemble le cône + quantiles + forecast équipondéré
core/next_candle.py   forecast de la prochaine bougie (masses calibrées)
ml/candle_model.py    modèle de bougie (softmax calibré + online SGD ancré, queue OOS 15 %)
core/decision.py      politique de budget de risque (la couche de décision)
core/portfolio.py     rebalancement fractionnaire, frais + slippage, MtM
core/bot.py           orchestration par barre (générer → décider → rebalancer)
core/engine.py        SimulationSession — modes, sessions Live/Paper, snapshots
core/live_eval.py     auto-évaluation Live (Wilson, Brier, couverture, note refondée)
core/paper_eval.py    verdict de rentabilité qualifié (t de Student, minima)
ui/app.py             TUI Textual (onglets, raccourcis, refresh sélectif)
ui/panels.py          panneaux (décision, pipeline, honnêteté, métriques…)
tests/measure_model.py  harnais walk-forward (NLL/QLIKE/Brier/CRPS/couverture/risque)
tests/data_checks.py    Hurst (roughness) + stabilité des régimes
```

## Tests

```bash
# scripts autonomes (pas de pytest) — exécuter individuellement :
PYTHONPATH=. .venv/bin/python tests/smoke.py
PYTHONPATH=. .venv/bin/python tests/test_decision.py
PYTHONPATH=. .venv/bin/python tests/measure_model.py BTC/USDT 1h   # harnais complet
```

Les tests qui entraînent isolent `models/` (ils n'écrasent jamais vos poids).
