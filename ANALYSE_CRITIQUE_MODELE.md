# Analyse critique du modèle mathématique — *notre bot de trading*

> Document d'analyse (18 juin 2026). Il décrit l'état **actuel** du code et ce
> qu'on pourrait améliorer. Aucune de ces remarques n'a été appliquée au code
> dans le même lot que les corrections d'interface — ce sont des
> recommandations à décider ensemble (voir §7 « Priorisation »).

## Verdict en une ligne

C'est un **excellent simulateur pédagogique**, mathématiquement honnête dans sa
plomberie (pas de look-ahead, vrai walk-forward, bornes de confiance de Wilson).
Ce **n'est pas** un outil de trading rentable — et la raison est structurelle,
pas un bug qu'on corrigerait : **les données par défaut sont un mouvement
brownien géométrique (GBM) à chocs indépendants, donc la direction d'une bougie
est imprédictible par construction.**

---

## 1. La racine de tout : le substrat est du GBM i.i.d.

`core/market.py::_generate_synthetic` (lignes 163-192) génère chaque barre ainsi :

```python
drift, vol = params[regime]          # ex. bull: (0.0004, 0.008)
shock = rng.normal(drift, vol)       # choc INDÉPENDANT de toute l'histoire
price *= np.exp(shock)
```

Conséquences mathématiques directes :

- Les rendements sont **indépendants** : `r_t ⟂ r_{t-1}`. Aucune autocorrélation
  au-delà du lent changement de régime.
- Le rapport **signal / bruit par barre** est minuscule :
  `drift_régime ≈ 0.0004` contre `σ ≈ 0.006–0.018`, soit un signal ≈ **0,02–0,05 σ**.
  Statistiquement indétectable à l'échelle d'une barre.
- **Donc aucun modèle** — linéaire, réseau de neurones, Markov, peu importe — ne
  peut avoir un *edge* directionnel réel sur ces données. L'« apprentissage » ne
  peut que chercher une structure qui n'existe pas. La fiabilité maximale
  atteignable est, par construction, **la baseline (classe majoritaire)**.

Le seul signal réellement exploitable est la **persistance de régime** (le drift
change lentement). C'est ce que le momentum capte un peu, mais il est noyé dans σ.

> C'est honnête pour un *simulateur*. Mais il faut le dire clairement : sur ce
> substrat, « entraîner plus » ou « ajouter de la capacité » ne peut pas créer
> de fiabilité.

## 2. La cible choisie est la moins prédictible possible

`ml/candle_model.py` prédit **hausse / neutre / baisse de la prochaine barre**.
Sur n'importe quel marché — y compris réel — c'est le rapport signal/bruit le
plus défavorable qui soit : la direction à 1 barre est ≈ 50/50. Le mémo projet
le note déjà : *le signal d'apprentissage honnête, c'est la courbe de perte, pas
l'accuracy brute.*

## 3. Le modèle appris est linéaire — mais ce n'est PAS le goulot d'étranglement

`CandleModel` (`ml/candle_model.py`) est une **régression logistique
multinomiale** (softmax linéaire) : frontières de décision linéaires uniquement.
On serait tenté d'ajouter de la capacité (réseau de neurones). **Inutile sur
données i.i.d.** : il n'y a rien à apprendre. L'ordre des priorités est donc :
d'abord le substrat et la cible (§6), la capacité du modèle vient loin derrière.

**Ce qui est rigoureux et à garder** dans ce module :
- `candle_features` est *look-ahead-free* et sert à la fois au dataset
  (`prices[:i+1]`) et à l'inférence live → **parité train/inférence** garantie.
- Split train/val **chronologique** (pas de shuffle sur série temporelle).
- Standardisation ajustée **sur le train seul**.
- Cross-entropy **pondérée** pour ne pas s'effondrer sur la classe « neutre ».

## 4. Incohérences concrètes — *lot (a) : rendre le simulateur plus honnête*

Ces points sont **corrigeables à faible coût** et **n'impliquent pas** que le
bot devienne rentable. Ils corrigent des incohérences internes.

1. **Seuil de direction incohérent à travers le pipeline.**
   `core/thresholds.py::adaptive_threshold` (k·σ, adaptatif par volatilité) est
   bien utilisé dans `candle_model`, `live_eval` et `next_candle`. **Mais** :
   - `core/markov_model.py::_classify_return` (ligne 37) → seuil **0.003 fixe** ;
   - `core/scenarios.py::_classify_returns` (ligne 95) → seuil **0.003 fixe**.

   Donc les **états de la chaîne de Markov** et la **classification des
   scénarios** n'utilisent pas la même définition de « hausse/baisse » que le
   modèle appris. Sur du 1m, presque tout tombe dans « neutre » ; sur du 1d, le
   seuil est trop serré. → **Uniformiser sur `adaptive_threshold`** (≈ 10 lignes,
   sûr).

2. **Échelle de récompense incohérente** (`core/bot.py`, ligne 125) :
   ```python
   reward = trade.pnl if trade.action == Action.SELL else realized * 100
   ```
   - Sur un **BUY** : `realized * 100` ≈ rendement de barre × 100 ≈ **±1 à ±2**.
   - Sur un **SELL** : `trade.pnl` ≈ PnL en **dollars** ≈ **±100 à ±300**.

   Comme le pas d'apprentissage est mis à l'échelle par `(1 + |reward|)`
   (`scenario_selector.py`, ligne 237), les mises à jour déclenchées par un SELL
   sont **~100× plus grosses** que celles d'un BUY → apprentissage déséquilibré.
   → normaliser la récompense (rendement, ou *R-multiples* en unités de σ).

3. **Pas de slippage ni de spread.** `core/portfolio.py` applique bien des frais
   (`fee_rate = 0.001`, soit 0,1 % par côté — correct) mais exécute **au prix
   exact de la barre**. Sur des ordres fréquents (1m) ou des actifs peu liquides,
   c'est optimiste. → ajouter un slippage proportionnel à σ.

4. **Métriques = direction seulement, jamais la calibration.** On suit
   `accuracy` / `edge` mais jamais la **calibration des probabilités** : quand le
   modèle annonce « 70 % », a-t-il raison 70 % du temps ? → ajouter le **score de
   Brier** et un **diagramme de fiabilité**. Une probabilité bien calibrée est
   plus utile qu'une accuracy un point au-dessus de la baseline.

5. **`autonomy_score` est cosmétique.** `ml/scenario_selector.py::update_autonomy`
   = `40 % accuracy + 30 % profit + 20 % sharpe + 10 % avancement`. C'est un
   indicateur de progression d'épisode, **pas** une mesure de performance hors
   échantillon. À ne jamais lire comme « le bot est fiable à X % ».

## 5. Ce qui est DÉJÀ bien fait (et qu'il ne faut surtout pas casser)

- `core/next_candle.py::_backtest_hit_rate` : **vrai walk-forward en ligne**
  (compte de transitions sur 0..i, prédit i+1 *avant* de l'apprendre). Pas de
  look-ahead.
- `core/live_eval.py` : `edge = accuracy − baseline`, **borne inférieure de
  Wilson**, et `significant_edge` qui exige que l'IC 95 % batte la baseline
  **avec n ≥ 25**. C'est exactement la rigueur statistique attendue — un 7/10
  chanceux ne passe pas pour du talent.
- `CandleModel.train_end_step` + `core/engine.py::_begin_live_session` :
  l'évaluation Live démarre **strictement après** les barres d'entraînement →
  fiabilité mesurée hors échantillon.
- `core/live_feed.py` : exclusion de la bougie en formation → pas de look-ahead
  en mode données réelles.

## 6. Ce qu'il faudrait pour une VRAIE puissance prédictive — *lot (b)*

> ⚠️ À ne pas confondre avec le lot (a). Les corrections (a) rendent le
> simulateur **honnête** ; elles ne le rendent **pas rentable**. Le lot (b) est
> un changement de cap.

1. **Un substrat avec structure apprenable**, ou — bien mieux — **entraîner et
   valider principalement sur des données RÉELLES** (le loader sait déjà charger
   du Binance/cache). Sans structure, le mot « fiabilité » n'a pas de sens.
2. **Changer la cible.** Plutôt que la direction à 1 barre :
   - la **volatilité** (très prédictible — clustering type GARCH) ;
   - le **régime** (bull/bear/range/volatile) ;
   - un **rendement multi-barres normalisé par σ**.
3. **Backtest walk-forward avec coûts comme métrique phare**, assorti d'un
   **intervalle de confiance** (Wilson / bootstrap) — c'est l'extension naturelle
   de ce que `live_eval` fait déjà, mais sur tout l'historique.
4. **Récompense nette de coûts** (frais + slippage) pour que la politique
   n'apprenne pas à sur-trader.
5. **Calibration** (Brier, reliability diagram) + régularisation ; éventuellement
   un petit ensemble.
6. **Validation multi-seed / multi-période** au lieu d'un unique holdout de 20 %.
7. **Message honnête « pas d'edge ».** Quand l'edge n'est pas significatif
   (`live_eval` le sait déjà via Wilson), l'UI devrait l'afficher clairement
   plutôt qu'une note flatteuse.

## 7. Priorisation

| Effort | Gain | Actions |
|---|---|---|
| **Rapide & sûr** (honnêteté) | moyen | seuil uniforme (§4.1), échelle de récompense (§4.2), score de Brier (§4.4) |
| **Moyen** | moyen | slippage (§4.3), backtest walk-forward à coûts + IC (§6.3) |
| **Structurel** (puissance réelle) | élevé | données réelles + cible volatilité/régime (§6.1–6.2) |

Le lot (a) — en particulier **le seuil uniforme** (~10 lignes) — peut être
implémenté tout de suite si tu le souhaites. Le lot (b) est une réorientation de
fond du projet, à décider ensemble.

---

## 8. Mesures — avant / après l'amélioration (18 juin 2026)

Harnais reproductible : `PYTHONPATH=. .venv/bin/python tests/measure_model.py
BTC/USDT 1h`. Toutes les mesures sont **hors échantillon** (holdout 20 %
chronologique + walk-forward à fenêtre expansive sans fuite). Données **réelles**
(`source: cache:BTC/USDT`, 745 barres 1h) — pas le substrat GBM synthétique, donc
les chiffres reflètent une vraie structure de marché.

### 8.1 Baseline AVANT (modèle linéaire, features brutes, 400 epochs fixes)

| Métrique (BTC/USDT 1h) | Valeur | Lecture |
|---|---|---|
| Accuracy entraînement | 0.484 | — |
| Accuracy validation | 0.433 | **sous** la classe majoritaire |
| Classe majoritaire (val) | 0.532 | baseline naïve « toujours neutre » |
| **Edge directionnel (holdout)** | **−0.099** | ❌ pire que la baseline (sur-apprentissage) |
| Edge directionnel (walk-forward) | −0.108 | ❌ confirme : pas d'edge 1-barre |
| Brier (holdout) | 0.6578 | probabilités mal calibrées |
| Brier (walk-forward) | 0.6426 | — |
| Vol forecast NLL — σ plat 20 | −3.8493 | tête de magnitude (GBM) |
| Vol forecast NLL — EWMA λ0.94 | −3.9237 | **+1.9 %** (meilleur, clustering) |

**Diagnostic confirmé** : la direction à 1 barre n'a pas d'edge exploitable
(edge ≤ 0 même hors échantillon), et le modèle linéaire à features brutes
**sur-apprend** (train 0.484 > val 0.433). Le gain honnête possible n'est donc
**pas** un edge directionnel miraculeux mais : (a) refermer l'écart de
sur-apprentissage → généralisation, (b) calibrer les probabilités → Brier, (c)
la **magnitude** (volatilité, structurellement prévisible via le clustering).

### 8.2 APRÈS (features v2 σ-normalisées, L2, early-stopping val-loss, temperature scaling)

Même harnais, mêmes données réelles (`cache:BTC/USDT`, 745 barres 1h, seuil
adaptatif `k·σ = 0.00206`). Mesures du **18 juin 2026**.

| Métrique (BTC/USDT 1h) | AVANT | APRÈS | Lecture |
|---|---|---|---|
| Accuracy entraînement | 0.484 | 0.477 | — |
| Accuracy validation | 0.433 | **0.475** | remonte vers la baseline |
| Classe majoritaire (val) | 0.532 | 0.532 | baseline naïve inchangée |
| **Écart train − val (sur-apprentissage)** | **+0.051** | **+0.002** | ✅ sur-apprentissage quasi éliminé |
| Edge directionnel (holdout) | −0.099 | **−0.057** | ✅ +0.042, mais toujours ≤ 0 |
| Edge directionnel (walk-forward) | −0.108 | **−0.042** | ✅ +0.066, mais toujours ≤ 0 |
| Brier (holdout) | 0.6578 | **0.6276** | ✅ −4.6 % (mieux calibré), T=0.95 |
| Brier (walk-forward) | 0.6426 | **0.6029** | ✅ −6.2 % (mieux calibré) |
| Vol forecast NLL — EWMA vs σ plat | +1.9 % | +1.9 % | tête de magnitude : seul edge **positif** |

**Lecture honnête.** Les deux gains prédits au §8.1 sont réalisés et mesurés :

1. **Sur-apprentissage refermé** — l'écart train−val passe de +0.051 à +0.002.
   Les features v2 sont *volatility-normalisées* (z-scores, log-ratios) donc
   stationnaires : les poids linéaires ne sont plus spécifiques d'un régime de
   volatilité. S'y ajoutent la régularisation L2 (`l2=2e-3`) et l'early-stopping
   sur la val-loss (patience 12 × 5 epochs) au lieu de 400 epochs fixes.
2. **Calibration améliorée** — Brier −4.6 % (holdout) / −6.2 % (walk-forward)
   grâce au *temperature scaling* post-hoc (T ajusté sur la val par minimisation
   de la NLL). Un « 70 % » annoncé est désormais plus proche de 70 % réels.

**Ce qui N'a PAS changé, et c'est honnête :** l'edge directionnel 1-barre reste
**négatif** (−0.057 holdout, −0.042 walk-forward). C'est conforme au §1–2 : sur
un actif réel comme sur le substrat GBM, la direction d'une seule barre est ≈ 50/50
et n'a pas d'edge exploitable. Refermer le sur-apprentissage et calibrer ne
*créent pas* d'edge — ils rendent les probabilités **fiables** (« je ne sais
pas » bien quantifié) au lieu de **faussement confiantes**. Le seul signal avec
edge positif reste la **magnitude** (EWMA σ, +1.9 % NLL vs σ plat), c'est-à-dire
le clustering de volatilité — la piste du lot (b) §6.2.

> En résumé : le modèle est maintenant *honnête et bien calibré*, pas *rentable*.
> C'est exactement la borne supérieure atteignable décrite au §1, et l'UI Live
> (note de fiabilité Wilson, `core/live_eval.py`) le reflète : quand l'edge n'est
> pas significatif, elle l'affiche.

---

## 8.3 Tête de volatilité — modèle appris (`ml/vol_model.py`, 18 juin 2026)

> **Ce qu'on améliore — et ce qu'on n'améliore pas.** L'edge directionnel
> 1-barre est plafonné à ≤ 0 par construction (§1-2, mesuré encore en §8.2). On
> ne *crée pas* d'edge directionnel ici. On exploite le **seul** signal mesuré
> positif : la **magnitude** (clustering de volatilité, §8.2 EWMA > σ-plat). Le
> gain est donc une **meilleure calibration du risque** — la largeur de chaque
> cône Monte-Carlo, des bandes et des probabilités est dimensionnée par un σ
> plus juste — **pas** un edge directionnel ni un PnL. C'est honnête : c'est la
> piste du lot (b) §6.2, et c'est la seule qui paie.

### Pourquoi ça compte dans CE bot

`core/scenarios.py` dimensionnait **tout** (spread des chemins, `path_vols`,
pondération de vraisemblance) par un seul σ = `state.ewma_volatility()`, un EWMA
RiskMetrics **sur les clôtures seules**. Or les données cache portent un OHLC
**réel** (vérifié : 96 % des barres ont des mèches hors corps, formule de
synthèse 0.35 rejetée). Le **range intra-barre haut-bas** est un estimateur de
variance 5-8× plus efficace qu'un seul rendement de clôture au carré — une
information que le modèle **ignorait**.

### Méthode — départage walk-forward, règles de scoring propres

Harnais `tests/measure_model.py::_vol_models_eval` : pour chaque barre de test,
chaque modèle prévoit σ_t à partir des barres **strictement avant t** (sans
fuite ; GARCH/HAR réajustés en fenêtre expansive). Trois métriques (plus bas =
mieux), sur le rendement de **clôture** réalisé (donc les modèles de range
n'ont **aucun** repas gratuit) :

- **NLL** gaussienne — vraisemblance prédictive, la métrique-titre ;
- **QLIKE** — robuste au proxy bruité r² (Patton 2011) ;
- **MAE** — erreur de bande moyenne (là où EWMA *perdait* contre σ-plat, §8.2).

### Résultats (250 barres de test, walk-forward sans fuite)

| Modèle (1h) | BTC NLL | BTC QLIKE | ETH NLL | ETH QLIKE | NLL moyen vs EWMA |
|---|---|---|---|---|---|
| σ-plat 20 | −3.8493 | 1.9445 | −3.2742 | 2.7273 | −4.0 % |
| **close-EWMA** (ancien) | −3.9237 | 1.7957 | −3.4887 | 2.2984 | — |
| Parkinson-EWMA | −3.9335 | 1.7761 | −3.5628 | 2.1501 | +1.15 % |
| Garman-Klass-EWMA | −3.9325 | 1.7780 | −3.5748 | 2.1262 | +1.35 % |
| GARCH(1,1) | **−3.9510** | **1.7410** | −3.5292 | 2.2174 | +0.95 % |
| Realized-GARCH | −3.9425 | 1.7581 | −3.6143 | 2.0471 | +2.05 % |
| HAR-RV (log) | −3.8716 | 1.8999 | −3.5139 | 2.2479 | −0.3 % |
| **Rough-vol (RFSV)** ✅ | **−4.045** | **1.692** | **−3.725** | **1.885** | **+2.9 % / +5.6 %** |

> **MISE À JOUR (refonte, 22 juin 2026) — le champion a changé.** Realized-GARCH
> n'est plus le défaut. La **volatilité rough** (`ml/rough_vol.py`, RFSV de
> Gatheral-Jaisson-Rosenbaum 2018) **bat RealizedGARCH sur NLL *et* QLIKE sur
> BTC/ETH/SOL** (BTC NLL −4.045 vs −3.995, QLIKE 1.69 vs 1.73 ; ETH +5.6 % NLL).
> C'est le paiement d'un fait mesuré : l'exposant de Hurst du log-RV est ≈ 0.07
> (`tests/data_checks.py`) — la vol est *décisivement rough*, plus que les actions
> (H≈0.1). Le noyau power-law continu de la formule de prédiction fBm (Nuzman-
> Poor) bat la cascade à 3 buckets de HAR et la décroissance géométrique β de
> GARCH. Le défaut est désormais `VolForecaster(method="rough")`.

**Lecture (historique).** Aucun modèle ne dominait partout — d'où le **départage
empirique**. GARCH gagnait sur BTC (mean-reversion de variance), les estimateurs
de range sur ETH. **Realized-GARCH** les réunissait (récursion GARCH sur la
variance de Parkinson) et était le plus robuste *à l'époque* (+2.05 % NLL moyen).
Le rough-vol l'a ensuite dépassé en généralisant le même principe (mémoire de la
variance) à un noyau continu — voir la mise à jour ci-dessus.

**Compromis honnête (NLL/QLIKE vs MAE).** Les modèles qui gagnent les *règles de
scoring propres* (NLL, QLIKE) dégradent légèrement la **MAE** (Realized-GARCH :
BTC 2.70 vs 2.65, ETH 4.75 vs 4.18 ×1e-3). C'est attendu et **voulu** : pour
dimensionner un *cône de risque* (une densité de probabilité), on veut la
queue calibrée — `P(gros mouvement)` juste — pas la bande moyenne minimale. HAR
gagne la MAE (2.25 / 3.52) mais perd la densité ; c'est le mauvais objectif pour
du sizing de risque. NLL+QLIKE sont les bonnes fonctions de perte ici.

### Câblage (faible rayon d'impact)

- `core/scenarios.py::ScenarioEngine` porte un `VolForecaster` **persistant**
  (`self._vol`) ; `generate()` remplace `state.ewma_volatility()` par
  `self._vol.sigma_for(state)`, et passe ce même σ à `NextCandlePredictor.predict`
  (`vol=`) pour que la tête de magnitude partage le σ amélioré.
- **Cache** : seul le grid-MLE est coûteux ; il est réajusté au plus tous les
  `refit_every=24` barres, la récursion O(n) (plafonnée à 400 barres) tourne à
  chaque tick. Coût mesuré : ~16 ms/tick médian, ~145 ms au réajustement
  (≪ 350 ms du tick TUI). L'engine ne lague pas.
- **`adaptive_threshold` n'est PAS touché** : il définit les labels hausse/
  neutre/baisse du pipeline *directionnel* (et est figé par `test_quant_fixes`).
  L'amélioration est strictement cantonnée aux **consommateurs de magnitude**
  (σ des cônes), pour que les métriques directionnelles du §8.2 restent
  inchangées et comparables.

### Vérité affichée

Tests : `tests/test_vol_model.py` (gate réel/synthétique, **absence de
look-ahead**, stationnarité α+β<1, cadence du cache, repli gracieux, et la
revendication empirique rgarch NLL ≤ close-EWMA sur données réelles). Les 10
fichiers de tests passent. **Bottom-line inchangée et honnête : calibré, pas
rentable** — mais désormais le *risque* est calibré avec un modèle qui bat
réellement l'ancien sur sa propre métrique, et non par décret.

## 8.4 Tête de drift/direction — le cône Monte-Carlo est désormais une martingale (18 juin 2026)

> **Ce qu'on corrige — et son périmètre.** La tête de **volatilité** (§8.3) a
> été validée OOS ; la tête de **drift/direction** du `ScenarioEngine` ne
> l'avait **jamais** été. Mesurée, elle injectait un **pari directionnel non
> soutenu par les données** — exactement le §1-2 (direction 1-barre ≈ 50/50)
> qui ressurgit dans le moteur de scénarios.

### Le harnais manquant

`tests/measure_model.py::_scenario_engine_eval` (nouveau) score la **distribution
terminale à h barres** du moteur, en walk-forward sans fuite (état construit sur
les barres `0..t` uniquement). Deux règles :

- **NLL gaussienne** du rendement terminal réalisé `R = P[t+h]/P[t]−1` sous la
  distribution de scénarios, **drift complet (μ = centre du cône) vs martingale
  (μ = 0) au σ identique** (la largeur du cône). σ fixe ⇒ on isole le *drift*.
  σ et μ sont les moments **équipondérés** des N tirages MC (chaque chemin est un
  tirage équiprobable du processus génératif ; la `probability` par scénario est
  une repondération a posteriori d'affichage, pas la densité prédictive — s'en
  servir comme σ le fait s'effondrer vers 0 et produit des NLL absurdes).
- **Brier directionnel** des masses (P↑,P●,P↓) vs la classe réalisée, pour les
  masses **softmax-pondérées** (ce que l'UI montre) et les **fréquences MC
  équipondérées** (densité générative brute), vs **climatologie** (fréquences
  empiriques courantes).

### Constat AVANT (drift actif) — mesuré à h=12, BTC/USDT & ETH/USDT 1h

| Métrique (h=12, walk-forward) | BTC | ETH | Lecture |
|---|---|---|---|
| NLL drift vs martingale (médiane) | −1.37 vs −3.18 (**−57 %**) | −1.85 vs −2.78 (**−34 %**) | ❌ le drift **dégrade** la densité |
| dir-Brier pondéré | 0.94 | 0.75 | ❌ pire que la climatologie 0.52 |
| dir-Brier équipondéré MC | 1.22 | 1.08 | ❌ cône directionnellement biaisé |
| sign-acc (E[r] vs réalisé, non-flat) | 0.46 | 0.28 | ≤ pile-ou-face (ETH anti-prédictif) |

Le drift `base_drift = mom·0.3 + regime_bias`, plus `REGIME_DRIFTS` (±0.0004/barre)
et `DIR_DRIFT_SCALE·σ` (±0.6σ) le long des chemins de Markov, est un pari
directionnel que la donnée **ne soutient pas** — il empire à la fois la densité
(NLL) et la direction (Brier).

### Correction (faible rayon) — `core/scenarios.py::_DRIFT_SHRINK = 0.0`

Un **unique bouton** `_DRIFT_SHRINK` multiplie le drift directionnel
(`base_drift + r_drifts + m_drifts`) ; à 0 le cône est centré sur le prix courant
(**martingale**) et dimensionné par la tête de volatilité validée (§8.3). Le même
bouton recentre la **vraisemblance** (`lik_center = _DRIFT_SHRINK·mom`) — sinon
les `prob_up/prob_down/expected_return` affichés héritaient du biais momentum via
la repondération. Le bruit de dispersion (moyenne nulle) et `path_vols` (largeur
du cône) sont **intacts**. Le drift n'est **pas supprimé** mais shrunk : la
machinerie Markov reste affichée, et le choix reste mesurable (le bouton à 1.0
ré-injecte la dérive — gelé par `tests/test_scenario_drift.py`).

### Constat APRÈS

| Métrique (h=12) | BTC | ETH |
|---|---|---|
| NLL drift vs martingale (médiane) | −3.21 vs −3.23 (**−0.6 %**, ≈ neutre) | −2.80 vs −2.81 (**−0.3 %**) |
| dir-Brier pondéré | 0.94 → **0.67** | 0.75 → **0.65** |
| dir-Brier équipondéré MC | 1.22 → **0.50** (< clim 0.52) | 1.08 → **0.60** |
| sign-acc | 0.46 → **0.58** | 0.28 → **0.51** |

La densité générative est désormais **calibrée comme une martingale** (le Brier
équipondéré bat la climatologie sur BTC). La direction n'est plus un pari : la
vue directionnelle du bot vient du **candle model calibré** (mode Live), pas d'un
drift codé en dur. Tests : `tests/test_scenario_drift.py` (martingale sous forte
tendance, bouton actif). Les 11 fichiers de tests passent.

### Séparation forecast / sélection — `_FORECAST_EQUAL_WEIGHT = True` (câblé)

Le dir-Brier **pondéré** (0.67/0.65) restait au-dessus de l'**équipondéré**
(0.50/0.60) et de la climatologie (0.52) : la **pénalité de drawdown**
`−0.3·(max_dd²)/σ²` sous-pondère mécaniquement les chemins baissiers (drawdown
plus grand) ⇒ un biais **haussier** résiduel dans `prob_up/prob_down/
expected_return`. La pénalité de drawdown est une **préférence de risque
légitime pour la *sélection*** de scénario, mais elle ne doit pas se faire passer
pour un **forecast directionnel**. D'où la séparation, bouton `_FORECAST_EQUAL_WEIGHT` :

- **forecast** (`prob_up/prob_down/expected_return`, `direction_probs`) =
  fréquences MC **équipondérées** (densité générative honnête) ;
- **sélection** (`probability`, `confidence`, scénario le plus probable, biais de
  politique) = masses **softmax pénalisées-drawdown** (conscientes du risque).

### Impact trading mesuré — `tests/measure_model.py::_policy_backtest_eval`

La policy (`ml/scenario_selector.py`) **trade** sur `prob_up/prob_down/margin/
E[r]`, donc le changement modifie le comportement. Backtest déterministe (exploration
0, moteur seedé ⇒ seules les masses diffèrent), net de frais 10 bps/côté :

| Masses (400/219 barres) | rdt % | trades | round-trips | win% | turnover |
|---|---|---|---|---|---|
| pondérées — BTC | −19.1 | **2** | 1 | 0 % | 0.005 |
| équipondérées — BTC | −14.6 | 80 | 40 | 42.5 % | 0.200 |
| pondérées — ETH | +5.5 | **2** | 1 | 100 % | 0.009 |
| équipondérées — ETH | −3.1 | 40 | 20 | 30 % | 0.183 |

**Lecture honnête.** Les masses pondérées ne faisaient pas que mal calibrer le
forecast : le biais haussier rendait `margin = prob_up − prob_down` **toujours
positif**, donc le logit SELL était **toujours bloqué par la gate** (`_apply_
confidence_gate`) ⇒ le bot **ne pouvait structurellement jamais vendre** : il
achetait une fois et tenait (buy-and-hold déguisé — ETH +5.5 % par chance en
marché haussier, BTC −19 % en tenant la baisse). L'équipondéré **restaure le
trading bilatéral** (il vend), mais sur un signal **sans edge directionnel** ça ne
fait que payer des frais ⇒ pas rentable non plus. **Aucune des deux n'est
rentable** : c'est la bottom-line « calibré, pas rentable » confirmée côté PnL. Le
gain réel est double : (1) forecast calibré (dir-Brier 0.50 vs 0.67), (2)
suppression d'un **défaut structurel** (incapacité de vendre) masqué en stratégie.

Ce que ça **expose** (piste suivante, §4.4) : la policy trade sur un signal
directionnel sans edge → elle devrait surtout **ne pas sur-trader**. Récompense
**nette de coûts** + pénalité de turnover, pour qu'elle apprenne l'abstention
quand l'edge n'est pas significatif (ce que `live_eval` sait déjà via Wilson).
Tests : `tests/test_scenario_drift.py` (forecast = équipondéré symétrique ;
sélection = `probability` toujours pénalisée-drawdown, tilt haussier conservé).
Les 11 fichiers passent.

### Cohérence d'affichage — la sélection ne doit pas se faire passer pour un forecast

Câbler l'équipondéré dans les *nombres* (`prob_up/prob_down`) ne suffisait pas :
le chemin de **sélection** (`most_probable`, bulle surlignée) reste pondéré-risque
donc à tilt haussier, et `verdict()` l'affichait « Scénario le plus probable —
**HAUSSE +2 %** » à côté d'un consensus désormais symétrique — réintroduisant le
biais directionnel dans un *autre* widget. Corrigé (`core/scenarios.py::verdict`,
`ui/panels.py`) : le **forecast directionnel** (densité équipondérée) est en tête
et coloré ; le scénario sélectionné est présenté comme **« scénario d'action
(pondéré vraisemblance × risque, DD …) »**, en gris, **jamais** sa direction
colorée comme un appel haussier/baissier. C'est le standard §6.7 (« l'UI doit dire
la vérité ») appliqué ici : sélection ≠ forecast, visuellement distincts.

## 8.5 Renforcement en ligne — le modèle de bougie apprend en Live (19 juin 2026)

> **Ce qu'on ajoute — et son périmètre.** La boucle d'auto-évaluation existait
> déjà (`core/live_eval.py` scorait chaque bougie prédit-vs-réel) mais elle était
> **ouverte** : elle mesurait, elle n'apprenait pas. On la **ferme** — chaque
> bougie clôturée en Live devient **un pas de descente de gradient** sur le modèle
> de bougie (`CandleModel.online_update`). Périmètre honnête : ça **suit le régime**
> (non-stationnarité) et garde la **calibration** vivante ; ça **ne crée pas**
> d'edge directionnel à 1 barre — ce plafond reste celui du §1-2 (≈ 50/50), comme
> sur GBM. C'est l'analogue, côté **direction**, du refit live de la tête de
> volatilité (§8.3) — sauf que la vol a un edge mesuré et la direction non.

### Garde-fous (sinon on ré-ouvre des bugs déjà corrigés)

- **Ancrage / région de confiance.** L'update est tiré vers les poids batch
  validés (`online_anchor`) et **projeté** dans une boule L2 relative
  (`online_max_drift = 0.75`) : le modèle Live reste un *voisin* du modèle qu'on a
  réellement validé, même sous un flux adverse à sens unique.
- **En mémoire uniquement, jamais persisté.** Persister les poids adaptés ferait
  démarrer la prochaine marche Live sur des barres déjà vues en ligne →
  résurrection du look-ahead non-OOS (red-flag #3). Chaque session Live repart des
  poids propres (`ensure_anchor` + `reset_online` dans `_begin_live_session`).
- **Features capturées à l'instant de la prédiction** (`features_asof`, stockées
  dans `PendingPrediction`) : l'update apprend sur *exactement* ce qu'il a prédit,
  pas un slice recomposé (zéro décalage / zéro fuite — *predict-before-update*).

### Méthode — A/B online vs figé, hors échantillon (`tests/measure_model.py online`)

Un modèle batch est entraîné, puis on parcourt la queue OOS (strictement après le
split d'entraînement). À chaque barre on score **le modèle figé ET une copie
online** sur la direction réalisée (Brier + NLL), **puis** on injecte la bougie
settled dans la copie online (1 pas SGD). Prédire-avant-mettre-à-jour ⇒ pas de
fuite. Balayage du pas `lr0` sur **BTC + ETH + SOL 1h** (multi-actifs = garde-fou
anti-surajustement d'un seul jeu).

### Résultats — Δ moyen (online − figé ; négatif = online meilleur)

| `lr0` | ΔBrier | ΔNLL | Δacc | sûreté par actif |
|------:|-------:|-----:|-----:|------------------|
| **0.003** | **−0.0046** | −0.0030 | +0.016 | aide/neutre sur **les 3** (BTC −.0025, ETH −.012, SOL +.0006) ; dérive 0.15–0.31 |
| 0.01  | −0.0029 | +0.0007 | +0.042 | ok |
| 0.03  | −0.0047 | −0.0033 | +0.009 | la moyenne masque **+0.015 (HURTS BTC)** |
| 0.05  | +0.0017 | +0.0099 | +0.005 | **nuisible** (chasse le bruit) |

**Décision : `online_lr0 = 0.003`** — le seul pas qui améliore (ou laisse neutre) la
calibration sur **les trois** actifs, avec une dérive douce (≪ région de confiance,
donc vraie adaptation de régime, pas du sur-ajustement de bruit). Les pas élevés
chassent le bruit par bougie et **dégradent** la calibration.

**Conclusion honnête.** Gain **petit mais réel** : ≈ **0.7 % de Brier** en mieux,
direction toujours ≈ 50/54 % (pas d'alpha). C'est une amélioration de **calibration
/ suivi de régime**, exactement ce que la structure (§1-2) autorise — pas un edge
directionnel. L'UI le dit : panneau Honnêteté (« suit le régime + calibration —
ne crée pas d'edge directionnel »), badge `🔄N` du bandeau, ligne « En ligne » des
Métriques ; la note de fiabilité Wilson reste « 🔴 faible » quand l'edge est ≤ 0.

**Câblage** : `ml/candle_model.py` (online_update / anchor / trust-region /
features_asof / reset), `core/engine.py::_predict_tick` + `_reinforce_online`
(boucle, touche `o` = ON/OFF), `core/live_eval.py` (`PendingPrediction.features`).
Tests : `tests/test_online_learning.py` (loss ↓, dérive bornée, parité features,
reset, intégration moteur, gardes). **Les 12 fichiers de tests passent.**
