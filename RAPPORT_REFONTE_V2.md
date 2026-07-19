# Refonte V2 — modèle mathématique & interface (2026-07-02)

**Demande :** « le modèle semble très peu fonctionner » → revoir intégralement le
modèle mathématique et l'interface, après analyse profonde du projet.

**Méthode :** audit multi-agents en 6 angles (pile vol/distribution, pile
direction/policy, moteur/exécution, UI, reproduction empirique des rapports,
expérience utilisateur vécue), chaque défaut majeur vérifié adversarialement,
puis refonte implémentée et re-validée dans le harnais walk-forward
(`tests/measure_model.py`). Les 18 fichiers de tests passent.

---

## 1. Diagnostic — pourquoi « ça ne fonctionnait presque pas »

Le cœur *distributionnel* (rough-vol RFSV, cône Student-t, HMM, conformal) était
**correct et sans fuite** — vérifié numériquement. Le problème était triple :

### a) La couche de décision apprenait du bruit, puis ne faisait rien
* Les seuls signaux directionnels de la policy (`E[r]`, `P↑−P↓`) venaient d'un
  cône **martingale par construction** : la marge était du pur bruit
  d'échantillonnage Monte-Carlo (σ ∝ 1/√n). Les gates absolues (0.06/0.10)
  comparées à ce bruit ⇒ **0 trade à n ≥ 1000 scénarios**, et à n=100 un churn
  aléatoire qui perdait ses frais (−7.4 % mesuré).
* Le « biais appris par scénario » indexait des tirages MC **régénérés à chaque
  barre** — apprendre un biais par index n'a aucun sens ; sa règle +0.6/−0.4 à
  ~50 % de réussite créait même une dérive haussière mécanique.
* La récompense = rendement 1 barre : la cible dont le projet a prouvé (§1-2 de
  l'ANALYSE) qu'elle n'a pas d'edge. Aucun terme de coût nulle part.
* Le modèle de bougie entraîné (touche g) était **déconnecté du trading** :
  verdicts identiques au octet avec ou sans modèle, à 100 ou 10 000 scénarios.
* La couche Markov échantillonnait des chemins **sans lien** avec les prix
  qu'elle notait (terme de bruit dans la softmax + affichage trompeur).

### b) Des bugs d'intégration cassaient les têtes validées
* **Feed live sans OHLC** : les high/low Binance étaient jetés → σ̂ rough-vol
  passait sur le proxy r² avec une correction lognormale invalide → **σ̂ gonflé
  ×3.5** (variance ×13.8) dans le mode par défaut (paper-live).
* **Vraisemblance des scénarios** : un rendement 12 barres divisé par une
  variance 1 barre (×11.6 trop piquée) → le « scénario le plus probable »
  affiché était structurellement un chemin plat.
* **Plancher σ = 0.003 codé en dur** mordait sur ~32 % des barres BTC 1h
  (bulles, bandes conformes et sizing écrêtés arbitrairement).
* **Structure par terme rough** : `ŝ²·k^{2H}` faisait croître le bruit de mesure
  du proxy RV avec l'horizon → cône 12 barres sur-large (×1.54 réalisé).
* **Bougie en formation** dans le cache (mutée à chaque re-fetch) et cache
  **jamais rafraîchi** une fois ≥200 lignes → verdicts paper instables 🟠→✅→🔴
  à code identique.
* **Ordres réels atteignables** depuis TRAIN/PAPER quand la source était
  Binance LIVE (aucun garde-fou de run mode).
* Marche « OOS » Live/Paper démarrée **sur le split de validation** du modèle
  (early stopping + température choisis dessus) — pas vraiment hors échantillon.

### c) L'UI racontait une autre architecture et frustrait
* « Autonomie » cosmétique en titre, « cerveau » affichant des poids fabriqués,
  panneau Markov décoratif, note de fiabilité **plafonnée ~18/100 par
  construction** avec un conseil « ré-entraîne » impossible à satisfaire,
  épisodes CLI identiques au octet, boot par défaut sur un mode LIVE inerte.

---

## 2. Refonte mathématique

### 2.1 Nouvelle couche de décision : budget de risque (`core/decision.py`)
La policy à heuristiques/RL est **supprimée** (`ml/scenario_selector.py`,
`ml/features.py`). À la place, une politique déterministe pilotée par les têtes
*validées* :

```
e* = clip(σ_cible / σ̂_lissé, 0, 1) × (1 − 0.5·P(turbulent)) × (1 + tilt·1[edge Wilson])
```

* **σ̂** : rough-vol next-bar (gagnant du shoot-out NLL/QLIKE), **lissé EWMA**
  pour le sizing (le σ̂ brut, volontairement nerveux, faisait churner : 90→37
  trades/400 barres mesurés, coûts 341$→47$) ;
* **P(turbulent)** : probabilité **filtrée continue** du HMM (une porte binaire
  sur l'argmax sautait de ±40 % d'exposition à chaque bascule) ;
* **tilt directionnel fermé par défaut** : il ne s'ouvre que si l'évaluateur
  Wilson atteste un edge significatif hors échantillon — l'abstention
  directionnelle est codée, pas espérée ;
* **bande de non-trade ±10 %** avec exécution à mi-bande (tampon anti-rebond) ;
* exécution via un **portefeuille fractionnaire** (`core/portfolio.py` v2) :
  rebalancement partiel, frais 10 bps + **slippage ∝ σ̂**, frais d'achat inclus
  dans la base de coût, equity = mark-to-market partout.

Chaque décision expose sa **trace complète** (`DecisionTrace`) : σ̂, budget,
régime, porte Wilson, bande, coût estimé, raison — l'UI montre le *pourquoi*.

### 2.2 Corrections des têtes conservées
* feed live : **OHLC réels conservés** (σ̂ sur Parkinson comme validé) ;
* vraisemblance des scénarios à l'**échelle terminale** (Σ var_path) ;
* plancher σ → `SIGMA_FLOOR` (1e-4) partout ;
* structure par terme rough : **ŝ²ₖ mesuré par horizon** au fit (plus
  d'extrapolation k^{2H} du bruit de proxy) → cov95 du cône 0.948 (cible 0.95)
  et **meilleur CRPS de toute l'échelle** ;
* classification des rendements terminaux en **log** (cohérent avec le seuil
  adaptatif ; supprime le biais E[r] > 0 de Jensen) ;
* masses directionnelles affichées = la **source calibrée** (modèle/Markov),
  plus le nuage de bulles bruité ; matrice de transition estimée sur 500 barres
  (60 = bruit pur) ;
* conformal : la couverture juge l'**intervalle réellement émis** (stocké au
  moment de la prédiction) ; α_t de l'ACI affiché ; scoring en log-returns ;
* HMM : filtre causal borné à 1000 barres (coût O(T) par tick supprimé) ;
* Hawkes (OFF par défaut) : raffinement joint (α,β), tailles de sauts en unités
  de σ̂ (relatives au régime).

### 2.3 Honnêteté d'évaluation
* **Vraie queue OOS** : le modèle de bougie s'entraîne et se sélectionne sur les
  premiers 85 % ; Live/Paper démarrent sur la queue **jamais vue** ;
* **verdict paper qualifié** : test de Student sur les rendements par barre +
  minima (120 barres, 3 trades) — sinon « ⏳ échantillon insuffisant » ;
* cache avec **contrôle de fraîcheur** (re-fetch au-delà de 6 barres de retard,
  étiquette `cache-stale` si le réseau échoue) et **bougie en formation exclue** ;
* garde-fou ordres réels : jamais de client LIVE attaché au portefeuille +
  opt-in explicite `BINANCE_ALLOW_REAL_ORDERS=1` ;
* les tests qui entraînent isolent `models/` (ils écrasaient le modèle réel).

### 2.4 Validation (harnais complet, données fraîches 2026-07)
```
vol shoot-out    : rough NLL −3.758 / QLIKE 1.93 — toujours champion (+2.9 % vs EWMA)
cône t_rough     : cov95 0.948 (cible .95) · cov99 0.980 · CRPS 0.01241 (meilleur barreau)
conformal        : couverture 0.500/0.810/0.904/0.948/0.988 (cibles .5/.8/.9/.95/.99)
régime HMM       : η² next-|r| 0.018 vs 0.009 (heuristique) · équilibré
edge directionnel: holdout −0.044, WF +0.050 (n=120) → ≈ 0, inchangé (structurel)

politique de risque (400 DERNIÈRES barres = queue OOS, net de coûts, baissière) :
                   ret %   vol ann  Sharpe  maxDD
  bot              −9.09    34.1 %  −6.11   −12.12  (79 trades · 83 $ de coûts)
  buy & hold       −9.04    51.0 %  −4.07   −13.38
  expo constante   −6.51    36.8 %  −4.01    −9.79  (72 % = expo moy. du bot)
```
Lecture honnête (corrigée 2026-07-02 après-midi) : **aucun edge directionnel
revendiqué**. Le tableau initial de ce rapport (« le bot bat les deux
benchmarks ») était mesuré sur les barres ~100-500 du cache — la fenêtre où les
têtes ont été réglées ; le harnais démarre désormais sur la **queue jamais
touchée**. Sur celle-ci : le *tracking de vol* — LE job du bot — est tenu
(34,1 % réalisée vs 35 % cible, contre 51 % pour le buy & hold), le drawdown est
adouci vs B&H, mais le sizing dynamique **fait moins bien que l'exposition
constante équivalente** sur cette fenêtre (−9,09 % vs −6,51 %). Une fenêtre
unique ne prouve rien dans un sens comme dans l'autre ; la revendication
défendable reste : vol pilotée, coûts comptés, pas de sur-rendement promis.

---

## 3. Refonte de l'interface

* **🎯 Décision de risque** — nouveau panneau *en tête* de l'onglet Marché : la
  formule instanciée terme à terme (σ̂ → budget → régime → porte Wilson →
  bande), jauge d'exposition, sparkline 60 barres, et la raison de la dernière
  action. Remplace le « cerveau » à poids fabriqués.
* **Pipeline probabiliste** réécrit : σ̂ rough (H, ν) → cône (quantiles
  terminaux) → régime HMM → conformal (couverture réalisée, α_t) → modèle
  bougie (calibration) → décision. Le panneau Markov décoratif est supprimé ;
  la matrice de transition reste une statistique descriptive annotée
  « V≈0.09 — descriptif, pas un signal ».
* **Métriques** : exposition e→e*, σ̂ annualisé, budget de vol, vol réalisée
  (écart coloré), rotation, **coûts payés** ; « Autonomie », « Reward » et
  « Exploration » supprimés.
* **Note de fiabilité refondée** : couverture conforme (45 %) + calibration
  Brier vs climatologie (35 %) + bonus edge Wilson (20 %). Un modèle honnête
  bien calibré tourne à 55-75 au lieu d'un plafond structurel de ~18 ; le
  conseil « ré-entraîner » n'apparaît que quand la **calibration** se dégrade
  (là où ré-entraîner aide vraiment).
* **Equity vs buy & hold** sur la même échelle $ dans l'onglet Equity ; l'onglet
  Apprentissage sans modèle montre le suivi exposition/cible (plus de courbes
  précision/autonomie fabriquées).
* **Verdict paper** affiché avec sa significativité (t de Student) ou suspendu.
* Robustesse/latence : les panneaux lourds ne se redessinent que sur leur onglet
  actif ; un formatteur en erreur dégrade son panneau, plus tout l'écran ; les
  raccourcis ne traversent plus les modales ; la modale d'affichage applique
  vraiment « en direct » et Annuler restaure tout ; boot par défaut en TRAIN
  (actif) avec hint « m pour LIVE » ; `r` conserve actif/timeframe/politique.
* CLI : `--cli` déterministe sur cache par défaut, épisodes = **segments
  walk-forward** (PnL par fenêtre, plus de placebo identique) ; `--train`
  annonce l'edge honnêtement (plus de ✓ vert sous la baseline).

---

## 4. Ce qui n'a PAS changé (l'ADN)

* La direction 1 barre reste ≈ 50/50 — aucun modèle ne fabrique d'edge, aucun
  chiffre flatteur n'est réintroduit. Le panneau Honnêteté, les bornes Wilson,
  le walk-forward sans fuite et le harnais de mesure restent la loi.
* Rough-vol, cône Student-t martingale, HMM vol-3-états, conformal+ACI, seuil
  adaptatif quantile, renforcement en ligne ancré : conservés (et réparés là où
  l'intégration les trahissait).

## 5. Limites connues / honnêteté résiduelle

* Le §8.5 (gain du renforcement en ligne, ΔBrier −0.0046) **ne se reproduit
  plus** sur les données 2026-07 : ΔBrier ≈ −0.0007 (neutre, aide ETH, nuit
  marginalement à BTC). La fonctionnalité reste bornée par sa région de
  confiance et coupable (touche o) — à re-mesurer périodiquement.
* L'écart train−val du modèle de bougie sur ETH/SOL (+0.05/+0.07) est du
  *distribution shift* entre régimes, pas de la sur-capacité : le balayage L2
  (2e-3 → 5e-2) ne le bouge pas. Le panneau Honnêteté l'affiche.
* Le H≈0.06-0.08 du Hurst est **borné vers le bas** par le bruit du proxy RV
  (le vrai H est plausiblement 0.10-0.15) — le kernel power-law reste validé
  par le shoot-out, mais le chiffre n'est pas une constante physique.
* La supériorité du bot sur les benchmarks dans le tableau §2.4 vaut pour UNE
  fenêtre ; la revendication défendable est : vol pilotée, coûts contenus,
  drawdown adouci en turbulence, performance ≈ benchmark statique à exposition
  égale.

**Reproduire :** `PYTHONPATH=. .venv/bin/python tests/measure_model.py BTC/USDT 1h`
(section « politique de budget de risque ») · tests : `tests/test_decision.py`,
`tests/test_quant_fixes.py`, suite complète verte (19 fichiers).

---

## 6. Correctifs post-audit (2026-07-02, après-midi)

Un second audit runtime multi-agents (7 angles : TUI touche par touche, modes
moteur, feed live/paper, CLI, intégration décision/portefeuille, panneaux,
persistance) a reproduit **28 défauts résiduels**, tous corrigés le jour même
et verrouillés par `tests/test_v2_fixes.py` (16 tests). Les plus structurants :

**Intégrité & honnêteté des mesures**
* La suite de tests **écrasait le vrai `models/BTC_USDT_1h.npz`** avec un
  modèle-jouet « val 100 % » (66 barres d'un faux feed) — test isolé, override
  `BOT_MODELS_DIR` (le monkeypatch `_isolated_models()` reste prioritaire),
  et les 6 artefacts `.npz` ré-entraînés sur données réelles (val 31-42 %,
  edge ≈ 0, comme il se doit). Un `.npz` au schéma périmé est maintenant
  signalé au journal au lieu d'être ignoré en silence.
* Le tableau §2.4 était mesuré sur la **fenêtre de tuning** — harnais déplacé
  sur la queue OOS et chiffres corrigés (voir ci-dessus).
* 3296 lignes du cache SQLite (BTC/USDT 4h) avaient des timestamps en
  **secondes** (dates 1970) — migration en ms + garde d'échelle à l'écriture.

**Couche de décision**
* Le code clippait le **produit** au lieu du **ratio** : en vol basse la porte
  de régime était neutralisée (P(turbulent)=1 pouvait laisser e\*=100 %). Le
  code applique maintenant la formule documentée/affichée, et `formula_line()`
  reproduit exactement le calcul.
* L'état EWMA (σ̂ lissé, P(turbulent)) **fuyait à travers tous les
  ré-ancrages** (switch actif/timeframe/source, entrée paper/live, rewind) —
  `reset_state()` est câblé partout, comme le gotcha du CLAUDE.md l'exigeait.

**Mode PAPER (verdict de rentabilité)**
* Le bot paper héritait de la **dérive du renforcement en ligne** acquise en
  LIVE sur les mêmes bougies OOS que le backtest rejoue → il reçoit désormais
  une copie du modèle **sur ses poids d'ancre** (batch validés).
* En paper-trading forward (et TRAIN sur feed live), les fills s'exécutaient au
  close de la barre **précédente** (prix vieux d'une barre, mouvement encaissé
  rétroactivement) → nouveau pas « avancer-puis-décider »
  (`TradingBot.step_live_forward`) : exécution au close fraîchement clôturé.
* Le verdict s'affiche partout **avec** son qualificatif de significativité.

**Cohérence de l'affichage**
* Deux « plus probable » contradictoires (consensus calibré vs bulle ★ pondérée
  risque) : la prédiction scorée par la fiabilité Live, le bandeau et le
  journal lisent tous l'**argmax des masses calibrées** ; la bulle ★ est
  ré-étiquetée « scénario d'action (pondéré risque) ».
* En LIVE, le panneau Décision affichait « ⚖ rebalance … » (action jamais
  exécutée) → trace marquée « Live (aucun trade) — prendrait : … ».
* Le HMM de régime n'utilisait **jamais Parkinson** (`real_ohlc(None,…)`
  toujours faux) — les `opens` sont maintenant threadés jusqu'au filtre.
* La courbe Equity splicait deux portefeuilles après un aller-retour PAPER →
  historique segmenté par époque de bot.
* Badge modèle qualifié par n (`val 39 % (n=1274)`, ⚠ si n<300).

**Touches & CLI**
* `r` conserve actif/timeframe/source · `s` avance d'un pas pendant la pause ·
  `n` conserve le run mode (plus de retour TRAIN silencieux depuis LIVE) ·
  `x` vers LIVE sans clés → message clair, état de mode cohérent · le hint de
  boot dit la vérité (« m ×2 pour LIVE ») · l'avertissement « Live ignoré »
  atterrit dans le journal visible · timeframe persisté entre sessions ·
  `ui_settings.json` chargé champ par champ (un champ corrompu ne détruit plus
  tout ; tick_speed négatif refusé) · modale d'affichage : « Appliquer »
  n'écrase plus le layout custom par le preset périmé du Select.
* CLI : repli synthétique **bruyant** (`--cli`) et **refusé** (`--train`,
  exit 2) ; l'étiquette de source n'est plus avalée par le markup Rich ;
  `--timeframe` validé et réellement transmis à `--cli`.
