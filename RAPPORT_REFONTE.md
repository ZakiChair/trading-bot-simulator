# Refonte des modèles mathématiques — recherche récente, validée en walk-forward

**Date :** 22 juin 2026 · **Périmètre :** remplacer les couches *triviales* du bot
(génération de trajectoires, régime, calibration) par des modèles de recherche
récente, **chacun validé dans le harnais walk-forward existant** (`tests/measure_model.py`)
ou rapporté honnêtement comme n'apportant pas d'edge.

> **Règle de la refonte.** Tout nouveau modèle doit battre la baseline sur des
> règles de scoring propres (NLL / QLIKE / Brier / CRPS / couverture) en
> walk-forward sans fuite, sinon il est rapporté tel quel et **n'est pas activé
> par défaut**. C'est l'ADN anti-edge-fabriqué du projet (voir
> `ANALYSE_CRITIQUE_MODELE.md`). La tête de volatilité existante
> (RealizedGARCH/HAR, estimateurs de range) était déjà de niveau recherche et a
> été **conservée puis dépassée**, pas réécrite à l'aveugle.

---

## 0. Checks data qui ont conditionné la refonte (`tests/data_checks.py`)

Deux questions tranchées **avant** d'écrire le moindre modèle :

| Question | Mesure | Verdict |
|---|---|---|
| La vol est-elle *rough* ? | Hurst du log-RV (méthode structure-function ζ_q=q·H) : **H≈0.08 (BTC), 0.077 (ETH), 0.058 (SOL)** ; ζ_q linéaire en q (signature fBm) | **OUI, décisivement** — plus rough que les actions (H≈0.1) → rough-vol justifié |
| Les régimes sont-ils réels ? | Persistance vol-state 0.93 vs heuristique 0.63 ; séparation next-\|r\| ~1.8–2× | États de **volatilité** réels & persistants → HMM justifié (mais sur l'axe vol, pas direction) |

---

## 1. Cône Monte-Carlo : gaussien plat → **rough-vol + Student-t** ✅ ADOPTÉ

**Avant :** chocs gaussiens autour d'un σ *plat* avec jitter uniforme + un drift
de momentum.
**Après :** `core/cone.py` — innovations **Student-t standardisées à variance
unitaire** (ν fit par actif) dimensionnées par la **structure par terme rough-vol**
(`ml/rough_vol.py`, formule de prédiction fBm Nuzman-Poor / GJR 2018). Centre =
**martingale** (drift 0, préservé : la direction 1-barre est ≈50/50).

Validation walk-forward (échelle, `_cone_ladder_eval`, 12k barres) — couverture
des intervalles centraux, cible = niveau nominal :

| barreau | CRPS | Brier dir | cov 95 % | cov 99 % |
|---|---|---|---|---|
| gauss_flat | 0.0130 | 0.624 | 0.927 | 0.960 |
| gauss_rough | 0.0135 | 0.634 | 0.943 | 0.971 |
| **t_rough** ✅ | 0.0132 | **0.622** | 0.943 | **0.979** |

- **Rough term-structure** corrige la couverture du *corps* (95 %).
- **Student-t** corrige la queue *extrême* (99 % : 0.960→0.979 BTC, 0.939→0.961
  ETH, 0.956→0.971 SOL) sans dégrader CRPS ni Brier directionnel. ν médian ≈ 3–3.5
  (queues crypto réalistes).

**Tête σ par défaut = rough-vol** : bat RealizedGARCH sur NLL **et** QLIKE sur les
3 actifs (voir `ANALYSE_CRITIQUE §8.3` corrigé).

### ❌ Levier (effet d'asymétrie) — MESURÉ PUIS RETIRÉ
Le levier (corrélation négative rendement→vol) **régresse le Brier directionnel**
(BTC 0.622→0.632, ETH 0.616→0.625) — il ré-introduit un skew directionnel que le
projet avait justement supprimé (§8.4), et son estimateur est mal identifié
(collé au plafond). **Désactivé par défaut** (`leverage=0`).

---

## 2. Régime : heuristique codée en dur → **HMM gaussien de volatilité** ✅ ADOPTÉ (descriptif)

**Avant :** seuils en dur (`vol>0.025→volatile`, `mom>0.03→bull`) + matrices de
transition à la main. Dégénéré : ~93 % des barres étiquetées « range ».
**Après :** `core/hmm.py` — HMM gaussien (Baum-Welch, log-espace, numpy-only) à
**3 états de volatilité** (calm/normal/turbulent) sur log-RV lissé.

> **Itération honnête.** Un premier HMM 2-D `[return, log-RV]` *whipsaw*
> (persistance 0.39) : l'axe return à 1-barre est du bruit (la thèse du projet).
> Le bon modèle — celui validé par le data-check — est un HMM de **volatilité**.

Validation walk-forward (`_regime_eval`) :

| | persistance HMM/heur | η² next-\|r\| HMM/heur | équilibre |
|---|---|---|---|
| BTC | 0.85 / 0.95 | **0.0135 / 0.0066** | équilibré vs 93 % « range » |
| ETH | 0.86 / 0.92 | **0.0297 / 0.0236** | équilibré vs dégénéré |
| SOL | 0.82 / 0.86 | **0.0043 / 0.0039** | équilibré vs dégénéré |

**Verdict honnête :** le HMM sépare la magnitude next-bar **mieux que
l'heuristique sur les 3 actifs**, avec un découpage équilibré et une persistance
saine (la persistance supérieure de l'heuristique est *trompeuse* : dégénérée).
**MAIS ce n'est PAS un edge de forecasting** — le rough-vol possède déjà le
forecasting de vol (en continu, mieux qu'un régime discret). Le HMM est un régime
**descriptif/génératif** plus principié et plus informatif, surfacé dans l'UI
(`vol_regime`). Pas d'alpha revendiqué.

---

## 3. Calibration conforme ✅ ADOPTÉ — le gain le plus net

`core/conformal.py` — split-conformal normalisé (Lei et al.) + **Adaptive
Conformal Inference** (Gibbs & Candès 2021) pour la dérive. Intervalles de
rendement next-bar **distribution-free** : la couverture est garantie quelle que
soit la mauvaise spécification du modèle.

Validation walk-forward (`_conformal_eval`, 12k barres) — couverture empirique vs
nominale :

| niveau | conforme | gaussien |
|---|---|---|
| 0.50 | **0.501** | 0.577 (sur-couvre — corps épointé) |
| 0.90 | **0.899** | 0.914 |
| 0.95 | **0.951** | 0.939 |
| 0.99 | **0.989** | 0.972 (sous-couvre — queues) |

**Couverture empirique = nominale à ±0.001–0.002 à tous les niveaux, sur les 3
actifs.** Câblé dans la boucle Live (`core/live_eval.py`) en predict-before-update,
reset par session (pas de contamination inter-actif), μ=0 (martingale).

---

## 4. Sauts de Hawkes auto-excitants ✅ CONSTRUIT — ⚠ OFF par défaut

`core/hawkes.py` — processus de Hawkes à noyau exponentiel (MLE Ogata) pour le
**clustering temporel** des sauts (les crises s'auto-alimentent), que le
Student-t *iid* ne capture pas. Sauts à signe symétrique → martingale préservée.

Validation (barreau `t_rough_hawkes`) — **arbitrage, pas gain net** :
- ✅ **corrige la couverture 99 %** vers le nominal (BTC 0.979→0.997, ETH
  0.961→0.991, SOL 0.971→0.996) ;
- ❌ **dégrade légèrement CRPS et Brier** (les sauts sur-dispersent le corps) ;
- ⚠ **redondant avec le conformal** pour la couverture marginale (le conformal
  garantit déjà 99 %).

**Décision :** la valeur unique de Hawkes = modéliser la *structure* de clustering
(dépendance temporelle) pour le stress/risque de queue, pas le forecast central.
Fitté et disponible (`ConeParams(hawkes=True)`), **OFF par défaut** — pattern
établi du projet (« vrai modèle, bénéfice étroit, gardé mesurable »).

---

## Récapitulatif

| Couche | Modèle (recherche) | Statut | Preuve |
|---|---|---|---|
| Tête σ | Rough-vol RFSV (GJR 2018) | ✅ **défaut** | bat rgarch NLL+QLIKE ×3 actifs |
| Cône | Student-t + structure terme rough | ✅ **défaut** | couverture 99 % ✓, CRPS/Brier ✓ |
| Cône (levier) | effet d'asymétrie | ❌ **retiré** | régression Brier directionnel |
| Régime | HMM gaussien vol (Baum-Welch) | ✅ **descriptif** | η² > heuristique ×3, *pas* d'edge |
| Calibration | conforme + ACI (Gibbs-Candès 2021) | ✅ **défaut** | couverture = nominal ±0.002 |
| Sauts | Hawkes auto-excitant | ⚠ **OFF défaut** | corrige 99 % mais coûte CRPS |

**Invariant préservé :** la direction 1-barre reste ≈50/50 — aucun modèle ne
fabrique d'edge directionnel. La refonte améliore la **distribution** (queues,
structure par terme, calibration garantie) et le **régime descriptif** — là où il
y a une prévisibilité réelle — pas le signe du prochain rendement.

**Reproduire :** `PYTHONPATH=. .venv/bin/python tests/measure_model.py BTC/USDT 1h`
(+ `tests/data_checks.py` pour Hurst/régimes). Tests : `tests/test_{scenario_drift,
hmm,conformal,hawkes}.py`.
