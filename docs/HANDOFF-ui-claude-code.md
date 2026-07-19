# Résumé de passation — Trading Bot Simulator UI

**Projet :** `/root/trading-bot-simulator`  
**Stack :** Python, Textual TUI, Rich, plotext  
**Lancer :** `cd /root/trading-bot-simulator && python main.py` (paper live par défaut)

---

## Contexte

L'utilisateur veut une interface terminal stable pour un simulateur de bot probabiliste (Monte Carlo + Markov). Plusieurs itérations ont modifié le layout ; l'utilisateur signale que **les corrections ne sont pas satisfaisantes en runtime** malgré des changements en code.

---

## Exigences utilisateur (acceptance criteria)

### 1. Layout Personnalisé → Full
- Dans le panneau contrôle (Select `sel-layout`) ou la modale (`p`), passer de **Personnalisé** à **Full** doit **afficher immédiatement tous les panneaux**.
- Les checkboxes du panneau contrôle doivent se synchroniser.
- Le preset doit être persisté dans `config/ui_settings.json`.

### 2. Bordures des onglets continues
- Les cadres des onglets (Marché, Scénarios, Modèles, Equity, Apprentissage) **ne doivent plus « sauter »** à chaque tick UI.
- Bordure **continue et cohérente** autour de la zone centrale — pas de double bordure décalée entre `#center` et les `TabPane`.

### 3. Affichage Marché (layout d'origine)

```
Onglet Marché :
  ┌─ Graphique PRIX (toujours visible en haut) ─┐
  │  mode boogie (braille) par défaut           │
  └─────────────────────────────────────────────┘
  ┌─ scroll ────────────────────────────────────┐
  │  Bulles Markov (prochaine bougie)           │
  │  Distribution scénarios                     │
  │  Cerveau du bot                             │
  └─────────────────────────────────────────────┘
```

- **`v`** = basculer graphique prix **boogie ↔ ligne** (pas bulles ↔ prix).
- Le graphique prix et les bulles doivent coexister, pas se remplacer.

### 4. Bulles Markov (style original)
- Bulles **petites et discrètes** : glyphes `· ○ ◉` — **pas** de gros `⬤`, pas de numéros, pas de gros disques.
- Style proche de la **première version** : bandeau consensus horizontal, scatter XY, top 5 scénarios en liste.
- **Couleurs dépendantes du thème actif** (`success` / `warning` / `error` / `primary` via `ThemeStyles`), pas de `bright_green` / `bright_red` hardcodés.

### 5. Données live (déjà demandé avant)
- Mode **Paper Binance** par défaut (`exchange_mode: "paper"`).
- Repli simulation si connexion échoue.

---

## État actuel du code (à vérifier en runtime)

| Sujet | Fichier(s) | Statut code |
|-------|-----------|-------------|
| Layout preset full | `config/ui_settings.py`, `ui/app.py`, `ui/control_panel.py`, `ui/display_modal.py` | `apply_preset()` corrigé + `refresh_layout_preset()` ajouté — **à valider en UI** |
| Graphique prix boogie/ligne | `ui/charts.py`, `ui/app.py` (`action_toggle_price_style`, touche `v`) | Implémenté — boogie par défaut |
| Bulles Markov subtiles + thème | `ui/panels.py` (`format_bubble_chart(forecast, styles=ThemeStyles)`) | Réécrit — **à valider visuellement** |
| Bordures onglets | `ui/app.tcss` (`#main-tabs` avec border unifiée) | Modifié — **utilisateur dit que c'est insuffisant** |
| Layout marché | `ui/app.py` compose | PriceChart en haut + bulles dans scroll — restauré |
| Tests unitaires | `tests/test_bubble_chart.py` | Passent (`apply_preset full`, style boogie) |

**Config actuelle** (`config/ui_settings.json`) : `layout_preset: "full"`, tous panels `true`, `price_chart_style: "boogie"`, `exchange_mode: "paper"`.

---

## Problèmes probablement encore ouverts

### A. Layout custom → full ne marche pas en UI

**Hypothèses à investiguer :**
1. Le `Select` Textual (`sel-layout`) ne déclenche pas `on_select_changed` correctement quand on passe de `custom` à `full`.
2. `event.value` pourrait ne pas être la clé attendue (`"full"` vs `"Full"`).
3. `_apply_panel_visibility()` ne cible pas tous les widgets (vérifier `PANEL_WIDGETS` dans `ui/app.py`).
4. Conflit : checkbox qui repasse le preset en `custom` juste après le changement de layout.
5. Le fichier `ui_settings.json` est rechargé/écrasé au mauvais moment.

**Test manuel :**
1. Lancer l'app, cocher/décocher un panneau → preset = `custom`.
2. Select layout → `Full`.
3. Vérifier que `scenario_heatmap`, `learning_chart`, `prob_models`, etc. apparaissent.
4. Vérifier `config/ui_settings.json` : `"layout_preset": "full"` et tous `"panels": { ... true }`.

### B. Bordures onglets discontinues

**Hypothèses :**
1. `_refresh_ui()` + `_restore_tab()` provoque encore des reflows.
2. `height: auto` sur `#scenario-bubbles`, `#scenario-dist` fait bouger le layout à chaque tick.
3. CSS Textual : sélecteurs `#main-tabs > Tabs` / `ContentSwitcher` peuvent ne pas matcher la structure DOM réelle de Textual (inspecter avec `textual console` ou devtools).
4. Double bordure `#center` + `#main-tabs` — peut-être supprimer l'une des deux.

**Piste CSS :**
- Hauteurs **fixes** pour le graphique prix (`11` lignes) et **max-height** pour les panneaux scroll.
- `scrollbar-gutter: stable` déjà présent — vérifier effet.
- Éviter `update()` sur les `Select` du control panel à chaque tick (déjà partiellement corrigé dans `control_panel.py`).

### C. Bulles / thème
- `format_bubble_chart` reçoit `ThemeStyles(self)` depuis `app.py` — vérifier que les couleurs changent bien quand on fait `t` (cycle thème).
- `border_style` du Panel utilise `styles.hex("primary")` — OK.

---

## Fichiers clés à modifier

```
trading-bot-simulator/
├── config/ui_settings.py      # Presets layout, price_chart_style, apply_preset()
├── config/ui_settings.json    # Persistance utilisateur
├── ui/app.py                  # Compose, _apply_panel_visibility, on_select_changed, _refresh_ui
├── ui/app.tcss                # Layout onglets, hauteurs fixes
├── ui/panels.py               # format_bubble_chart() + ThemeStyles
├── ui/charts.py               # PriceChart boogie/line toggle
├── ui/control_panel.py        # sel-layout, refresh_layout_preset(), sync_checkboxes()
├── ui/display_modal.py        # Modale affichage (p)
├── ui/colors.py               # ThemeStyles (success/error/warning/primary)
└── tests/test_bubble_chart.py # Tests layout + bulles
```

---

## Plan d'action recommandé

### Priorité 1 — Reproduire et fixer layout full
1. Lancer `python main.py` en terminal interactif.
2. Reproduire custom → full.
3. Ajouter logs temporaires dans `on_select_changed` (`sel-layout`) : `event.value`, `panels` après `apply_preset`.
4. Si le handler ne fire pas : utiliser `Select.Changed` explicite ou bouton dédié « Appliquer layout ».
5. S'assurer que `_apply_panel_visibility()` retire la classe `hidden` sur **tous** les IDs de `PANEL_WIDGETS`.

### Priorité 2 — Stabiliser bordures / layout
1. Inspecter structure Textual de `TabbedContent` (version installée dans `.venv`).
2. Ajuster `app.tcss` avec les bons sélecteurs enfants.
3. Fixer hauteurs : prix = 11, bulles max-height = 16, pas de `height: auto` qui varie tick par tick.
4. Réduire `_restore_tab()` si ça cause des sauts — ou ne restaurer que si focus n'est pas sur un onglet.

### Priorité 3 — Valider bulles + thème
1. Comparer rendu actuel vs description utilisateur.
2. Si trop « vulgaire » encore : réduire `_stamp_bubble` à un seul caractère central (pas de halo).
3. Tester avec thèmes `obsidian`, `bloomberg`, `matrix` — couleurs doivent suivre.

### Priorité 4 — Tests

```bash
cd /root/trading-bot-simulator
PYTHONPATH=. .venv/bin/python tests/test_bubble_chart.py
PYTHONPATH=. .venv/bin/python tests/test_position_and_forecast.py
```

Ajouter un test d'intégration Textual (pilot) pour `sel-layout` custom → full si possible.

---

## Raccourcis clavier

| Touche | Action |
|--------|--------|
| `v` | Boogie ↔ ligne (graphique prix) |
| `t` | Cycle thème |
| `p` | Modale affichage |
| `1-5` | Onglets |
| `a` | Auto train |

---

## Message utilisateur (résumé une ligne)

> Le layout Personnalisé→Full ne s'applique pas en UI, les bordures d'onglets sautent encore, et il faut garder le graphique prix en boogie (défaut) en haut avec les bulles Markov discrètes et colorées par thème en dessous — pas un swap bulles/prix.