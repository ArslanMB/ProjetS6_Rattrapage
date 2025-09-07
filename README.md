#  Pentago IA - Implémentation avancée avec agents intelligents

##  Vue d'ensemble

Implémentation complète du jeu de stratégie **Pentago** avec interface graphique PyGame et plusieurs agents d'intelligence artificielle. Le projet intègre des algorithmes de recherche avancés (Alpha-Beta, MCTS) avec optimisations de performance et système d'évaluation statistique complet.

### Caractéristiques principales
-  Interface graphique interactive avec animations de rotation fluides
-  5 types d'agents IA avec niveaux de difficulté variés
-  Système d'analyse statistique pour comparaison d'agents
-  Optimisations avancées (caches, tables de transposition, livre d'ouvertures)
-  Suite de tests et benchmarks pour évaluation de performance

##  Architecture du projet

```
pentago/
├── main.py                      # Point d'entrée principal avec menu interactif
├── core/                        # Logique de jeu centrale
│   ├── pentago_logic.py        # Mécanique du jeu et règles
│   └── constants.py             # Constantes globales
├── gui/                         # Interface utilisateur
│   └── draw.py                  # Rendu graphique et animations
├── alphabeta_ia/                # Agent Minimax/Alpha-Beta
│   ├── alpha_beta.py            # Implémentation avec optimisations avancées
│   └── opening_book.py          # Livre d'ouvertures
├── mtcs_ia/                     # Agent Monte Carlo Tree Search
│   ├── mcts.py                  # MCTS standard
│   ├── mcts_fast.py             # MCTS optimisé haute performance
│   ├── mcts_evaluation.py      # Tests unitaires MCTS
│   └── mcts_evaluation_fast.py # Benchmarks MCTS rapide
├── greedy_ai/                   # Agent glouton
│   └── greedy_ai.py             # Stratégie gloutonne avec heuristiques
├── hybrid_ia/                   # Agent hybride
│   └── hybrid_ai.py             # Combinaison MCTS + Alpha-Beta
├── analysis/                    # Outils d'analyse
│   ├── statistical_analysis.py # Analyse statistique complète
│   ├── self_play_stats.py      # Auto-jeu et collecte de données
│   └── records.txt              # Historique des matchs
└── README.md                    # Documentation

```

##  Règles du Pentago

Le Pentago se joue sur un plateau 6×6 divisé en 4 quadrants 3×3. À chaque tour :
1. **Placement** : Le joueur place une bille sur une case vide
2. **Rotation** : Le joueur tourne un quadrant de 90° (horaire ou anti-horaire)
3. **Victoire** : Aligner 5 billes de sa couleur (horizontal, vertical ou diagonal)

##  Installation

### Prérequis
- Python 3.10+
- NumPy
- Pygame
- SciPy (pour analyses statistiques)

### Installation des dépendances
```bash
pip install numpy pygame scipy
```

Si problème avec Pygame :
```bash
pip install "pygame>=2.5,<2.6"
```

##  Utilisation

### Lancement du jeu
```bash
python main.py
```

### Modes de jeu disponibles
- **Joueur vs Joueur** : Mode classique à deux joueurs locaux
- **Joueur vs IA** : Affrontez l'IA Alpha-Beta adaptative
- **Joueur vs MCTS** : Défiez l'agent Monte Carlo Tree Search
- **IA vs IA** : Observez deux agents s'affronter

### Contrôles
- **Placement** : Clic sur une case vide
- **Rotation** : 
  - Boutons ↺/↻ sur chaque quadrant
  - Ou clic dans la moitié haute/basse du quadrant
- **Fin de partie** : Clic pour revenir au menu

##  Agents IA

### 1. Alpha-Beta Minimax (`alphabeta_ia/`)
Agent basé sur l'algorithme Minimax avec élagage Alpha-Beta et optimisations multiples :

**Caractéristiques :**
- **Profondeur adaptative** : Ajustement automatique selon la complexité (2-5 niveaux)
- **Tables de transposition** : Cache des positions évaluées
- **Livre d'ouvertures** : Coups pré-calculés pour début de partie
- **Ordonnancement de coups** : Killer moves + table d'historique
- **PVS (Principal Variation Search)** : Recherche optimisée
- **LMR (Late Move Reduction)** : Réduction sélective de profondeur
- **Futility Pruning** : Élagage des branches peu prometteuses

**Performances :**
- ~10,000 nœuds/seconde en profondeur 3
- Temps de réponse configurable (0.5s - 10s)
- Taux de victoire >95% contre joueur aléatoire

### 2. MCTS (`mtcs_ia/`)
Monte Carlo Tree Search avec deux implémentations :

**MCTS Standard (`mcts.py`) :**
- UCT classique avec exploration/exploitation
- Simulations aléatoires jusqu'à position terminale
- ~100-200 simulations/seconde

**MCTS Fast (`mcts_fast.py`) :**
- Version optimisée haute performance
- Cache de rotations de quadrants
- Réduction de l'espace de recherche selon phase de jeu
- Détection tactique complète (win-in-1, blocages)
- ~500-1000 simulations/seconde

### 3. Greedy (`greedy_ai/`)
Agent glouton avec évaluation heuristique :
- Détection immédiate de victoires/blocages
- Évaluation basée sur fenêtres de 5 cases
- Bonus pour positions centrales
- Temps de réponse < 0.1s

### 4. Hybrid (`hybrid_ia/`)
Combinaison MCTS + Alpha-Beta :
1. MCTS explore les K meilleurs coups (60% du temps)
2. Alpha-Beta évalue en profondeur chaque candidat (40% du temps)
3. Sélection du meilleur coup selon évaluation finale

##  Système d'analyse

### Analyse statistique (`statistical_analysis.py`)
Framework complet pour évaluation comparative :

```python
from statistical_analysis import run_statistical_series

# Comparer deux agents
agent1 = MinimaxAgent(depth=3, time_budget=2.5)
agent2 = MCTSAgent(time_limit=2.5, exploration_constant=1.41)
results = run_statistical_series(agent1, agent2, games=100)
```

**Métriques calculées :**
- Taux de victoire avec intervalles de confiance (Wilson)
- Tests statistiques (binomial, p-value)
- Taille d'effet (Cohen's h)
- Statistiques temporelles (avg/med/p95/max)
- Analyse des parties décisives

### Auto-jeu et collecte (`self_play_stats.py`)
Système automatisé pour tournois et collecte de données :
- Alternance automatique des premiers joueurs
- Enregistrement détaillé des coups et temps
- Sauvegarde persistante des résultats

### Évaluation MCTS (`mcts_evaluation_fast.py`)
Suite de tests complète pour validation :
- Tests unitaires (règles, backpropagation)
- Tests tactiques (win-in-1, blocages)
- Benchmarks de performance
- Tests de robustesse
- Génération de rapport avec recommandations

##  Configuration avancée

### Ajustement des IA

**Alpha-Beta :**
```python
# main.py - ligne ~30
best_move, dt = timed_find_best_move_minimax(
    game, 
    depth="A",        # "A" pour adaptatif, ou 1-5 pour fixe
    time_budget=10,   # Temps max en secondes
    BOOKING=True      # Activer livre d'ouvertures
)
```

**MCTS :**
```python
# main.py - ligne ~25
mcts_bot = MCTS_Fast(
    time_limit=10.0,           # Temps de réflexion
    exploration_constant=0.7   # Balance exploration/exploitation
)
```

### Paramètres visuels
Modifiables dans `gui/draw.py` :
- `SQUARE_SIZE` : Taille des cases (défaut: 110px)
- `ROTATION_SPEED` : Vitesse d'animation (défaut: 5°/frame)
- `CIRCLE_RADIUS` : Taille des billes

##  Problèmes connus et solutions

1. **Pygame ne s'installe pas :**
   ```bash
   pip install --upgrade pip
   pip install "pygame>=2.5,<2.6"
   ```

2. **Clics décalés après modification visuelle :**
   - Vérifier l'alignement entre `BOARD_OFFSET_X/Y` dans `draw.py`
   - Synchroniser avec calculs de position dans `main.py`

3. **Performance lente de MCTS :**
   - Réduire `time_limit` ou `iteration_limit`
   - Utiliser `MCTS_Fast` au lieu de `MCTS` standard
   - Désactiver les prints de debug

4. **Mémoire excessive avec caches :**
   - Ajuster `CACHE_MAX` dans `alpha_beta.py`
   - Implémenter nettoyage périodique des caches

##  Tests et développement

### Lancer les tests unitaires
```bash
python mcts_evaluation_fast.py  # Tests MCTS
python statistical_analysis.py  # Tests comparatifs
```

### Ajouter un nouvel agent
1. Créer classe héritant de `AgentBase`
2. Implémenter `choose_move(game) -> (move, time)`
3. Intégrer dans `main.py` ou scripts d'analyse

### Profiling performance
```python
import cProfile
cProfile.run('run_statistical_series(agent1, agent2, games=10)')
```

##  Références et ressources

- [Règles officielles Pentago](https://www.ultraboardgames.com/pentago/game-rules.php)
- [MCTS Survey (Browne et al.)](https://www.cs.swarthmore.edu/~bryce/cs63/s16/reading/mcts.pdf)
- [Alpha-Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [Transposition Tables](https://www.chessprogramming.org/Transposition_Table)

## Crédits

Projet réalisé dans le cadre d' un projet IA - CentraleSupélec

##  Licence

Ce projet est à des fins éducatives. Les contributions sont bienvenues via pull requests.