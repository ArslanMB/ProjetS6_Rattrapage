# Pentago — Projet S6 Rattrapage

Implémentation du jeu Pentago (6×6, quadrants 3×3) avec interface Pygame et IA (Alpha-Beta, MCTS). Modes : PvP, PvA, IA-IA.

## Prérequis

* Python 3.10+
* `numpy`, `pygame`

## Lancement

À la racine du projet :

python main.py

## Commandes

* Placement : clic sur une case vide.
* Rotation : clic sur un bouton du quadrant (↺ anti-horaire / ↻ horaire) ou clic dans la moitié haute/basse du quadrant.
* Fin de partie : clic pour revenir au menu.

## Arborescence

PROJETS6_RATTRAPAGE/
├─ alphabeta_ia/
│  ├─ alpha_beta.py        # Minimax + Alpha-Beta, caches, timing
│  └─ opening_book.py      # Livre d’ouvertures
├─ core/
│  ├─ constants.py         # Constantes du jeu
│  └─ pentago_logic.py     # Règles, état, rotations, victoire
├─ gui/
│  └─ draw.py              # Rendu Pygame (grille, billes, boutons, HUD)
├─ mtcs_ia/
│  └─ pentago_mcts.py      # MCTS 
├─ main.py                 # Boucle de jeu, événements, modes
├─ self_play_stats.py      # Tournoi / statistiques (optionnel)
└─ README.md

## Réglages IA

* Profondeurs/budgets modifiables dans `main.py` (sections PvA / IA-IA).
* Heuristique et tables de transposition : `alphabeta_ia/alpha_beta.py`.
* Ouvertures : `alphabeta_ia/opening_book.py`.

## Problèmes connus

* Si Pygame ne s’installe pas : pip install "pygame>=2.5,<2.6".
* Si les clics ne tombent pas au bon endroit après modification visuelle, aligner tailles/offsets entre gui/draw.py et les calculs de position dans main.py.
