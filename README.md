# Pentago — Projet S6 Rattrapage

Implémentation du jeu Pentago (6×6, quadrants 3×3) avec interface Pygame, IA (Alpha-Beta, MCTS, Glouton) et script pour tournois d'IA . Modes : PvP, PvIA, IA-IA.

## Prérequis

* Python 3.10+
* `numpy`, `pygame`

## Lancement

À la racine du projet :

python main.py

ou pour les tournois d'IA :

python self_play_stats.py

## Commandes

* Placement : clic sur une case vide.
* Rotation : clic sur un bouton du quadrant (↺ anti-horaire / ↻ horaire) ou clic dans la moitié haute/basse du quadrant.
* Fin de partie : clic pour revenir au menu.

## Réglages IA

* Profondeurs/budgets modifiables dans `main.py` (sections PvA / IA-IA).
* Heuristique et tables de transposition : `alphabeta_ia/alpha_beta.py`.
* Ouvertures : `alphabeta_ia/opening_book.py`.

## Problèmes connus

* Si Pygame ne s’installe pas : pip install "pygame>=2.5,<2.6".
* Si les clics ne tombent pas au bon endroit après modification visuelle, aligner tailles/offsets entre gui/draw.py et les calculs de position dans main.py.
