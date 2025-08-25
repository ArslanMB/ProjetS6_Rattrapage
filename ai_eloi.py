# ia_pentago.py
from __future__ import annotations
import time
import math
import random
from typing import Optional, Tuple, List, Dict
from pentago_game import Pentago, Move, Coord, Rotation

# ---------- Heuristique ----------
# On évalue toutes les "fenêtres" de 5 cases (lignes/colonnes/diagonales).
# Si une fenêtre contient uniquement des pions d'un seul joueur + des vides,
# on score selon le nombre de pions dans la fenêtre (1..4). Les fenêtres
# bloquées (contiennent X et O) valent 0.
# On ajoute un léger bonus de centralité.

WEIGHTS = {
    1: 1,
    2: 6,
    3: 30,
    4: 200,   # très fort : menaces directes
    # 5 est géré comme terminal dans minimax
}
CENTER_CELLS = {(2, 2), (2, 3), (3, 2), (3, 3)}
CENTER_BONUS = 3  # par pion au centre

def static_eval(game: Pentago, me: str) -> int:
    """Évalue la position du point de vue de 'me'."""
    winner = game.winner()
    if winner == me:
        return 1_000_000
    if winner is not None:
        # défaite ou draw
        return -1_000_000 if winner != "Draw" else 0

    score_me = 0
    score_opp = 0
    opp = "O" if me == "X" else "X"

    B = game.board
    n = 6

    def score_window(cells: List[str], player: str) -> int:
        if any(ch != "." and ch != player for ch in cells):
            return 0
        k = sum(1 for ch in cells if ch == player)
        if k == 0:
            return 0
        return WEIGHTS.get(k, 0)

    # horizontales
    for r in range(n):
        for c in range(n - 4):
            w = [B[r][c + i] for i in range(5)]
            score_me += score_window(w, me)
            score_opp += score_window(w, opp)
    # verticales
    for c in range(n):
        for r in range(n - 4):
            w = [B[r + i][c] for i in range(5)]
            score_me += score_window(w, me)
            score_opp += score_window(w, opp)
    # diag ↘
    for r in range(n - 4):
        for c in range(n - 4):
            w = [B[r + i][c + i] for i in range(5)]
            score_me += score_window(w, me)
            score_opp += score_window(w, opp)
    # diag ↗
    for r in range(4, n):
        for c in range(n - 4):
            w = [B[r - i][c + i] for i in range(5)]
            score_me += score_window(w, me)
            score_opp += score_window(w, opp)

    # bonus de centralité
    for (r, c) in CENTER_CELLS:
        if B[r][c] == me:
            score_me += CENTER_BONUS
        elif B[r][c] == opp:
            score_opp += CENTER_BONUS

    return score_me - score_opp


# ---------- IA ----------
class StrongPentagoAI:
    """
    IA basée sur minimax + alpha-beta, iterative deepening et table de transposition.
    Paramètres clés :
      - time_limit_s : budget temps par coup (par ex. 0.8 s)
      - max_depth     : profondeur plafond (sécurité)
    """
    def __init__(self, symbol: str = "O", time_limit_s: float = 0.8, max_depth: int = 5):
        self.symbol = symbol
        self.opponent = "X" if symbol == "O" else "O"
        self.time_limit_s = time_limit_s
        self.max_depth = max_depth
        self._tt: Dict[Tuple, Tuple[int, int, Optional[Move]]] = {}  # zobrist simplifié: key -> (depth, score, best_move)
        self._deadline: float = math.inf

    # ---------- API principale ----------
    def choose_move(self, game: Pentago) -> Optional[Move]:
        # Si ce n'est pas à nous de jouer, on renvoie None
        if game.current != self.symbol:
            return None

        moves = game.legal_moves()
        if not moves:
            return None

        # 1) coups gagnants immédiats
        for mv in moves:
            g2 = game.clone()
            g2.play(*mv)
            if g2.winner() == self.symbol:
                return mv

        # 2) si l'adversaire a un gain immédiat, essayer de le bloquer
        for mv in moves:
            g2 = game.clone()
            g2.play(*mv)
            # si après notre mv, l'adversaire a gagné (impossible ici car c'est nous qui venons de jouer),
            # on le verrait déjà dans winner(). Par contre on peut simuler son meilleur coup :
            opp_moves = g2.legal_moves()
            for omv in opp_moves:
                g3 = g2.clone()
                g3.play(*omv)
                if g3.winner() == self.opponent:
                    # ce mv ne bloque pas -> on continue
                    break
            else:
                # aucune réplique gagnante immédiate trouvée pour l'adversaire : bon candidat
                pass

        # 3) itérative deepening sous contrainte de temps
        self._deadline = time.time() + self.time_limit_s
        best_move: Optional[Move] = None
        best_score = -math.inf

        # léger ordre de coups initial : centralité de la pose + évaluation rapide
        ordered = self._order_moves(game, moves)

        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self._alphabeta_root(game, ordered, depth)
                if move is not None:
                    best_move = move
                    best_score = score
            except TimeoutError:
                break  # on garde le meilleur trouvé jusqu'ici

        # fallback
        if best_move is None:
            best_move = random.choice(moves)

        # print(f"[AI {self.symbol}] depth<= {depth} score={best_score}")  # debug
        return best_move

    # ---------- Minimax / Alpha-Beta ----------
    def _alphabeta_root(self, game: Pentago, moves: List[Move], depth: int) -> Tuple[int, Optional[Move]]:
        alpha = -math.inf
        beta = math.inf
        best_move = None
        best_score = -math.inf

        for mv in moves:
            self._check_time()
            g2 = game.clone()
            g2.play(*mv)  # joue self.symbol puis g2.current devient opponent (sauf terminal)
            score = self._alphabeta(g2, depth - 1, alpha, beta, maximizing=(g2.current == self.symbol))
            if score > best_score:
                best_score = score
                best_move = mv
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

        return best_score, best_move

    def _alphabeta(self, game: Pentago, depth: int, alpha: float, beta: float, maximizing: bool) -> int:
        self._check_time()

        key = self._hash(game)
        if key in self._tt:
            tt_depth, tt_score, _ = self._tt[key]
            if tt_depth >= depth:
                return tt_score

        # terminal ou profondeur nulle
        win = game.winner()
        if win is not None:
            score = self._terminal_score(win)
            self._tt[key] = (depth, score, None)
            return score

        if depth == 0:
            score = static_eval(game, self.symbol)
            self._tt[key] = (depth, score, None)
            return score

        moves = game.legal_moves()
        if not moves:
            score = static_eval(game, self.symbol)
            self._tt[key] = (depth, score, None)
            return score

        # ordre des coups : jouer d'abord ceux qui "ont l'air bons" (évalue après coup)
        moves = self._order_moves(game, moves)

        if maximizing:
            value = -math.inf
            for mv in moves:
                g2 = game.clone()
                g2.play(*mv)
                value = max(value, self._alphabeta(g2, depth - 1, alpha, beta, maximizing=(g2.current == self.symbol)))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for mv in moves:
                g2 = game.clone()
                g2.play(*mv)
                value = min(value, self._alphabeta(g2, depth - 1, alpha, beta, maximizing=(g2.current == self.symbol)))
                beta = min(beta, value)
                if alpha >= beta:
                    break

        self._tt[key] = (depth, value, None)
        return value

    # ---------- Utilitaires ----------
    def _terminal_score(self, win: str) -> int:
        if win == "Draw":
            return 0
        return 1_000_000 if win == self.symbol else -1_000_000

    def _order_moves(self, game: Pentago, moves: List[Move]) -> List[Move]:
        # Priorité : coups gagnants, centralité de la pose, éval rapide après coup
        scored: List[Tuple[int, Move]] = []
        for mv in moves:
            g2 = game.clone()
            g2.play(*mv)
            w = g2.winner()
            if w == self.symbol:
                return [mv] + [m for m in moves if m != mv]  # coup gagnant en tête
            # score de centralité (plus la pose est centrale, mieux c'est)
            (r, c), _ = mv
            centrality = -((r - 2.5) ** 2 + (c - 2.5) ** 2)  # plus proche du centre -> plus grand
            quick = static_eval(g2, self.symbol)
            scored.append((int(1000 * centrality) + quick // 100, mv))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [mv for _, mv in scored]

    def _hash(self, game: Pentago) -> Tuple:
        # clé simple pour transpo : (tuple des lignes, current)
        return (tuple("".join(row) for row in game.board), game.current)

    def _check_time(self):
        if time.time() > self._deadline:
            raise TimeoutError()

