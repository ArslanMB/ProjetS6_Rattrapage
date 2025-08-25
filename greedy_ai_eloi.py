import math
from pentago_game import Pentago, Move

class GreedyAI:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.opp = "O" if symbol == "X" else "X"

    def evaluate(self, game: Pentago) -> int:
        winner = game.winner()
        if winner == self.symbol:
            return 1_000_000
        elif winner == self.opp:
            return -1_000_000
        elif winner == "Draw":
            return 0

        score_self = self._count_windows(game, self.symbol)
        score_opp = self._count_windows(game, self.opp)
        return score_self - score_opp

    def _count_windows(self, game: Pentago, player: str) -> int:
        B = game.board
        n = 6
        total = 0
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        weights = {1:1, 2:10, 3:100, 4:10_000}
        center_bonus = 50
        centers = {(2,2),(2,3),(3,2),(3,3)}

        for r in range(n):
            for c in range(n):
                for dr,dc in dirs:
                    cells = []
                    for k in range(5):
                        rr, cc = r+k*dr, c+k*dc
                        if not (0 <= rr < n and 0 <= cc < n):
                            break
                        cells.append(B[rr][cc])
                    if len(cells) == 5:
                        if all(v in (".", player) for v in cells):
                            cnt = cells.count(player)
                            if cnt > 0:
                                total += weights.get(cnt, 0)

        # bonus pour pièces au centre
        for (r,c) in centers:
            if B[r][c] == player:
                total += center_bonus
        return total

    def choose_move(self, game: Pentago) -> Move:
        moves = game.legal_moves()
        if not moves:
            return None

        # --- 1. Cherche victoire immédiate ---
        for move in moves:
            g2 = game.clone()
            g2.play(*move)
            if g2.winner() == self.symbol:
                return move

        # --- 2. Bloque victoire immédiate de l’adversaire ---
        for move in moves:
            g2 = game.clone()
            g2.play(*move)
            if g2.winner() == self.opp:
                return move

        # --- 3. Sinon évaluation gloutonne ---
        best_score = -math.inf
        best_move = None
        for move in moves:
            g2 = game.clone()
            g2.play(*move)
            score = self.evaluate(g2)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
