import math
import numpy as np
from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2, BOARD_ROWS, BOARD_COLS

class GreedyAI:
    def __init__(self, player: int):
        """
        player = PLAYER_1 (1) ou PLAYER_2 (-1)
        """
        self.player = player
        self.opp = PLAYER_1 if player == PLAYER_2 else PLAYER_2

    def evaluate(self, board: np.ndarray) -> int:
        """
        Évalue un plateau numpy.
        """
        win_code = PentagoGame.check_win_on_board(board)
        if win_code == self.player:
            return 1_000_000
        elif win_code == self.opp:
            return -1_000_000
        elif win_code == 2:  # double victoire = nul
            return 0

        score_self = self._count_windows(board, self.player)
        score_opp = self._count_windows(board, self.opp)
        return score_self - score_opp

    def _count_windows(self, board: np.ndarray, player: int) -> int:
        n = BOARD_ROWS
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
                        cells.append(board[rr, cc])
                    if len(cells) == 5:
                        if all(v in (0, player) for v in cells):
                            cnt = sum(1 for v in cells if v == player)
                            if cnt > 0:
                                total += weights.get(cnt, 0)

        # bonus pour pièces au centre
        for (r,c) in centers:
            if board[r, c] == player:
                total += center_bonus
        return total

    def generate_all_moves(self, board: np.ndarray, player: int):
        """
        Génère tous les coups possibles :
        (row, col, quadrant_idx, direction)
        """
        moves = []
        empties = np.argwhere(board == 0)
        for (r, c) in empties:
            # poser la bille
            board_after_place = PentagoGame.get_board_after_placement(board, r, c, player)
            # puis tester toutes rotations possibles
            for q in range(4):
                for d in [+1, -1]:
                    new_board = PentagoGame.get_board_after_rotation(board_after_place, q, d)
                    moves.append((r, c, q, d, new_board))
        return moves

    def choose_move(self, game: PentagoGame):
        """
        Retourne (row, col, quadrant_idx, direction)
        """
        moves = self.generate_all_moves(game.board, self.player)
        if not moves:
            return None

        # 1. Victoire immédiate
        for (r, c, q, d, b2) in moves:
            if PentagoGame.check_win_on_board(b2) == self.player:
                return (r, c, q, d)

        # 2. Bloquer victoire adverse
        for (r, c, q, d, b2) in moves:
            if PentagoGame.check_win_on_board(b2) == self.opp:
                return (r, c, q, d)

        # 3. Évaluation gloutonne
        best_score = -math.inf
        best_move = None
        for (r, c, q, d, b2) in moves:
            score = self.evaluate(b2)
            if score > best_score:
                best_score = score
                best_move = (r, c, q, d)
        return best_move
