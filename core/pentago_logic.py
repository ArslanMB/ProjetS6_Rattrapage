# pentago_core.py

import numpy as np
from core.constants import BOARD_ROWS, BOARD_COLS, QUADRANT_SIZE, PLAYER_1, PLAYER_2, ROTATION_SPEED


class PentagoGame:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.current_player = PLAYER_1
        self.game_phase = "PLACEMENT"
        self.winner = 0
        self.game_state = 'START_MENU'
        self.animating_quadrant_idx = -1
        self.animating_direction = 0          # +1 = anti-horaire, -1 = horaire
        self.animation_angle = 0
        self.board_before_rotation = None

    def place_marble(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.game_phase = "ROTATION"
            return True
        return False

    def start_quadrant_rotation_animation(self, quadrant_idx, direction: int):
        """
        Lance l'animation de rotation d'un quadrant.

        direction: +1 = 90° anti-horaire (CCW), -1 = 90° horaire (CW)
        """
        self.game_phase = "ANIMATING_ROTATION"
        self.animating_quadrant_idx = quadrant_idx
        self.animating_direction = direction
        self.animation_angle = 0
        self.board_before_rotation = np.copy(self.board)

    def update_rotation_animation(self):
        """
        Avance l'animation. Retourne True si la rotation vient d'être finalisée.
        """
        if self.board_before_rotation is None:
            return False

        self.animation_angle += ROTATION_SPEED * self.animating_direction

        if abs(self.animation_angle) >= 90:
            # On fige l'angle à +/- 90° selon le sens
            self.animation_angle = 90 * self.animating_direction

            row_start = (self.animating_quadrant_idx // 2) * QUADRANT_SIZE
            col_start = (self.animating_quadrant_idx % 2) * QUADRANT_SIZE

            quadrant_slice = self.board_before_rotation[row_start:row_start+QUADRANT_SIZE,
                                                        col_start:col_start+QUADRANT_SIZE]
            # np.rot90(k=1) = 90° anti-horaire; k=-1 = 90° horaire
            rotated_slice = np.rot90(quadrant_slice, k=self.animating_direction)
            self.board[row_start:row_start+QUADRANT_SIZE,
                       col_start:col_start+QUADRANT_SIZE] = rotated_slice

            # Reset animation state
            self.animating_quadrant_idx = -1
            self.animating_direction = 0
            self.animation_angle = 0
            self.board_before_rotation = None

            # Check victoire / nul / fin
            win_code = self.check_win()

            if win_code == PLAYER_1 or win_code == PLAYER_2:
                self.winner = win_code
                self.game_state = 'GAME_OVER'
            elif win_code == 2:  # double-victoire = nul
                self.winner = 0
                self.game_state = 'GAME_OVER'
            elif np.all(self.board != 0):
                self.winner = 0
                self.game_state = 'GAME_OVER'
            else:
                self.current_player *= -1
                self.game_phase = "PLACEMENT"

            return True

        return False

    def check_win(self):
        return PentagoGame.check_win_on_board(self.board)

    @staticmethod
    def check_win_on_board(board_state: np.ndarray) -> int:
        """
        Retourne:
            PLAYER_1 (1) si P1 gagne,
            PLAYER_2 (-1) si P2 gagne,
            2 si double-victoire,
            0 sinon.
        """
        p1_win = False
        p2_win = False

        # Lignes
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS - 4):
                s = int(np.sum(board_state[r, c:c+5]))
                if s == 5:   p1_win = True
                if s == -5:  p2_win = True

        # Colonnes
        for c in range(BOARD_COLS):
            for r in range(BOARD_ROWS - 4):
                s = int(np.sum(board_state[r:r+5, c]))
                if s == 5:   p1_win = True
                if s == -5:  p2_win = True

        # Diagonales sur chaque sous-matrice 5x5
        for r in range(BOARD_ROWS - 4):
            for c in range(BOARD_COLS - 4):
                window = board_state[r:r+5, c:c+5]
                d = window.diagonal()
                ad = np.fliplr(window).diagonal()
                sd = int(np.sum(d))
                sad = int(np.sum(ad))
                if sd == 5 or sad == 5:
                    p1_win = True
                if sd == -5 or sad == -5:
                    p2_win = True

        if p1_win and p2_win:
            return 2
        if p1_win:
            return PLAYER_1
        if p2_win:
            return PLAYER_2
        return 0

    @staticmethod
    def get_board_after_placement(board_state: np.ndarray, row: int, col: int, player: int) -> np.ndarray:
        new_board = np.copy(board_state)
        new_board[row, col] = player
        return new_board

    @staticmethod
    def get_board_after_rotation(board_state: np.ndarray, quadrant_idx: int, direction: int) -> np.ndarray:
        """
        Applique une rotation de 90° à un quadrant et renvoie un nouveau board.

        direction: +1 = 90° anti-horaire (CCW), -1 = 90° horaire (CW)
        """
        new_board = np.copy(board_state)
        row_start = (quadrant_idx // 2) * QUADRANT_SIZE
        col_start = (quadrant_idx % 2) * QUADRANT_SIZE

        quadrant_slice = new_board[row_start:row_start+QUADRANT_SIZE,
                                   col_start:col_start+QUADRANT_SIZE]
        rotated_slice = np.rot90(quadrant_slice, k=direction)
        new_board[row_start:row_start+QUADRANT_SIZE,
                  col_start:col_start+QUADRANT_SIZE] = rotated_slice
        return new_board
