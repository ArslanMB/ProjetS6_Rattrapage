import numpy as np

BOARD_ROWS, BOARD_COLS = 6, 6
QUADRANT_SIZE = 3
PLAYER_1 = 1
PLAYER_2 = -1
ROTATION_SPEED = 5

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
        self.animating_direction = 0
        self.animation_angle = 0
        self.board_before_rotation = None

    def place_marble(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            if self.check_win():
                self.winner = self.current_player
                self.game_state = 'GAME_OVER'
            else:
                self.game_phase = "ROTATION"
            return True
        return False

    def start_quadrant_rotation_animation(self, quadrant_idx, direction):
        self.game_phase = "ANIMATING_ROTATION"
        self.animating_quadrant_idx = quadrant_idx
        self.animating_direction = direction
        self.animation_angle = 0
        self.board_before_rotation = np.copy(self.board)

    def update_rotation_animation(self):
        self.animation_angle += ROTATION_SPEED * self.animating_direction
        
        if abs(self.animation_angle) >= 90:
            self.animation_angle = 90 * self.animating_direction

            row_start = (self.animating_quadrant_idx // 2) * QUADRANT_SIZE
            col_start = (self.animating_quadrant_idx % 2) * QUADRANT_SIZE
            
            quadrant_slice = self.board_before_rotation[row_start:row_start+3, col_start:col_start+3]
            rotated_slice = np.rot90(quadrant_slice, k=self.animating_direction)
            self.board[row_start:row_start+3, col_start:col_start+3] = rotated_slice

            self.animating_quadrant_idx = -1
            self.animating_direction = 0
            self.animation_angle = 0
            self.board_before_rotation = None

            win_after_rotation = self.check_win()
            
            if win_after_rotation:
                self.winner = self.current_player
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
    def check_win_on_board(board_state):
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS - 4):
                segment = board_state[r, c:c+5]
                if np.sum(segment) == 5: return PLAYER_1
                if np.sum(segment) == -5: return PLAYER_2

        for c in range(BOARD_COLS):
            for r in range(BOARD_ROWS - 4):
                segment = board_state[r:r+5, c]
                if np.sum(segment) == 5: return PLAYER_1
                if np.sum(segment) == -5: return PLAYER_2

        for r in range(BOARD_ROWS - 4):
            for c in range(BOARD_COLS - 4):
                diag = board_state[r:r+5, c:c+5].diagonal()
                if np.sum(diag) == 5: return PLAYER_1
                if np.sum(diag) == -5: return PLAYER_2
                anti_diag = np.fliplr(board_state[r:r+5, c:c+5]).diagonal()
                if np.sum(anti_diag) == 5: return PLAYER_1
                if np.sum(anti_diag) == -5: return PLAYER_2
        return 0

    @staticmethod
    def get_board_after_placement(board_state, row, col, player):
        new_board = np.copy(board_state)
        new_board[row, col] = player
        return new_board

    @staticmethod
    def get_board_after_rotation(board_state, quadrant_idx, direction):
        new_board = np.copy(board_state)
        row_start = (quadrant_idx // 2) * QUADRANT_SIZE
        col_start = (quadrant_idx % 2) * QUADRANT_SIZE
        
        quadrant_slice = new_board[row_start:row_start+3, col_start:col_start+3]
        rotated_slice = np.rot90(quadrant_slice, k=direction)
        new_board[row_start:row_start+3, col_start:col_start+3] = rotated_slice
        return new_board