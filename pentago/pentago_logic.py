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
            self.game_phase = "ROTATION"                          
            return True
        return False

    def start_quadrant_rotation_animation(self, quadrant_idx, direction):
        # 1 = horaire , -1 = anti-horaire
        self.game_phase = "ANIMATING_ROTATION"
        self.animating_quadrant_idx = quadrant_idx
        self.animating_direction = direction
        self.animation_angle = 0
        self.board_before_rotation = np.copy(self.board)

    def update_rotation_animation(self):
        if self.board_before_rotation is None:
            return False
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

            # Chek de la victoir post rota
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
    def check_win_on_board(board_state):
        p1_win = False
        p2_win = False

        # Lignes
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS - 4):
                s = np.sum(board_state[r, c:c+5])
                if s == 5:   p1_win = True
                if s == -5:  p2_win = True

        # Colonnes
        for c in range(BOARD_COLS):
            for r in range(BOARD_ROWS - 4):
                s = np.sum(board_state[r:r+5, c])
                if s == 5:   p1_win = True
                if s == -5:  p2_win = True

        # Diagonales
        for r in range(BOARD_ROWS - 4):
            for c in range(BOARD_COLS - 4):
                d = board_state[r:r+5, c:c+5].diagonal()
                ad = np.fliplr(board_state[r:r+5, c:c+5]).diagonal()
                sd = np.sum(d)
                sad = np.sum(ad)
                if sd == 5 or sad == 5:     p1_win = True
                if sd == -5 or sad == -5:   p2_win = True

        if p1_win and p2_win:
            return 2  # double-victoire  
        if p1_win:
            return PLAYER_1
        if p2_win:
            return PLAYER_2
        return 0

    @staticmethod
    def get_board_after_placement(board_state, row, col, player):
        new_board = np.copy(board_state)
        new_board[row, col] = player
        return new_board

    @staticmethod
    def get_board_after_rotation(board_state, quadrant_idx, direction):

        #direction = 1 = 90° anti-horaire et -1 = 90° horaire 
 
        new_board = np.copy(board_state)
        row_start = (quadrant_idx // 2) * QUADRANT_SIZE
        col_start = (quadrant_idx % 2) * QUADRANT_SIZE
        
        quadrant_slice = new_board[row_start:row_start+3, col_start:col_start+3]
        rotated_slice = np.rot90(quadrant_slice, k=direction)
        new_board[row_start:row_start+3, col_start:col_start+3] = rotated_slice
        return new_board
