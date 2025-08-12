import numpy as np
import math
import random
import time

from pentago_logic import PentagoGame, BOARD_ROWS, BOARD_COLS, PLAYER_1, PLAYER_2

UCT_EXPLORATION_PARAM = math.sqrt(2)

def _get_player_for_state(board):
    p1_marbles = np.sum(board == PLAYER_1)
    p2_marbles = np.sum(board == PLAYER_2)
    if p1_marbles == p2_marbles:
        return PLAYER_1
    else:
        return PLAYER_2

def _get_all_possible_moves_for_state(board):
    moves = []
    empty_cells = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r, c] == 0:
                empty_cells.append((r, c))

    if not empty_cells:
        return []

    for r, c in empty_cells:
        for quadrant_idx in range(4):
            for direction in [1, -1]:
                moves.append((r, c, quadrant_idx, direction))
    return moves

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.player_to_move = _get_player_for_state(state)
        self.untried_moves = _get_all_possible_moves_for_state(state)
        random.shuffle(self.untried_moves)

    def select_child_uct(self):
        return max(self.children, key=lambda c:
                   (c.wins / c.visits) + UCT_EXPLORATION_PARAM * math.sqrt(math.log(self.visits) / c.visits))

    def expand(self):
        move = self.untried_moves.pop()
        r, c, quad_idx, direction = move
        
        board_after_placement = PentagoGame.get_board_after_placement(self.state, r, c, self.player_to_move)
        new_state = PentagoGame.get_board_after_rotation(board_after_placement, quad_idx, direction)
        
        child_node = MCTSNode(state=new_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_terminal_node(self):
        return PentagoGame.check_win_on_board(self.state) != 0 or np.all(self.state != 0)

def mcts_simulation_policy(board_state, player):
    current_state = np.copy(board_state)
    current_player = player
    
    while True:
        winner = PentagoGame.check_win_on_board(current_state)
        if winner != 0:
            return 1 if winner == PLAYER_2 else -1
        if np.all(current_state != 0):
            return 0
        
        possible_moves = _get_all_possible_moves_for_state(current_state)
        if not possible_moves: return 0
        
        move = random.choice(possible_moves)
        r, c, quad_idx, direction = move
        
        board_after_placement = PentagoGame.get_board_after_placement(current_state, r, c, current_player)
        current_state = PentagoGame.get_board_after_rotation(board_after_placement, quad_idx, direction)
        
        current_player *= -1

def find_best_move_mcts(game_instance, time_budget_seconds):
    start_time = time.time()
    
    root_node = MCTSNode(state=np.copy(game_instance.board), parent=None)
    
    while time.time() - start_time < time_budget_seconds:
        node = root_node
        
        while node.untried_moves == [] and node.children != []:
            node = node.select_child_uct()

        if not node.is_terminal_node():
            node = node.expand()

        result = mcts_simulation_policy(node.state, node.player_to_move)

        while node is not None:
            node.update(result)
            result *= -1
            node = node.parent
            
    most_visited_child = max(root_node.children, key=lambda c: c.visits)
    
    return most_visited_child.move