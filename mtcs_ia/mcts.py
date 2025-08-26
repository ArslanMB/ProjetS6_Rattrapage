# mcts_ia/mcts.py

import numpy as np
import random
import time
import math
from collections import defaultdict

from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2, BOARD_ROWS, BOARD_COLS, QUADRANT_SIZE

# --- MCTS Node Class ---
class MCTSNode:
    """
    A node in the Monte Carlo Search Tree.
    Each node represents a game state (a specific board configuration).
    """
    def __init__(self, board_state, player, parent=None, move=None):
        self.board_state = board_state
        self.player = player
        self.parent = parent
        self.move = move

        self.children = []
        self.wins = 0
        self.visits = 0
        
        self.untried_moves = self._get_legal_moves()

    def _get_legal_moves(self):
        """
        Generates all possible legal moves from the current board state.
        """
        legal_moves = []
        empty_cells = list(zip(*np.where(self.board_state == 0)))

        if not empty_cells:
            return []

        for r, c in empty_cells:
            for quad_idx in range(4):
                for direction in [-1, 1]:
                    legal_moves.append((r, c, quad_idx, direction))
        
        random.shuffle(legal_moves)
        return legal_moves

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal_node(self):
        return PentagoGame.check_win_on_board(self.board_state) != 0 or np.all(self.board_state != 0)

    def select_child(self, exploration_constant=1.41):
        """
        Selects the best child node using the UCT (Upper Confidence Bound for Trees) formula.
        """
        best_child = None
        best_score = -1

        for child in self.children:
            if child.visits == 0:
                return child

            exploit_term = child.wins / child.visits
            explore_term = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            uct_score = exploit_term + explore_term

            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        
        return best_child

    def expand(self):
        """
        Expands the tree by creating a new child node for one of the untried moves.
        """
        if not self.untried_moves:
            return None

        move = self.untried_moves.pop()
        r, c, quad_idx, direction = move

        board_after_placement = PentagoGame.get_board_after_placement(self.board_state, r, c, self.player)
        
        if PentagoGame.check_win_on_board(board_after_placement) != 0:
            new_board_state = board_after_placement
        else:
            new_board_state = PentagoGame.get_board_after_rotation(board_after_placement, quad_idx, direction)

        child_node = MCTSNode(
            board_state=new_board_state,
            player=-self.player,
            parent=self,
            move=move
        )
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        """
        Updates the win/visit counts from this node all the way up to the root.
        """
        node = self
        while node is not None:
            node.visits += 1
            if node.parent and node.parent.player == result:
                node.wins += 1
            elif result == 0 or result == 2: # Draw or double-win
                node.wins += 0.5
            node = node.parent

# --- MCTS Main Class ---
class MCTS_AI:
    def __init__(self, time_limit=None, iteration_limit=None, exploration_constant=1.41):
        if time_limit is None and iteration_limit is None:
            raise ValueError("Either time_limit or iteration_limit must be set.")
        
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.root = None

    def find_best_move(self, game_instance):
        start_time = time.time()
        
        current_board = np.copy(game_instance.board)
        current_player = game_instance.current_player
        self.root = MCTSNode(board_state=current_board, player=current_player)

        # Handle case where there are no moves left
        if not self.root.untried_moves:
            return None

        iterations = 0
        while True:
            if self.time_limit:
                if time.time() - start_time > self.time_limit:
                    break
            else:
                if iterations >= self.iteration_limit:
                    break
            
            node = self._select(self.root)

            if not node.is_terminal_node():
                node = node.expand()

            if node:
                result = self._simulate_fast(node)
                node.backpropagate(result)

            iterations += 1
        
        if not self.root.children:
            # This can happen if the time limit is too short to even expand the root node once.
            # In this case, we return a random move.
            print("[MCTS] Warning: Search time too short. Returning a random move.")
            return random.choice(self.root.untried_moves)

        best_child = max(self.root.children, key=lambda c: c.visits)
        
        print(f"[MCTS] Search complete. Iterations: {iterations}. Best move: {best_child.move} with {best_child.visits} visits and {best_child.wins:.1f} wins.")
        
        return best_child.move

    def _select(self, node):
        while not node.is_terminal_node():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.select_child(self.exploration_constant)
        return node

    # =================================================================
    # ================= FINAL, HIGH-SPEED SIMULATION ==================
    # =================================================================
    def _simulate_fast(self, node):
        """
        Plays a random game to completion from the node's state.
        Crucially, it only checks for a winner ONCE at the very end.
        """
        board = np.copy(node.board_state)
        player = node.player
        
        empty_cells = list(zip(*np.where(board == 0)))
        random.shuffle(empty_cells)
        
        # Play moves until the board is full
        while empty_cells:
            r, c = empty_cells.pop()
            board[r, c] = player
            
            # Perform a random rotation
            quad_idx = random.randint(0, 3)
            direction = random.choice([-1, 1])
            
            row_start = (quad_idx // 2) * QUADRANT_SIZE
            col_start = (quad_idx % 2) * QUADRANT_SIZE
            
            quadrant_slice = board[row_start:row_start+QUADRANT_SIZE, col_start:col_start+QUADRANT_SIZE]
            rotated_slice = np.rot90(quadrant_slice, k=direction)
            board[row_start:row_start+QUADRANT_SIZE, col_start:col_start+QUADRANT_SIZE] = rotated_slice
            
            player = -player
            
        # Now, check for the winner just once.
        return PentagoGame.check_win_on_board(board)
