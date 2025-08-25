import numpy as np
import random
import time
import math
from collections import defaultdict

from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2, BOARD_ROWS, BOARD_COLS, QUADRANT_SIZE

def fast_check_win_numpy(board):
    if not isinstance(board, np.ndarray):
        board = np.array(board)
    
    p1_win = False
    p2_win = False
    
    for r in range(6):
        for c in range(2):
            row_sum = np.sum(board[r, c:c+5])
            if row_sum == 5: p1_win = True
            elif row_sum == -5: p2_win = True
    
    for c in range(6):
        for r in range(2):
            col_sum = np.sum(board[r:r+5, c])
            if col_sum == 5: p1_win = True
            elif col_sum == -5: p2_win = True
    
    for r in range(2):
        for c in range(2):
            window = board[r:r+5, c:c+5]
            
            main_diag = np.diagonal(window)
            diag_sum = np.sum(main_diag)
            if diag_sum == 5: p1_win = True
            elif diag_sum == -5: p2_win = True
            
            anti_diag = np.diagonal(np.fliplr(window))
            anti_diag_sum = np.sum(anti_diag)
            if anti_diag_sum == 5: p1_win = True
            elif anti_diag_sum == -5: p2_win = True
    
    if p1_win and p2_win: return 2
    elif p1_win: return 1
    elif p2_win: return -1
    else: return 0

def fast_rotate_quadrant_numpy(board, quad_idx, direction):
    result = board.copy()
    row_start = (quad_idx // 2) * 3
    col_start = (quad_idx % 2) * 3
    
    quadrant = result[row_start:row_start+3, col_start:col_start+3]
    rotated = np.rot90(quadrant, k=direction)
    result[row_start:row_start+3, col_start:col_start+3] = rotated
    
    return result

def get_empty_positions_numpy(board):
    empty_indices = np.where(board == 0)
    return list(zip(empty_indices[0], empty_indices[1]))

def evaluate_position_heuristic(board, player):
    score = 0
    opponent = -player
    
    for r in range(6):
        for c in range(2):
            line = board[r, c:c+5]
            score += evaluate_line(line, player)
    
    for c in range(6):
        for r in range(2):
            line = board[r:r+5, c]
            score += evaluate_line(line, player)
    
    for r in range(2):
        for c in range(2):
            window = board[r:r+5, c:c+5]
            main_diag = np.diagonal(window)
            anti_diag = np.diagonal(np.fliplr(window))
            score += evaluate_line(main_diag, player)
            score += evaluate_line(anti_diag, player)
    
    return score

def evaluate_line(line, player):
    player_count = np.sum(line == player)
    opponent_count = np.sum(line == -player)
    
    if opponent_count > 0:
        return 0
    
    if player_count == 4: return 50
    elif player_count == 3: return 10
    elif player_count == 2: return 3
    elif player_count == 1: return 1
    else: return 0

class OptimizedMCTSNode:
    __slots__ = ['board_hash', 'board_state', 'player', 'parent', 'move', 
                 'children', 'wins', 'visits', 'untried_moves', '_terminal_checked', '_is_terminal']
    
    def __init__(self, board_state, player, parent=None, move=None):
        self.board_state = board_state
        self.board_hash = hash(board_state.tobytes())
        self.player = player
        self.parent = parent
        self.move = move
        
        self.children = []
        self.wins = 0.0
        self.visits = 0
        
        self._terminal_checked = False
        self._is_terminal = None
        
        self.untried_moves = self._get_smart_prioritized_moves()
    
    def _get_smart_prioritized_moves(self):
        empty_positions = get_empty_positions_numpy(self.board_state)
        
        if not empty_positions:
            return []
        
        moves = []
        
        position_scores = {}
        for r, c in empty_positions:
            score = 0
            
            center_distance = abs(r - 2.5) + abs(c - 2.5)
            score += max(0, 8 - center_distance)
            
            if (r, c) in [(0, 0), (0, 5), (5, 0), (5, 5)]:
                score += 6
            
            if (r, c) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
                score += 8
            
            quad_centers = [(1, 1), (1, 4), (4, 1), (4, 4)]
            if (r, c) in quad_centers:
                score += 5
            
            temp_board = self.board_state.copy()
            temp_board[r, c] = self.player
            tactical_score = evaluate_position_heuristic(temp_board, self.player)
            score += tactical_score * 0.1
            
            position_scores[(r, c)] = score
        
        sorted_positions = sorted(empty_positions, 
                                key=lambda pos: position_scores[pos], 
                                reverse=True)
        
        game_phase = len(empty_positions)
        
        if game_phase > 30:
            rotation_options = [
                (0, 1), (1, 1), (2, 1), (3, 1),
                (0, -1), (1, -1), (2, -1), (3, -1)
            ]
        elif game_phase > 15:
            rotation_options = [
                (0, 1), (1, 1), (2, 1), (3, 1),
                (0, -1), (3, -1)
            ]
        else:
            rotation_options = [
                (0, 1), (1, 1), (2, 1), (3, 1)
            ]
        
        num_positions = min(len(sorted_positions), max(8, len(sorted_positions) // 3))
        top_positions = sorted_positions[:num_positions]
        
        for r, c in top_positions:
            for quad_idx, direction in rotation_options:
                moves.append((r, c, quad_idx, direction))
        
        return moves
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal_node(self):
        if not self._terminal_checked:
            win_result = fast_check_win_numpy(self.board_state)
            is_full = not np.any(self.board_state == 0)
            self._is_terminal = (win_result != 0) or is_full
            self._terminal_checked = True
        return self._is_terminal
    
    def select_child_enhanced(self, exploration_constant=1.41):
        if not self.children:
            return None
        
        best_child = None
        best_score = -float('inf')
        
        for child in self.children:
            if child.visits == 0:
                uct_score = 1000.0 + random.random()
            else:
                exploitation_term = child.wins / child.visits
                exploration_term = exploration_constant * math.sqrt(
                    math.log(max(1, self.visits)) / child.visits
                )
                
                bias = 0
                if child.move:
                    r, c, quad_idx, direction = child.move
                    center_distance = abs(r - 2.5) + abs(c - 2.5)
                    bias += max(0, (6 - center_distance) * 0.02)
                    
                    if direction == 1:
                        bias += 0.01
                
                robustness_bonus = min(0.2, child.visits / 25)
                
                uct_score = exploitation_term + exploration_term + bias + robustness_bonus
            
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        
        return best_child
    
    def expand(self):
        if not self.untried_moves:
            return None
        
        move = self.untried_moves.pop(0)
        r, c, quad_idx, direction = move
        
        new_board = self.board_state.copy()
        new_board[r, c] = self.player
        
        if fast_check_win_numpy(new_board) != 0:
            final_board = new_board
        else:
            final_board = fast_rotate_quadrant_numpy(new_board, quad_idx, direction)
        
        child_node = OptimizedMCTSNode(
            board_state=final_board,
            player=-self.player,
            parent=self,
            move=move
        )
        
        self.children.append(child_node)
        return child_node
    
    def backpropagate(self, result):
        node = self
        while node is not None:
            node.visits += 1
            
            if node.parent:
                if node.parent.player == result:
                    node.wins += 1.0
                elif result == 0 or result == 2:
                    node.wins += 0.5
            else:
                if result == node.player:
                    node.wins += 1.0
                elif result == 0 or result == 2:
                    node.wins += 0.5
            
            node = node.parent

class OptimizedMCTS:
    
    def __init__(self, time_limit=5.0, exploration_constant=1.41):
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        
        self.transposition_table = {}
        self.root = None
        
        self.last_iterations = 0
        self.last_think_time = 0.0
    
    def find_best_move(self, game_instance):
        start_time = time.time()
        
        current_board = game_instance.board.copy()
        current_player = game_instance.current_player
        
        board_hash = hash(current_board.tobytes())
        if board_hash in self.transposition_table:
            self.root = self.transposition_table[board_hash]
            self.root.player = current_player
        else:
            self.root = OptimizedMCTSNode(board_state=current_board, player=current_player)

        if not self.root.untried_moves and not self.root.children:
            return None

        iterations = 0
        max_iterations = 50000
        
        empty_count = np.sum(current_board == 0)
        if empty_count > 30:
            target_visits_per_move = 30
            early_stop_threshold = 100
        elif empty_count > 15:
            target_visits_per_move = 75
            early_stop_threshold = 200
        else:
            target_visits_per_move = 150
            early_stop_threshold = 500
        
        last_best_move = None
        stable_iterations = 0
        
        while (time.time() - start_time < self.time_limit and 
               iterations < max_iterations):
            
            node = self._select(self.root)
            
            if not node.is_terminal_node() and node.untried_moves:
                node = node.expand()
            
            if node:
                result = self._simulate_strategic(node)
                node.backpropagate(result)
            
            iterations += 1
            
            if iterations > early_stop_threshold and iterations % 25 == 0:
                current_best = self._get_current_best_move()
                if current_best == last_best_move:
                    stable_iterations += 1
                    if stable_iterations >= 5:
                        break
                else:
                    stable_iterations = 0
                last_best_move = current_best
        
        if len(self.transposition_table) > 1000:
            keys_to_remove = list(self.transposition_table.keys())[:500]
            for key in keys_to_remove:
                del self.transposition_table[key]
        
        self.transposition_table[board_hash] = self.root
        
        if not self.root.children:
            if self.root.untried_moves:
                return random.choice(self.root.untried_moves[:3])
            return None
        
        best_child = self._select_best_move_enhanced()
        
        self.last_iterations = iterations
        self.last_think_time = time.time() - start_time
        
        if best_child:
            win_rate = best_child.wins / max(1, best_child.visits)
            total_visits = sum(c.visits for c in self.root.children)
            visit_concentration = best_child.visits / max(1, total_visits)
            
            sorted_children = sorted(self.root.children, key=lambda c: c.visits, reverse=True)
            top_moves = []
            for i, child in enumerate(sorted_children[:3]):
                child_wr = child.wins / max(1, child.visits)
                top_moves.append(f"#{i+1}: {child.move} ({child.visits}v, {child_wr:.3f}wr)")
            
            print(f"[Optimized MCTS] Iterations: {iterations} | "
                  f"Best: {best_child.move} | "
                  f"Visits: {best_child.visits} | "
                  f"Win rate: {win_rate:.3f} | "
                  f"Concentration: {visit_concentration:.3f}")
            print(f"[Top moves] {' | '.join(top_moves)}")
        
        return best_child.move if best_child else None
    
    def _select(self, node):
        while not node.is_terminal_node():
            if not node.is_fully_expanded():
                return node
            
            if not node.children:
                return node
            
            node = node.select_child_enhanced(self.exploration_constant)
            if node is None:
                break
        
        return node if node else self.root
    
    def _simulate_strategic(self, node):
        board = node.board_state.copy()
        player = node.player
        moves_played = 0
        max_moves = 40
        
        while moves_played < max_moves:
            empty_positions = get_empty_positions_numpy(board)
            if not empty_positions:
                break
            
            moves_played += 1
            
            if moves_played <= 5:
                strategic_positions = []
                for r, c in empty_positions:
                    center_dist = abs(r - 2.5) + abs(c - 2.5)
                    if center_dist <= 3:
                        strategic_positions.append((r, c))
                
                if strategic_positions:
                    r, c = random.choice(strategic_positions)
                else:
                    r, c = random.choice(empty_positions)
            elif moves_played <= 15:
                if random.random() < 0.3:
                    best_pos = None
                    best_score = -float('inf')
                    
                    for r, c in empty_positions[:min(10, len(empty_positions))]:
                        temp_board = board.copy()
                        temp_board[r, c] = player
                        score = evaluate_position_heuristic(temp_board, player)
                        if score > best_score:
                            best_score = score
                            best_pos = (r, c)
                    
                    if best_pos:
                        r, c = best_pos
                    else:
                        r, c = random.choice(empty_positions)
                else:
                    r, c = random.choice(empty_positions)
            else:
                r, c = random.choice(empty_positions)
            
            board[r, c] = player
            
            result = fast_check_win_numpy(board)
            if result != 0:
                return result
            
            if moves_played <= 10:
                quad_scores = []
                for quad_idx in range(4):
                    for direction in [1, -1]:
                        test_board = fast_rotate_quadrant_numpy(board, quad_idx, direction)
                        score = evaluate_position_heuristic(test_board, player)
                        quad_scores.append((score, quad_idx, direction))
                
                if quad_scores and random.random() < 0.4:
                    quad_scores.sort(reverse=True)
                    _, quad_idx, direction = quad_scores[0]
                else:
                    quad_idx = random.randint(0, 3)
                    direction = random.choice([-1, 1])
            else:
                quad_idx = random.randint(0, 3)
                direction = random.choice([-1, 1])
            
            board = fast_rotate_quadrant_numpy(board, quad_idx, direction)
            
            result = fast_check_win_numpy(board)
            if result != 0:
                return result
            
            player = -player
        
        return 0
    
    def _get_current_best_move(self):
        if not self.root.children:
            return None
        return max(self.root.children, key=lambda c: c.visits).move
    
    def _select_best_move_enhanced(self):
        if not self.root.children:
            return None
        
        min_visits = max(3, sum(c.visits for c in self.root.children) * 0.02)
        viable_children = [c for c in self.root.children if c.visits >= min_visits]
        
        if not viable_children:
            viable_children = self.root.children
        
        best_child = None
        best_score = -1
        
        total_visits = sum(c.visits for c in viable_children)
        
        for child in viable_children:
            if child.visits == 0:
                continue
            
            win_rate = child.wins / child.visits
            visit_share = child.visits / max(1, total_visits)
            
            confidence = min(1.0, child.visits / 50)
            
            position_bonus = 0
            if child.move:
                r, c, _, _ = child.move
                center_dist = abs(r - 2.5) + abs(c - 2.5)
                position_bonus = max(0, (5 - center_dist) * 0.02)
            
            score = (win_rate * 0.6 +
                    visit_share * 0.25 +
                    confidence * 0.1 +
                    position_bonus * 0.05)
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child or max(self.root.children, key=lambda x: x.visits)