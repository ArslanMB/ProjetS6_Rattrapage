# mcts_fast_optimized.py - CORRECTIONS POUR PERFORMANCE ET TACTIQUE

import numpy as np
import random
import time
import math
from functools import lru_cache

from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2, BOARD_ROWS, BOARD_COLS, QUADRANT_SIZE

# Cache global pour les rotations de quadrants
_rotation_cache = {}

def cached_rotate_quadrant(board, quad_idx, direction):
    """Rotation avec cache pour éviter recalculs"""
    row_start = (quad_idx // 2) * 3
    col_start = (quad_idx % 2) * 3
    
    # Créer une clé pour le cache basée sur le quadrant
    quad_key = tuple(board[row_start:row_start+3, col_start:col_start+3].flatten())
    cache_key = (quad_key, direction)
    
    if cache_key in _rotation_cache:
        # Reconstruire le board avec le quadrant caché
        result = board.copy()
        cached_quad = _rotation_cache[cache_key]
        result[row_start:row_start+3, col_start:col_start+3] = cached_quad.reshape(3, 3)
        return result
    
    # Calculer et cacher
    result = board.copy()
    quad = result[row_start:row_start+3, col_start:col_start+3].copy()
    rotated = np.rot90(quad, k=direction)
    result[row_start:row_start+3, col_start:col_start+3] = rotated
    
    # Limiter la taille du cache
    if len(_rotation_cache) < 10000:
        _rotation_cache[cache_key] = rotated.flatten()
    
    return result

class MCTSNodeLite:
    """Nœud MCTS ultra-optimisé"""
    __slots__ = ['board', 'player', 'parent', 'move', 'children', 
                 'wins', 'visits', 'untried_moves', '_terminal', '_winner']
    
    def __init__(self, board_state, player, parent=None, move=None):
        self.board = board_state
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = None
        self._terminal = None  # Cache pour terminal check
        self._winner = None     # Cache pour winner
    
    def get_moves(self):
        """Génération optimisée avec moins d'allocations"""
        if self.untried_moves is None:
            # Utiliser une liste pré-allouée
            moves = []
            
            # Trouver cases vides plus efficacement
            empty_mask = (self.board == 0)
            empty_positions = np.column_stack(np.where(empty_mask))
            
            if len(empty_positions) == 0:
                self.untried_moves = []
                return []
            
            n_pieces = np.count_nonzero(self.board)
            
            # OPTIMISATION: Réduire drastiquement l'espace en début de partie
            if n_pieces < 4:
                # Très peu de rotations en début
                for pos in empty_positions:
                    r, c = pos[0], pos[1]
                    # Seulement 2 rotations par case
                    moves.append((r, c, 0, 1))
                    moves.append((r, c, 3, -1))
            elif n_pieces < 12:
                # Milieu de partie: 4 rotations par case
                for pos in empty_positions:
                    r, c = pos[0], pos[1]
                    moves.append((r, c, 0, 1))
                    moves.append((r, c, 1, 1))
                    moves.append((r, c, 2, -1))
                    moves.append((r, c, 3, -1))
            else:
                # Fin de partie: toutes les rotations
                for pos in empty_positions:
                    r, c = pos[0], pos[1]
                    for q in range(4):
                        moves.append((r, c, q, 1))
                        moves.append((r, c, q, -1))
            
            random.shuffle(moves)
            self.untried_moves = moves
        
        return self.untried_moves
    
    def is_terminal(self):
        """Check terminal avec cache"""
        if self._terminal is None:
            self._winner = PentagoGame.check_win_on_board(self.board)
            self._terminal = (self._winner != 0) or np.all(self.board != 0)
        return self._terminal
    
    def get_winner(self):
        """Récupère le gagnant (avec cache)"""
        if self._winner is None:
            self.is_terminal()  # Force le calcul
        return self._winner
    
    def expand(self):
        """Expansion avec rotation cachée"""
        moves = self.get_moves()
        if not moves:
            return None
        
        move = moves.pop()
        r, c, quad, direction = move
        
        # Application directe sans copie intermédiaire
        new_board = self.board.copy()
        new_board[r, c] = self.player
        new_board = cached_rotate_quadrant(new_board, quad, direction)
        
        child = MCTSNodeLite(new_board, -self.player, self, move)
        self.children.append(child)
        return child

class MCTS_Fast:
    def __init__(self, time_limit=None, iteration_limit=None, exploration_constant=1.2):
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.c = exploration_constant
        # Pré-calculs pour UCT
        self.sqrt_2 = math.sqrt(2)
        
    def find_best_move(self, game_instance):
        start = time.time()
        board = game_instance.board.copy()
        player = game_instance.current_player
        
        # AMÉLIORATION: Détection tactique complète (pas juste échantillon)
        urgent = self._full_tactical_check(board, player)
        if urgent:
            print(f"[MCTS-V2] Tactical move found: {urgent}")
            return urgent
        
        root = MCTSNodeLite(board, player)
        root.get_moves()  # Pré-génération
        
        iterations = 0
        
        # CORRECTION: Respect strict du time_limit
        if self.time_limit:
            deadline = start + self.time_limit * 0.98  # 98% pour marge de sécurité
            
            # Batch processing pour réduire les checks de temps
            batch_size = 50
            while time.time() < deadline:
                batch_end = min(batch_size, int((deadline - time.time()) * 200))  # Estimation
                if batch_end <= 0:
                    break
                    
                for _ in range(batch_end):
                    self._single_iteration_fast(root)
                    iterations += 1
                    
                # Adapter la taille du batch selon la vitesse
                if iterations > 100:
                    # Estimer le temps par itération
                    elapsed = time.time() - start
                    time_per_iter = elapsed / iterations
                    remaining_time = deadline - time.time()
                    batch_size = max(10, min(100, int(remaining_time / time_per_iter / 2)))
        else:
            for _ in range(self.iteration_limit or 1000):
                self._single_iteration_fast(root)
                iterations += 1
        
        # Sélection finale
        if not root.children:
            moves = root.get_moves()
            return moves[0] if moves else None
        
        # Meilleur = plus visité
        best = max(root.children, key=lambda c: c.visits)
        
        elapsed = time.time() - start
        speed = iterations / elapsed if elapsed > 0 else 0
        print(f"[MCTS-V2] {iterations} iterations in {elapsed:.2f}s = {speed:.0f} it/s")
        if best.visits > 0:
            print(f"          Best move: {best.move[:2]} -> Q{best.move[2]}, Visits: {best.visits}")
        
        return best.move
    
    def _single_iteration_fast(self, root):
        """Itération optimisée avec moins d'allocations"""
        # Selection (sans créer de liste path)
        node = root
        depth = 0
        path_nodes = []  # Réutiliser la même liste
        
        while not node.is_terminal() and depth < 50:  # Limite de profondeur
            path_nodes.append(node)
            
            moves = node.get_moves()
            if moves:  # Peut expand
                child = node.expand()
                if child:
                    path_nodes.append(child)
                    # Simulation inline rapide
                    result = self._ultra_fast_rollout(child.board, child.player)
                    # Backprop inline
                    self._fast_backprop(path_nodes, result)
                return
            
            if not node.children:
                break
            
            # Sélection UCT inline
            node = self._select_child_inline(node)
            if node is None:
                break
            depth += 1
        
        # Terminal ou leaf
        if path_nodes:
            path_nodes.append(node)
            result = node.get_winner() if node.is_terminal() else 0
            self._fast_backprop(path_nodes, result)
    
    def _select_child_inline(self, node):
        """Sélection UCT sans allocations"""
        if not node.children:
            return None
            
        best = None
        best_value = -999999
        
        # Pré-calcul
        if node.visits > 0:
            log_parent = math.log(node.visits)
        else:
            return node.children[0] if node.children else None
        
        for child in node.children:
            if child.visits == 0:
                return child
            
            # UCT direct
            if node.player == PLAYER_1:
                value = child.wins / child.visits + self.c * math.sqrt(log_parent / child.visits)
            else:
                value = (1.0 - child.wins / child.visits) + self.c * math.sqrt(log_parent / child.visits)
            
            if value > best_value:
                best_value = value
                best = child
        
        return best
    
    def _fast_backprop(self, path, result):
        """Backprop optimisée"""
        if result == PLAYER_1:
            for node in path:
                node.visits += 1
                node.wins += 1.0
        elif result == PLAYER_2:
            for node in path:
                node.visits += 1
                # wins reste à 0
        else:  # Nul
            for node in path:
                node.visits += 1
                node.wins += 0.5
    
    def _ultra_fast_rollout(self, board, player):
        """Simulation minimaliste pour vitesse max"""
        b = board  # Pas de copie!
        p = player
        
        for step in range(20):  # Limite basse
            # Check win inline
            w = PentagoGame.check_win_on_board(b)
            if w != 0:
                return w
            
            # Trouver case vide rapidement
            empty_mask = (b == 0)
            if not np.any(empty_mask):
                return PentagoGame.check_win_on_board(b)
            
            # Position aléatoire parmi les vides
            empty_positions = np.column_stack(np.where(empty_mask))
            if len(empty_positions) == 0:
                break
            
            # Copier seulement maintenant
            if step == 0:
                b = b.copy()
            
            # Placement
            pos = empty_positions[random.randint(0, len(empty_positions)-1)]
            b[pos[0], pos[1]] = p
            
            # Check win après placement
            w = PentagoGame.check_win_on_board(b)
            if w != 0:
                return w
            
            # Rotation simple
            q = random.randint(0, 3)
            b = cached_rotate_quadrant(b, q, 1 if random.random() > 0.5 else -1)
            
            p = -p
        
        return 0
    
    def _full_tactical_check(self, board, player):
        """
        CORRECTION: Vérification tactique COMPLÈTE pour win_detection
        Teste TOUTES les combinaisons pour garantir de trouver une victoire en 1
        """
        empty_positions = np.column_stack(np.where(board == 0))
        
        if len(empty_positions) == 0:
            return None
        
        # 1. VICTOIRE IMMÉDIATE - Tester TOUTES les possibilités
        for pos in empty_positions:
            r, c = pos[0], pos[1]
            test_board = board.copy()
            test_board[r, c] = player
            
            # Tester TOUTES les rotations
            for quad in range(4):
                for direction in [-1, 1]:
                    rotated = cached_rotate_quadrant(test_board, quad, direction)
                    if PentagoGame.check_win_on_board(rotated) == player:
                        return (r, c, quad, direction)
        
        # 2. BLOCAGE NÉCESSAIRE - Vérifier TOUTES les menaces
        opponent = -player
        blocking_moves = []
        
        for pos in empty_positions:
            r, c = pos[0], pos[1]
            test_board = board.copy()
            test_board[r, c] = opponent
            
            threat_found = False
            for quad in range(4):
                for direction in [-1, 1]:
                    rotated = cached_rotate_quadrant(test_board, quad, direction)
                    if PentagoGame.check_win_on_board(rotated) == opponent:
                        blocking_moves.append((r, c))
                        threat_found = True
                        break
                if threat_found:
                    break
        
        # Si menaces détectées, bloquer la première avec une rotation sûre
        if blocking_moves:
            r, c = blocking_moves[0]
            test_board = board.copy()
            test_board[r, c] = player
            
            # Trouver une rotation qui ne fait pas perdre
            best_move = None
            for quad in range(4):
                for direction in [-1, 1]:
                    rotated = cached_rotate_quadrant(test_board, quad, direction)
                    winner = PentagoGame.check_win_on_board(rotated)
                    
                    if winner == player:  # On gagne!
                        return (r, c, quad, direction)
                    elif winner == 0:  # Pas de victoire adverse
                        best_move = (r, c, quad, direction)
            
            return best_move if best_move else (r, c, 0, 1)
        
        return None