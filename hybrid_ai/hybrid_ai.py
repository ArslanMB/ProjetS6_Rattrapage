"""
hybrid_ai.py

Hybrid Root-MCTS → Alpha-Beta for Pentago

Strategy:
 1) Run MCTS at the root for ~60% of the allocated time.
 2) Take the top-K moves ranked by visits from MCTS.
 3) For each candidate, apply the move and run a deeper Alpha-Beta search from that child position.
 4) Return the move with the best Alpha-Beta score.

"""

import time
import numpy as np
from mtcs_ia.optimized_mcts import OptimizedMCTS
import alphabeta_ia.alpha_beta as ab


class HybridAI:
    def __init__(self, total_time=3.0, top_k=5, ab_depth_or_A='A'):
        """
        total_time: total time budget in seconds for find_move
        top_k: number of top candidate moves from MCTS to check with Alpha-Beta
        ab_depth_or_A: depth parameter for alpha_beta.find_best_move_minimax
        """
        self.total_time = float(total_time)
        self.top_k = int(top_k)
        self.ab_depth_or_A = ab_depth_or_A

    def find_move(self, game_instance):
        """Return a move (r,c,quad,d) for the given PentagoGame instance."""
        t0 = time.perf_counter()
        player = game_instance.current_player

        # 1) Run MCTS at the root
        mcts_time = self.total_time * 0.6
        mcts = OptimizedMCTS(time_limit=mcts_time)
        best_from_mcts = mcts.find_best_move(game_instance)

        # 2) Get top-K moves by visits
        candidates = []
        if hasattr(mcts, 'root') and mcts.root and mcts.root.children:
            children_sorted = sorted(mcts.root.children, key=lambda c: c.visits, reverse=True)
            for c in children_sorted[:self.top_k]:
                if c.move is not None:
                    candidates.append(c.move)
        if not candidates and best_from_mcts:
            candidates = [best_from_mcts]
        if not candidates:
            legal = ab.get_legal_moves(game_instance.board)
            return legal[0] if legal else None

        # 3) For each candidate, run Alpha-Beta from that child position
        best_move = None
        best_score = -float("inf")
        per_candidate_time = (self.total_time - (time.perf_counter() - t0)) / max(1, len(candidates))

        for mv in candidates:
            # Apply the candidate move
            board2 = ab.apply_move_cached(game_instance.board, mv, player)

            # Create a minimal PentagoGame-like object for AB search
            class _TmpGame:
                def __init__(self, board, current_player):
                    self.board = np.copy(board)
                    self.current_player = -current_player

            tmp = _TmpGame(board2, player)

            try:
                # Timed Alpha-Beta search on the resulting position
                _, score = ab.timed_find_best_move_minimax(
                    tmp,
                    depth=self.ab_depth_or_A,
                    time_budget=per_candidate_time,
                    return_score=True
                )
            except Exception:
                # fallback to static eval if AB fails
                score = ab.evaluate(board2, player)

            if score > best_score:
                best_score = score
                best_move = mv

        return best_move


# Simple CLI demonstration
if __name__ == '__main__':
    try:
        from core.pentago_logic import PentagoGame
        print('HybridAI demo: create new game and ask for one move.')
        game = PentagoGame()
        ai = HybridAI(total_time=3.0, top_k=5)
        mv = ai.find_move(game)
        print('Selected move:', mv)
    except Exception as e:
        print('Demo failed (missing core package?). Error:', e)
