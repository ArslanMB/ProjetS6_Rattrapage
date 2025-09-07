"""

Stratégie :
1) Exécuter MCTS à la racine pendant environ 60 % du temps alloué (arbitraire)
2) Prendre les K meilleurs coups classés par nombre de visites dans MCTS
3) Pour chaque coup candidat, appliquer le coup et exécuter une recherche Alpha-Bêta plus profonde depuis cette position fille
4) Retourner le coup avec le meilleur score Alpha-Bêta
"""

import time
import numpy as np
from mtcs_ia.mcts_fast import MCTS_Fast
import alphabeta_ia.alpha_beta as ab


class HybridAI:
    def __init__(self, total_time=3.0, top_k=5, ab_depth_or_A='A'):
        """
        total_time : budget de temps total en secondes pour find_move
        top_k : nombre de meilleurs coups candidats issus de MCTS à vérifier avec Alpha-Bêta
        ab_depth_or_A : paramètre de profondeur pour alpha_beta.find_best_move_minimax
        """
        self.total_time = float(total_time)
        self.top_k = int(top_k)
        self.ab_depth_or_A = ab_depth_or_A

    def find_move(self, game_instance):
        """Renvoie un move (r,c,quad,d) pour l'instance PentagoGame donnée."""
        t0 = time.perf_counter()
        player = game_instance.current_player

        # 1) MCTS à la racine
        mcts_time = self.total_time * 0.6
        mcts = MCTS_Fast(time_limit=mcts_time)
        best_from_mcts = mcts.find_best_move(game_instance)

        # 2) Récupère les top-K moves les plus visités
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

        # 3) Pour chaque candidat on fait tourner AB à partir de ce child
        best_move = None
        best_score = -float("inf")
        per_candidate_time = (self.total_time - (time.perf_counter() - t0)) / max(1, len(candidates))

        for mv in candidates:
            # Appliquer le move candidat
            board2 = ab.apply_move_cached(game_instance.board, mv, player)

            # PentagoGame factice minimal pour AB
            class _TmpGame:
                def __init__(self, board, current_player):
                    self.board = np.copy(board)
                    self.current_player = -current_player # car après avoir jouer notre coup candidat, c'est à ladversaire de jouer

            tmp = _TmpGame(board2, player)

            try:
                # recherche AB timée, sur les positions obtenue par les candidats
                _, score = ab.timed_find_best_move_minimax(
                    tmp,
                    depth=self.ab_depth_or_A,
                    time_budget=per_candidate_time,
                    return_score=True
                )
            except Exception:
                # retour à une évaluation statique si AB ne fonctionne pas
                score = ab.evaluate(board2, player)

            if score > best_score:
                best_score = score
                best_move = mv

        return best_move


# Test CLI
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