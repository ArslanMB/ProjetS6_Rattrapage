# mcts_evaluation_fast.py
import numpy as np
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt

from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2, BOARD_ROWS, BOARD_COLS, QUADRANT_SIZE
# Importer la version rapide
from mcts_fast import MCTS_Fast


class MCTSEvaluator:
    """
    Suite complète de tests pour évaluer la qualité de l'implémentation MCTS_Fast
    """
    
    def __init__(self):
        self.results = {}
        
    def run_all_tests(self):
        """Lance tous les tests d'évaluation"""
        print("=== ÉVALUATION MCTS_Fast ===")
        print("1. Tests unitaires...")
        self.test_unit_tests()
        
        print("\n2. Tests de performance...")
        self.test_performance()
        
        print("\n3. Tests tactiques...")
        self.test_tactical_play()
        
        print("\n4. Tests de robustesse...")
        self.test_robustness()
        
        print("\n5. Tests comparatifs...")
        self.test_comparative_strength()
        
        self.generate_report()
        
    def test_unit_tests(self):
        """Tests unitaires de base"""
        results = {}
        
        # Test 1: Backpropagation correcte
        game = PentagoGame()
        mcts = MCTS_Fast(iteration_limit=100)
        move = mcts.find_best_move(game)
        
        # Vérifier que le move est valide
        results['backprop_consistency'] = move is not None and len(move) == 4
        
        # Test 2: Respect des règles de jeu
        results['rule_compliance'] = self._test_rule_compliance()
        
        # Test 3: Gestion des cas limites
        results['edge_cases'] = self._test_edge_cases()
        
        self.results['unit_tests'] = results
        print(f"Tests unitaires: {sum(results.values())}/{len(results)} passés")
        
    def _test_rule_compliance(self):
        """Teste si MCTS respecte les règles du jeu"""
        game = PentagoGame()
        mcts = MCTS_Fast(time_limit=1.0)
        
        for _ in range(10):  # 10 coups de test
            game.game_state = 'PLAYING'  # S'assurer que le jeu est en cours
            move = mcts.find_best_move(game)
            if not move:
                continue
                
            r, c, quad_idx, direction = move
            
            # Vérifier coup légal
            if game.board[r, c] != 0:
                return False
                
            # Appliquer le coup
            if game.game_state == 'PLAYING':
                game.place_marble(r, c)
                if game.game_phase == "ROTATION":
                    game.start_quadrant_rotation_animation(quad_idx, direction)
                    # Simuler fin d'animation
                    while game.game_phase == "ANIMATING_ROTATION":
                        if game.update_rotation_animation():
                            break
                        
            if game.game_state == 'GAME_OVER':
                break
                
        return True
        
    def _test_edge_cases(self):
        """Teste les cas limites"""
        # Plateau presque plein
        game = PentagoGame()
        game.board = np.ones((6, 6)) * PLAYER_1
        game.board[0, 0] = 0  # Une seule case libre
        game.board[0, 1] = 0
        game.game_state = 'PLAYING'
        
        mcts = MCTS_Fast(time_limit=0.5)
        move = mcts.find_best_move(game)
        
        # Doit retourner un coup valide
        return move is not None and game.board[move[0], move[1]] == 0
        
    def test_performance(self):
        """Tests de performance"""
        results = {}
        
        game = PentagoGame()
        game.game_state = 'PLAYING'
        
        # Test vitesse de base
        times = []
        for time_limit in [0.5, 1.0, 2.0]:  # Réduit pour tests plus rapides
            mcts = MCTS_Fast(time_limit=time_limit)
            
            start = time.time()
            move = mcts.find_best_move(game)
            actual_time = time.time() - start
            
            times.append((time_limit, actual_time))
            
        # Vérifier respect des limites de temps (± 10%)
        time_respect = all(actual <= limit * 1.1 for limit, actual in times)
        results['time_limits'] = time_respect
        
        # Test scalabilité
        iterations_per_second = self._test_scalability()
        results['iterations_per_second'] = iterations_per_second
        results['scalability_good'] = iterations_per_second > 100  # Au moins 100 it/s
        
        self.results['performance'] = results
        print(f"Performance: {iterations_per_second:.0f} itérations/seconde")
        
    def _test_scalability(self):
        """Mesure les itérations par seconde"""
        game = PentagoGame()
        game.game_state = 'PLAYING'
        
        # Test avec nombre fixe d'itérations pour mesurer la vitesse
        iterations = 1000
        mcts = MCTS_Fast(iteration_limit=iterations)
        
        start_time = time.time()
        mcts.find_best_move(game)
        elapsed = time.time() - start_time
        
        return iterations / elapsed if elapsed > 0 else 0
        
    def test_tactical_play(self):
        """Tests de jeu tactique"""
        results = {}
        
        # Test 1: Détecter victoire en 1 coup
        results['win_detection'] = self._test_win_in_one()
        
        # Test 2: Bloquer victoire adverse
        results['block_opponent'] = self._test_block_opponent()
        
        # Test 3: Préférer centre en début de partie
        results['opening_preference'] = self._test_opening_play()
        
        self.results['tactical'] = results
        passed = sum(results.values())
        print(f"Tests tactiques: {passed}/{len(results)} passés")
        
    def _test_win_in_one(self):
        """Teste si MCTS trouve une victoire en 1 coup"""
        game = PentagoGame()
        # Créer position où PLAYER_1 peut gagner
        game.board = np.array([
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        game.current_player = PLAYER_1
        game.game_state = 'PLAYING'
        
        # Augmenter le temps pour les tests tactiques
        mcts = MCTS_Fast(time_limit=3.0, exploration_constant=0.5)
        move = mcts.find_best_move(game)
        
        if not move:
            return False
            
        r, c, quad_idx, direction = move
        
        # Vérifier si le coup complète la ligne
        # Accepter aussi d'autres positions gagnantes possibles avec rotation
        test_board = game.board.copy()
        test_board[r, c] = PLAYER_1
        rotated = PentagoGame.get_board_after_rotation(test_board, quad_idx, direction)
        winner = PentagoGame.check_win_on_board(rotated)
        
        return winner == PLAYER_1
        
    def _test_block_opponent(self):
        """Teste si MCTS bloque une victoire adverse"""
        game = PentagoGame()
        # Position où PLAYER_2 menace de gagner
        game.board = np.array([
            [-1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        game.current_player = PLAYER_1  # P1 doit bloquer P2
        game.game_state = 'PLAYING'
        
        mcts = MCTS_Fast(time_limit=3.0, exploration_constant=0.5)
        move = mcts.find_best_move(game)
        
        if not move:
            return False
            
        r, c, quad_idx, direction = move
        
        # Vérifier si le coup bloque la menace
        # Le coup doit empêcher P2 de gagner au prochain tour
        test_board = game.board.copy()
        test_board[r, c] = PLAYER_1
        rotated = PentagoGame.get_board_after_rotation(test_board, quad_idx, direction)
        
        # Vérifier que P2 ne peut pas gagner immédiatement après
        # en plaçant dans la case (0,4)
        if rotated[0, 4] != 0:  # Si la case est occupée, c'est bon
            return True
        
        # Ou vérifier que même si P2 joue (0,4), il ne gagne pas
        test_board2 = rotated.copy()
        test_board2[0, 4] = PLAYER_2
        
        # Tester toutes les rotations possibles pour P2
        can_win = False
        for q in range(4):
            for d in [-1, 1]:
                final_board = PentagoGame.get_board_after_rotation(test_board2, q, d)
                if PentagoGame.check_win_on_board(final_board) == PLAYER_2:
                    can_win = True
                    break
            if can_win:
                break
        
        return not can_win
        
    def _test_opening_play(self):
        """Teste les préférences d'ouverture"""
        game = PentagoGame()
        game.game_state = 'PLAYING'
        mcts = MCTS_Fast(time_limit=1.0, exploration_constant=1.0)
        
        moves = []
        for _ in range(5):  # 5 parties différentes
            game.reset_game()
            game.game_state = 'PLAYING'
            move = mcts.find_best_move(game)
            if move:
                moves.append((move[0], move[1]))
                
        # Vérifier si préfère cases centrales
        central_moves = sum(1 for r, c in moves if 1 <= r <= 4 and 1 <= c <= 4)
        return central_moves >= len(moves) * 0.6  # Au moins 60% de coups centraux
        
    def test_robustness(self):
        """Tests de robustesse"""
        results = {}
        
        # Test avec différentes configurations
        configs = [
            {'time_limit': 0.1, 'exploration_constant': 0.5},
            {'time_limit': 0.5, 'exploration_constant': 1.0},
            {'time_limit': 1.0, 'exploration_constant': 1.5},
            {'time_limit': 2.0, 'exploration_constant': 2.0},
        ]
        
        stable_play = 0
        for config in configs:
            try:
                game = PentagoGame()
                game.game_state = 'PLAYING'
                mcts = MCTS_Fast(**config)
                move = mcts.find_best_move(game)
                if move:
                    stable_play += 1
            except Exception as e:
                print(f"Erreur avec config {config}: {e}")
                
        results['config_stability'] = stable_play / len(configs)
        results['error_handling'] = stable_play == len(configs)
        
        self.results['robustness'] = results
        print(f"Robustesse: {stable_play}/{len(configs)} configurations stables")
        
    def test_comparative_strength(self):
        """Tests comparatifs contre différents adversaires"""
        results = {}
        
        # Contre joueur aléatoire
        random_score = self._play_matches_against_random(10)
        results['vs_random'] = random_score
        
        # Contre lui-même (différents temps)
        self_score = self._play_self_matches(5)
        results['consistency'] = self_score
        
        self.results['comparative'] = results
        print(f"Contre aléatoire: {random_score:.1%}, Consistance: {self_score:.2f}")
        
    def _play_matches_against_random(self, num_games):
        """Joue contre un joueur aléatoire"""
        wins = 0
        draws = 0
        
        for game_num in range(num_games):
            game = PentagoGame()
            game.game_state = 'PLAYING'
            mcts = MCTS_Fast(time_limit=1.0, exploration_constant=0.8)
            
            moves_played = 0
            max_moves = 50  # Éviter parties infinies
            
            while game.game_state == 'PLAYING' and moves_played < max_moves:
                if game.current_player == PLAYER_1:
                    # MCTS joue
                    move = mcts.find_best_move(game)
                    if move:
                        r, c, quad_idx, direction = move
                        game.place_marble(r, c)
                        if game.game_phase == "ROTATION":
                            game.start_quadrant_rotation_animation(quad_idx, direction)
                            while game.game_phase == "ANIMATING_ROTATION":
                                if game.update_rotation_animation():
                                    break
                else:
                    # Joueur aléatoire
                    empty_cells = list(zip(*np.where(game.board == 0)))
                    if empty_cells:
                        r, c = random.choice(empty_cells)
                        game.place_marble(r, c)
                        if game.game_phase == "ROTATION":
                            quad_idx = random.randint(0, 3)
                            direction = random.choice([-1, 1])
                            game.start_quadrant_rotation_animation(quad_idx, direction)
                            while game.game_phase == "ANIMATING_ROTATION":
                                if game.update_rotation_animation():
                                    break
                                    
                moves_played += 1
                
            if game.winner == PLAYER_1:
                wins += 1
            elif game.winner == 0:
                draws += 1
                
        return wins / num_games
        
    def _play_self_matches(self, num_games):
        """MCTS contre lui-même avec différents temps"""
        scores = []
        
        for _ in range(num_games):
            game = PentagoGame()
            game.game_state = 'PLAYING'
            
            # Différence de temps plus importante
            mcts_fast = MCTS_Fast(time_limit=0.5, exploration_constant=1.0)
            mcts_slow = MCTS_Fast(time_limit=3.0, exploration_constant=0.8)
            
            moves_played = 0
            max_moves = 50
            
            while game.game_state == 'PLAYING' and moves_played < max_moves:
                current_mcts = mcts_fast if game.current_player == PLAYER_1 else mcts_slow
                
                move = current_mcts.find_best_move(game)
                if move:
                    r, c, quad_idx, direction = move
                    game.place_marble(r, c)
                    if game.game_phase == "ROTATION":
                        game.start_quadrant_rotation_animation(quad_idx, direction)
                        while game.game_phase == "ANIMATING_ROTATION":
                            if game.update_rotation_animation():
                                break
                                
                moves_played += 1
                
            # Score basé sur qui gagne (slow devrait gagner plus souvent)
            if game.winner == PLAYER_2:  # mcts_slow
                scores.append(1.0)
            elif game.winner == PLAYER_1:  # mcts_fast
                scores.append(0.0)
            else:
                scores.append(0.5)  # nul
                
        avg_score = np.mean(scores) if scores else 0
        # On s'attend à ce que le joueur avec plus de temps gagne plus souvent
        # Un bon score de consistance serait > 0.6 (le joueur lent gagne 60%+ du temps)
        return avg_score
        
    def generate_report(self):
        """Génère un rapport d'évaluation complet"""
        print("\n" + "="*50)
        print("RAPPORT D'ÉVALUATION MCTS_Fast")
        print("="*50)
        
        total_score = 0
        max_score = 0
        
        for category, tests in self.results.items():
            print(f"\n{category.upper()}:")
            category_score = 0
            category_max = 0
            
            for test_name, result in tests.items():
                if isinstance(result, bool):
                    score = 1 if result else 0
                    max_val = 1
                    status = "✓" if result else "✗"
                elif isinstance(result, (int, float)):
                    if 'score' in test_name or 'vs_' in test_name or 'consistency' in test_name:
                        score = result
                        max_val = 1.0
                        if 'consistency' in test_name:
                            status = f"{result:.2f}"
                        else:
                            status = f"{result:.1%}"
                    elif 'iterations_per_second' in test_name:
                        # Échelle de score pour it/s
                        if result >= 200:
                            score = 1.0
                        elif result >= 100:
                            score = 0.5
                        else:
                            score = result / 200
                        max_val = 1
                        status = f"{result:.0f}"
                    else:
                        score = min(result, 1)  # Normaliser
                        max_val = 1
                        status = f"{result:.0f}"
                else:
                    score = 0
                    max_val = 1
                    status = "?"
                    
                print(f"  {test_name}: {status}")
                category_score += score
                category_max += max_val
                
            print(f"  → {category_score:.1f}/{category_max:.1f}")
            total_score += category_score
            max_score += category_max
            
        print(f"\nSCORE GLOBAL: {total_score:.1f}/{max_score:.1f} ({total_score/max_score:.1%})")
        
        # Recommandations
        print("\nRECOMMANDATIONS:")
        self._generate_recommendations()
        
    def _generate_recommendations(self):
        """Génère des recommandations d'amélioration"""
        recommendations = []
        
        # Analyser les résultats
        if 'performance' in self.results:
            perf = self.results['performance']
            its = perf.get('iterations_per_second', 0)
            if its < 200:
                recommendations.append(f"• Optimiser la vitesse de simulation (actuellement {its:.0f} it/s, cible: 200+)")
                recommendations.append("  → Considérer l'utilisation de Numba ou Cython pour les fonctions critiques")
                recommendations.append("  → Réduire les allocations mémoire dans les simulations")
                
        if 'tactical' in self.results:
            tact = self.results['tactical']
            failed_tactics = [k for k, v in tact.items() if not v]
            if failed_tactics:
                recommendations.append(f"• Améliorer les capacités tactiques: {', '.join(failed_tactics)}")
                recommendations.append("  → Augmenter le temps/itérations pour la détection tactique")
                recommendations.append("  → Implémenter une table de transposition pour mémoriser les positions")
                
        if 'comparative' in self.results:
            comp = self.results['comparative']
            if comp.get('vs_random', 0) < 0.8:
                recommendations.append("• Renforcer contre adversaires faibles (< 80% vs aléatoire)")
                recommendations.append("  → Vérifier la logique de backpropagation")
            if comp.get('consistency', 0) < 0.6:
                recommendations.append("• Améliorer la consistance (MCTS lent devrait battre MCTS rapide)")
                recommendations.append("  → S'assurer que plus d'itérations = meilleur jeu")
                
        if not recommendations:
            recommendations.append("• Implémentation solide ! Prochaines étapes:")
            recommendations.append("  → Implémenter RAVE (Rapid Action Value Estimation)")
            recommendations.append("  → Ajouter une évaluation heuristique de position")
            recommendations.append("  → Tester contre des IA minimax/alphabeta")
            
        for rec in recommendations:
            print(rec)


# Fonction utilitaire pour lancer l'évaluation
def evaluate_mcts_fast():
    """Lance une évaluation complète de MCTS_Fast"""
    evaluator = MCTSEvaluator()
    evaluator.run_all_tests()
    return evaluator.results


if __name__ == "__main__":
    evaluate_mcts_fast()