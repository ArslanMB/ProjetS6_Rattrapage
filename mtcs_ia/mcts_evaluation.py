# mcts_evaluation.py
import numpy as np
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt

from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2, BOARD_ROWS, BOARD_COLS, QUADRANT_SIZE
from mcts import MCTS_AI


class MCTSEvaluator:
    """
    Suite complète de tests pour évaluer la qualité de l'implémentation MCTS
    """
    
    def __init__(self):
        self.results = {}
        
    def run_all_tests(self):
        """Lance tous les tests d'évaluation"""
        print("=== ÉVALUATION MCTS ===")
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
        mcts = MCTS_AI(iteration_limit=100)
        mcts.find_best_move(game)
        
        # Vérifier que les statistiques sont cohérentes
        root = mcts.root
        total_child_visits = sum(child.visits for child in root.children)
        
        results['backprop_consistency'] = abs(root.visits - 1 - total_child_visits) <= 1
        
        # Test 2: Respect des règles de jeu
        results['rule_compliance'] = self._test_rule_compliance()
        
        # Test 3: Gestion des cas limites
        results['edge_cases'] = self._test_edge_cases()
        
        self.results['unit_tests'] = results
        print(f"Tests unitaires: {sum(results.values())}/{len(results)} passés")
        
    def _test_rule_compliance(self):
        """Teste si MCTS respecte les règles du jeu"""
        game = PentagoGame()
        mcts = MCTS_AI(time_limit=1.0)
        
        for _ in range(10):  # 10 coups de test
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
                    game.update_rotation_animation()
                    while game.game_phase == "ANIMATING_ROTATION":
                        game.update_rotation_animation()
                        
            if game.game_state != 'PLAYING':
                break
                
        return True
        
    def _test_edge_cases(self):
        """Teste les cas limites"""
        # Plateau presque plein
        game = PentagoGame()
        game.board = np.ones((6, 6)) * PLAYER_1
        game.board[0, 0] = 0  # Une seule case libre
        game.board[0, 1] = 0
        
        mcts = MCTS_AI(time_limit=0.5)
        move = mcts.find_best_move(game)
        
        # Doit retourner un coup valide
        return move is not None and game.board[move[0], move[1]] == 0
        
    def test_performance(self):
        """Tests de performance"""
        results = {}
        
        game = PentagoGame()
        
        # Test vitesse de base
        times = []
        for time_limit in [0.5, 1.0, 2.0, 5.0]:
            mcts = MCTS_AI(time_limit=time_limit)
            
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
        mcts = MCTS_AI(time_limit=3.0)
        
        start_time = time.time()
        mcts.find_best_move(game)
        elapsed = time.time() - start_time
        
        # Estimer iterations depuis les visites du root
        iterations = mcts.root.visits if mcts.root else 0
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
        
        mcts = MCTS_AI(time_limit=2.0)
        move = mcts.find_best_move(game)
        
        if not move:
            return False
            
        r, c, quad_idx, direction = move
        
        # Vérifier si le coup complète la ligne
        return r == 0 and c == 4
        
    def _test_block_opponent(self):
        """Teste si MCTS bloque une victoire adverse"""
        game = PentagoGame()
        # Position où PLAYER_1 menace de gagner
        game.board = np.array([
            [-1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        game.current_player = PLAYER_1  # P1 doit bloquer P2
        
        mcts = MCTS_AI(time_limit=2.0, exploration_constant=0.7)
        move = mcts.find_best_move(game)
        
        if not move:
            return False
            
        r, c, quad_idx, direction = move
        
        # Vérifier si le coup bloque la menace
        return r == 0 and c == 4
        
    def _test_opening_play(self):
        """Teste les préférences d'ouverture"""
        game = PentagoGame()
        mcts = MCTS_AI(time_limit=1.0, exploration_constant=1.0)
        
        moves = []
        for _ in range(5):  # 5 parties différentes
            game.reset_game()
            game.game_state = 'PLAYING'
            move = mcts.find_best_move(game)
            if move:
                moves.append((move[0], move[1]))
                
        # Vérifier si préfère cases centrales (2,2), (2,3), (3,2), (3,3)
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
                mcts = MCTS_AI(**config)
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
        
        for game_num in range(num_games):
            game = PentagoGame()
            game.game_state = 'PLAYING'
            mcts = MCTS_AI(time_limit=1.0)
            
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
                            # Simuler animation
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
                
        return wins / num_games
        
    def _play_self_matches(self, num_games):
        """MCTS contre lui-même avec différents temps"""
        scores = []
        
        for _ in range(num_games):
            game = PentagoGame()
            game.game_state = 'PLAYING'
            
            mcts_fast = MCTS_AI(time_limit=1.0)
            mcts_slow = MCTS_AI(time_limit=3.0)
            
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
                
        return np.mean(scores)
        
    def generate_report(self):
        """Génère un rapport d'évaluation complet"""
        print("\n" + "="*50)
        print("RAPPORT D'ÉVALUATION MCTS")
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
                    if 'score' in test_name or 'vs_' in test_name:
                        score = result
                        max_val = 1.0
                        status = f"{result:.1%}"
                    else:
                        score = min(result / 1000, 1)  # Normaliser
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
        print("\nRECOMMANDANTIONS:")
        self._generate_recommendations()
        
    def _generate_recommendations(self):
        """Génère des recommandations d'amélioration"""
        recommendations = []
        
        # Analyser les résultats
        if 'performance' in self.results:
            perf = self.results['performance']
            if perf.get('iterations_per_second', 0) < 200:
                recommendations.append("• Optimiser la vitesse de simulation (< 200 it/s)")
                
        if 'tactical' in self.results:
            tact = self.results['tactical']
            failed_tactics = [k for k, v in tact.items() if not v]
            if failed_tactics:
                recommendations.append(f"• Améliorer tactiques: {', '.join(failed_tactics)}")
                
        if 'comparative' in self.results:
            comp = self.results['comparative']
            if comp.get('vs_random', 0) < 0.8:
                recommendations.append("• Renforcer contre adversaires faibles (< 80%)")
                
        if not recommendations:
            recommendations.append("• Implémentation solide ! Tester contre IA plus fortes")
            
        for rec in recommendations:
            print(rec)


# Fonction utilitaire pour lancer l'évaluation
def evaluate_mcts():
    """Lance une évaluation complète de MCTS"""
    evaluator = MCTSEvaluator()
    evaluator.run_all_tests()
    return evaluator.results


if __name__ == "__main__":
    evaluate_mcts()