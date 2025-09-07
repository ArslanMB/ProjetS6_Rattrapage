import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
from scipy import stats
from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2
from alphabeta_ia.alpha_beta import (
    timed_find_best_move_minimax,  
    apply_move_cached,            
    check_win_cached,             
    get_legal_moves              
)

# ============================================================================
# AGENTS UNIFIÉS 
# ============================================================================

class AgentBase:
    name: str
    def choose_move(self, game: PentagoGame) -> Tuple[Optional[Tuple[int,int,int,int]], float]:
        raise NotImplementedError
    def on_game_start(self): pass

class MinimaxAgent(AgentBase):
    def __init__(self, depth: Any = "A", time_budget: float = 2.5, label: Optional[str] = None, BOOKING: bool = True):
        self.BOOKING = BOOKING
        self.depth = depth
        self.time_budget = float(time_budget)
        self.name = label or f"Minimax_d{depth}_tb{self.time_budget:g}s"
    def choose_move(self, game: PentagoGame):
        mv, dt = timed_find_best_move_minimax(game, depth=self.depth, time_budget=self.time_budget, BOOKING=self.BOOKING)
        return mv, dt

class MCTSAgent(AgentBase):
    def __init__(self, time_limit: float = 2.5, exploration_constant: float = 1.41, label: Optional[str] = None):
        try:
            from mtcs_ia.mcts_fast import MCTS_Fast
            self.bot = MCTS_Fast(time_limit=float(time_limit), exploration_constant=float(exploration_constant))
        except ImportError:
            print("Warning: MCTS non disponible")
            self.bot = None
        self.name = label or f"MCTS_t{time_limit:g}s_C{exploration_constant:g}"
    def choose_move(self, game: PentagoGame):
        if self.bot is None:
            return None, 0.0
        t0 = time.perf_counter()
        mv = self.bot.find_best_move(game)
        dt = time.perf_counter() - t0
        return mv, dt

class GreedyAgent(AgentBase):
    def __init__(self, label: Optional[str] = None):
        try:
            from greedy_ai.greedy_ai import GreedyAI
            self.impl = GreedyAI(PLAYER_1)
        except ImportError:
            print("Warning: GreedyAI non disponible")
            self.impl = None
        self.name = label or "Greedy"

    def choose_move(self, game: PentagoGame):
        if self.impl is None:
            return None, 0.0
        side = game.current_player
        self.impl.player = side
        self.impl.opp = PLAYER_1 if side == PLAYER_2 else PLAYER_2
        t0 = time.perf_counter()
        mv = self.impl.choose_move(game)
        dt = time.perf_counter() - t0
        return mv, dt

class HybridAgent(AgentBase):
    def __init__(self, total_time: float = 3.0, top_k: int = 5, ab_depth: Any = "A", label: Optional[str] = None):
        try:
            from hybrid_ia.hybrid_ai import HybridAI
            self.impl = HybridAI(total_time=total_time, top_k=top_k, ab_depth_or_A=ab_depth)
        except ImportError:
            print("Warning: HybridAI non disponible")
            self.impl = None
        self.name = label or f"Hybrid_t{total_time:g}s_K{top_k}_d{ab_depth}"

    def choose_move(self, game: PentagoGame):
        if self.impl is None:
            return None, 0.0
        t0 = time.perf_counter()
        mv = self.impl.find_move(game)
        dt = time.perf_counter() - t0
        return mv, dt

# ============================================================================
# RÉSULTATS ET STATISTIQUES
# ============================================================================

@dataclass
class GameResult:
    winner: int                 # 1, -1, 0 (nul)
    plies: int
    p1_times: List[float]
    p2_times: List[float]
    p1_agent: str
    p2_agent: str

@dataclass
class StatResult:
    """Résultats statistiques pour une comparaison"""
    agent1: str
    agent2: str
    n_games: int
    agent1_wins: int
    agent2_wins: int
    draws: int
    
    # Parties décisives seulement
    decisive_games: int
    agent1_decisive_rate: float
    p_value: float
    is_significant: bool
    ci_lower: float
    ci_upper: float
    cohen_h: float
    effect_interpretation: str

def _apply_move(game: PentagoGame, mv: Tuple[int,int,int,int]):
    player = game.current_player
    game.board = apply_move_cached(game.board, mv, player)
    game.current_player = -player

def play_one_game(agent1: AgentBase, agent2: AgentBase, starter: int) -> GameResult:
    """Joue une partie entre deux agents"""
    game = PentagoGame()
    game.current_player = starter
    agent1.on_game_start()
    agent2.on_game_start()

    p1_times, p2_times = [], []

    for ply in range(200):
        # Terminal avant coup
        w = check_win_cached(game.board)
        if w != 0:
            return GameResult(winner=0 if w == 2 else w, plies=ply, 
                            p1_times=p1_times, p2_times=p2_times,
                            p1_agent=agent1.name, p2_agent=agent2.name)
        if not np.any(game.board == 0):
            return GameResult(winner=0, plies=ply, 
                            p1_times=p1_times, p2_times=p2_times,
                            p1_agent=agent1.name, p2_agent=agent2.name)

        agent = agent1 if game.current_player == PLAYER_1 else agent2
        mv, dt = agent.choose_move(game)

        if mv is None:
            legal = get_legal_moves(game.board)
            if not legal:
                return GameResult(winner=0, plies=ply, 
                                p1_times=p1_times, p2_times=p2_times,
                                p1_agent=agent1.name, p2_agent=agent2.name)
            mv = legal[0]
        
        _apply_move(game, mv)

        if agent is agent1: p1_times.append(float(dt))
        else:               p2_times.append(float(dt))

        w = check_win_cached(game.board)
        if w != 0:
            return GameResult(winner=0 if w == 2 else w, plies=ply+1, 
                            p1_times=p1_times, p2_times=p2_times,
                            p1_agent=agent1.name, p2_agent=agent2.name)

    return GameResult(winner=0, plies=200, 
                      p1_times=p1_times, p2_times=p2_times,
                      p1_agent=agent1.name, p2_agent=agent2.name)

# ============================================================================
# STATISTIQUES 
# ============================================================================

def summarize_times(ts: List[float]) -> Dict[str, float]:
    """Calcule avg/med/p95/max pour une liste de temps"""
    if not ts: return {"avg": 0.0, "med": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(ts); n = len(s)
    med = s[n//2] if n % 2 else 0.5*(s[n//2 - 1] + s[n//2])
    p95 = s[min(n-1, max(0, int(math.ceil(0.95*n))-1))]
    return {"avg": sum(s)/n, "med": med, "p95": p95, "max": s[-1]}

def compute_statistical_test(agent1_wins: int, agent2_wins: int) -> StatResult:
    """Calcule les tests statistiques pour parties décisives"""
    decisive_games = agent1_wins + agent2_wins
    
    if decisive_games == 0:
        return None  # Pas de parties décisives
    
    agent1_rate = agent1_wins / decisive_games
    
    # Test binomial bilatéral
    p_value = stats.binomtest(agent1_wins, decisive_games, p=0.5, alternative='two-sided')
    
    # Intervalle de confiance Wilson (95%)
    z = 1.96  # Pour 95%
    n = decisive_games
    p = agent1_rate
    
    if n > 0:
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
    else:
        ci_lower = ci_upper = 0.5
    
    # Taille d'effet Cohen h
    p1 = agent1_rate
    p2 = 1 - agent1_rate
    if p1 > 0 and p2 > 0:
        cohen_h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
    else:
        cohen_h = 0
    
    # Interprétation taille d'effet
    h_abs = abs(cohen_h)
    if h_abs < 0.2:
        effect_interp = "minime"
    elif h_abs < 0.5:
        effect_interp = "petite"
    elif h_abs < 0.8:
        effect_interp = "moyenne"
    else:
        effect_interp = "grande"
    
    return {
        'decisive_games': decisive_games,
        'agent1_decisive_rate': agent1_rate,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohen_h': cohen_h,
        'effect_interpretation': effect_interp
    }

def run_statistical_series(agent1: AgentBase, agent2: AgentBase, games: int = 100, 
                          seed: int = 42, verbose: bool = True):
    """Lance une série avec analyses statistiques complètes"""
    
    random.seed(seed)
    np.random.seed(seed)
    
    if verbose:
        print(f"\n🏆 SÉRIE STATISTIQUE: {agent1.name} vs {agent2.name} (N={games})")
        print("=" * 70)
    
    # Collecter tous les résultats
    all_results = []
    wins = {agent1.name: 0, agent2.name: 0}
    draws = 0
    times = {agent1.name: [], agent2.name: []}

    for i in range(games):
        # Alternance du starter pour équité
        if i % 2 == 0:
            p1, p2 = agent1, agent2
        else:
            p1, p2 = agent2, agent1

        res = play_one_game(p1, p2, starter=PLAYER_1)
        all_results.append(res)

        # Comptabiliser victoires par agent (pas par position)
        if res.winner == PLAYER_1:
            wins[res.p1_agent] += 1
        elif res.winner == PLAYER_2:
            wins[res.p2_agent] += 1
        else:
            draws += 1

        # Agréger temps par agent
        times[res.p1_agent] += res.p1_times
        times[res.p2_agent] += res.p2_times
        
        if verbose and (i + 1) % 20 == 0:
            print(f"Progression: {i + 1}/{games} parties")

    # Statistiques temporelles
    p1_stats = summarize_times(times[agent1.name])
    p2_stats = summarize_times(times[agent2.name])
    
    # Tests statistiques
    stat_test = compute_statistical_test(wins[agent1.name], wins[agent2.name])
    
    # Affichage des résultats
    if verbose:
        print(f"\n RÉSULTATS:")
        print(f"{agent1.name:>25}: {wins[agent1.name]:3d} victoires | "
              f"temps(avg/med/p95/max) = {p1_stats['avg']:.3f}/{p1_stats['med']:.3f}/"
              f"{p1_stats['p95']:.3f}/{p1_stats['max']:.3f}s")
        print(f"{agent2.name:>25}: {wins[agent2.name]:3d} victoires | "
              f"temps(avg/med/p95/max) = {p2_stats['avg']:.3f}/{p2_stats['med']:.3f}/"
              f"{p2_stats['p95']:.3f}/{p2_stats['max']:.3f}s")
        print(f"{'Nuls':>25}: {draws:3d} ({draws/games:.1%})")
        
        if stat_test and stat_test['decisive_games'] > 0:
            print(f"\n ANALYSE STATISTIQUE (parties décisives n={stat_test['decisive_games']}):")
            print(f"Taux victoire {agent1.name}: {stat_test['agent1_decisive_rate']:.3f}")
            print(f"IC95%: [{stat_test['ci_lower']:.3f}, {stat_test['ci_upper']:.3f}]")
            print(f"Test binomial vs p=0.5: p-value = {stat_test['p_value']:.4f}")
            print(f"Significatif (α=0.05): {'OUI' if stat_test['is_significant'] else 'NON'}")
            print(f"Taille effet (Cohen h): {stat_test['cohen_h']:.3f} ({stat_test['effect_interpretation']})")
            
            if stat_test['is_significant']:
                winner = agent1.name if stat_test['cohen_h'] > 0 else agent2.name
                print(f"→ {winner} surpasse significativement l'adversaire!")
            else:
                print(f"→ Aucune différence significative détectée.")
    
    # Retourner résultats complets
    return {
        'results': all_results,
        'wins': wins,
        'draws': draws,
        'time_stats': {agent1.name: p1_stats, agent2.name: p2_stats},
        'statistical_test': stat_test
    }

def run_mirror_test(agent_class, params: dict, games: int = 100, label: str = None):
    """Test miroir pour vérifier la cohérence d'un agent"""
    
    agent1 = agent_class(**params, label=f"{label}_A" if label else None)
    agent2 = agent_class(**params, label=f"{label}_B" if label else None)
    
    print(f"\n TEST MIROIR: {agent1.name}")
    return run_statistical_series(agent1, agent2, games=games)

def run_depth_comparison(depths: List[int], time_budget: float = 10.0, games: int = 100):
    """Compare différentes profondeurs Minimax"""
    
    print(f"\n⚡ COMPARAISON PROFONDEURS (budget {time_budget}s)")
    results = {}
    
    for d1 in depths:
        for d2 in depths:
            if d1 < d2:  # Éviter doublons
                agent1 = MinimaxAgent(depth=d1, time_budget=time_budget, 
                                    label=f"MM_d{d1}_{time_budget}s")
                agent2 = MinimaxAgent(depth=d2, time_budget=time_budget,
                                    label=f"MM_d{d2}_{time_budget}s")
                
                key = f"d{d1}_vs_d{d2}"
                results[key] = run_statistical_series(agent1, agent2, games=games)
    
    return results

# ============================================================================
# SAUVEGARDE RÉSULTATS
# ============================================================================

def save_results_to_file(results: dict, filename: str = "statistical_results.txt"):
    """Sauvegarde les résultats dans un fichier texte"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RÉSULTATS STATISTIQUES DÉTAILLÉS\n")
        f.write("=" * 80 + "\n\n")
        
        for key, result in results.items():
            f.write(f"Série: {key}\n")
            f.write("-" * 40 + "\n")
            
            wins = result['wins']
            draws = result['draws']
            time_stats = result['time_stats']
            stat_test = result['statistical_test']
            
            agent_names = list(wins.keys())
            f.write(f"{agent_names[0]}: {wins[agent_names[0]]} victoires\n")
            f.write(f"{agent_names[1]}: {wins[agent_names[1]]} victoires\n")
            f.write(f"Nuls: {draws}\n")
            
            if stat_test:
                f.write(f"P-value: {stat_test['p_value']:.4f}\n")
                f.write(f"Significatif: {stat_test['is_significant']}\n")
                f.write(f"Taille effet: {stat_test['cohen_h']:.3f} ({stat_test['effect_interpretation']})\n")
            
            f.write("\n")
    
    print(f"Résultats sauvés dans: {filename}")

# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    
    print(" Système d'analyse statistique pour Pentago")
    print("=" * 50)
    
    # Exemple 1: Test miroir Minimax
    print("\n1️ Test miroir Minimax d=2")
    mirror_result = run_mirror_test(
        MCTSAgent, 
        {'time_limit': 2.5, 'exploration_constant': 0.7},
        games=10,
        label="MCTS_0.7_2.5s"
    )
    
    # Exemple 2: Comparaison profondeurs
    #print("\n2️ Comparaison profondeurs")
    #depth_results = run_depth_comparison([1, 2, 3], time_budget=5.0, games=20)
    
    # Exemple 3: Minimax vs autres agents (si disponibles)
    try:
        mm_agent = MinimaxAgent(depth=2, time_budget=2.5, label="MM_d2_3s")
        greedy_agent = MCTSAgent(time_limit=2.5, exploration_constant=0.7,label="MCTS_0.7_2.5s")
        
        print("\n3️ Minimax vs MCTS")
        mm_vs_greedy = run_statistical_series(mm_agent, greedy_agent, games=10)
        
    except Exception as e:
        print(f"Certains agents non disponibles: {e}")
    
    print("\n Analyses terminées!")