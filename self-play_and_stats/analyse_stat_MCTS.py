"""
Analyse Statistique Complète pour MCTS dans Pentago
=====================================================
Ce script fournit une analyse statistique rigoureuse de ton agent MCTS
avec tous les tests, métriques et visualisations nécessaires.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import math
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
from scipy import stats

# Import de tes classes existantes
from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2
from mtcs_ia.mcts_fast import MCTS_Fast
from greedy_ai.greedy_ai import GreedyAI
from hybrid_ia.hybrid_ai import HybridAI
from alphabeta_ia.alpha_beta import (
    timed_find_best_move_minimax,
    apply_move_cached,
    check_win_cached,
    get_legal_moves
)

# ============= IMPORT DES AGENTS EXISTANTS =============
# (Je reprends tes classes AgentBase, MCTSAgent, MinimaxAgent, etc.)

class AgentBase:
    name: str
    def choose_move(self, game: PentagoGame) -> Tuple[Optional[Tuple[int,int,int,int]], float]:
        raise NotImplementedError
    def on_game_start(self): pass

class MCTSAgent(AgentBase):
    def __init__(self, time_limit: float = 2.5, exploration_constant: float = 1.41, label: Optional[str] = None):
        self.bot = MCTS_Fast(time_limit=float(time_limit), exploration_constant=float(exploration_constant))
        self.name = label or f"MCTS_t{time_limit:g}s_C{exploration_constant:g}"
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
    
    def choose_move(self, game: PentagoGame):
        t0 = time.perf_counter()
        mv = self.bot.find_best_move(game)
        dt = time.perf_counter() - t0
        return mv, dt

class MinimaxAgent(AgentBase):
    def __init__(self, depth: Any = "A", time_budget: float = 2.5, label: Optional[str] = None, BOOKING: bool = True):
        self.BOOKING = BOOKING
        self.depth = depth
        self.time_budget = float(time_budget)
        self.name = label or f"Minimax_d{depth}_tb{self.time_budget:g}s"
    
    def choose_move(self, game: PentagoGame):
        mv, dt = timed_find_best_move_minimax(game, depth=self.depth, time_budget=self.time_budget, BOOKING=self.BOOKING)
        return mv, dt

class GreedyAgent(AgentBase):
    def __init__(self, label: Optional[str] = None):
        self.impl = GreedyAI(PLAYER_1)
        self.name = label or "Greedy"
    
    def choose_move(self, game: PentagoGame):
        side = game.current_player
        self.impl.player = side
        self.impl.opp = PLAYER_1 if side == PLAYER_2 else PLAYER_2
        t0 = time.perf_counter()
        mv = self.impl.choose_move(game)
        dt = time.perf_counter() - t0
        return mv, dt

# ============= STRUCTURE DE DONNÉES ENRICHIE =============

@dataclass
class DetailedGameResult:
    """Structure enrichie pour stocker tous les détails d'une partie"""
    game_id: int
    agent1_name: str
    agent2_name: str
    winner: str  # 'agent1', 'agent2', ou 'draw'
    starter: str  # qui a commencé
    total_moves: int
    total_duration: float
    agent1_avg_time: float
    agent2_avg_time: float
    agent1_max_time: float
    agent2_max_time: float
    seed: int
    
    def to_dict(self):
        return asdict(self)

# ============= MOTEUR DE JEU AMÉLIORÉ =============

def _apply_move(game: PentagoGame, mv: Tuple[int,int,int,int]):
    """Applique un coup sur le plateau"""
    player = game.current_player
    game.board = apply_move_cached(game.board, mv, player)
    game.current_player = -player

def play_single_game(agent1: AgentBase, agent2: AgentBase, 
                    starter: int, seed: int) -> DetailedGameResult:
    """
    Joue une seule partie entre deux agents avec seed fixé
    """
    # Initialiser les générateurs aléatoires avec le seed
    random.seed(seed)
    np.random.seed(seed)
    
    game = PentagoGame()
    game.current_player = starter
    
    agent1.on_game_start()
    agent2.on_game_start()
    
    p1_times, p2_times = [], []
    start_time = time.perf_counter()
    
    for ply in range(200):  # Limite de sécurité
        # Vérifier si partie terminée avant le coup
        w = check_win_cached(game.board)
        if w != 0:
            winner = 'draw' if w == 2 else ('agent1' if w == PLAYER_1 else 'agent2')
            break
        
        if not np.any(game.board == 0):  # Plateau plein
            winner = 'draw'
            break
        
        # Déterminer quel agent joue
        if game.current_player == PLAYER_1:
            current_agent = agent1
            is_agent1 = True
        else:
            current_agent = agent2
            is_agent1 = False
        
        # Choisir et appliquer le coup
        mv, dt = current_agent.choose_move(game)
        
        if mv is None:  # Sécurité si l'agent ne retourne pas de coup
            legal = get_legal_moves(game.board)
            if not legal:
                winner = 'draw'
                break
            mv = legal[0]
        
        _apply_move(game, mv)
        
        # Enregistrer le temps
        if is_agent1:
            p1_times.append(dt)
        else:
            p2_times.append(dt)
        
        # Vérifier victoire après le coup
        w = check_win_cached(game.board)
        if w != 0:
            winner = 'draw' if w == 2 else ('agent1' if w == PLAYER_1 else 'agent2')
            break
    else:
        winner = 'draw'  # Si on atteint 200 coups
    
    total_duration = time.perf_counter() - start_time
    
    # Calculer les statistiques de temps
    agent1_avg = np.mean(p1_times) if p1_times else 0
    agent2_avg = np.mean(p2_times) if p2_times else 0
    agent1_max = max(p1_times) if p1_times else 0
    agent2_max = max(p2_times) if p2_times else 0
    
    return DetailedGameResult(
        game_id=0,  # sera mis à jour
        agent1_name=agent1.name,
        agent2_name=agent2.name,
        winner=winner,
        starter='agent1' if starter == PLAYER_1 else 'agent2',
        total_moves=ply + 1,
        total_duration=total_duration,
        agent1_avg_time=agent1_avg,
        agent2_avg_time=agent2_avg,
        agent1_max_time=agent1_max,
        agent2_max_time=agent2_max,
        seed=seed
    )

# ============= EXPÉRIMENTATION STATISTIQUE =============

class StatisticalExperiment:
    """
    Classe principale pour réaliser l'analyse statistique complète
    """
    
    def __init__(self, output_dir: str = "mcts_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results_df = None
        
    def run_experiment(self, agent1: AgentBase, agent2: AgentBase, 
                      n_games: int = 500, base_seed: int = 42,
                      alternate_starter: bool = True) -> pd.DataFrame:
        """
        Lance une série de parties avec randomisation contrôlée
        
        Args:
            agent1: Premier agent
            agent2: Deuxième agent
            n_games: Nombre de parties à jouer
            base_seed: Seed de base pour la reproductibilité
            alternate_starter: Si True, alterne qui commence
        
        Returns:
            DataFrame avec tous les résultats
        """
        print(f"\n{'='*60}")
        print(f"EXPÉRIENCE: {agent1.name} vs {agent2.name}")
        print(f"Nombre de parties: {n_games}")
        print(f"{'='*60}\n")
        
        results = []
        rng = random.Random(base_seed)
        
        for i in range(n_games):
            # Générer un seed unique pour cette partie
            game_seed = rng.randint(0, 2**30)
            
            # Déterminer qui commence
            if alternate_starter:
                # Alterne qui commence
                if i % 2 == 0:
                    starter = PLAYER_1
                    agents_order = (agent1, agent2)
                else:
                    # On échange les agents pour que chacun joue des deux côtés
                    starter = PLAYER_1
                    agents_order = (agent2, agent1)
            else:
                # Randomise qui commence
                if rng.random() < 0.5:
                    starter = PLAYER_1
                    agents_order = (agent1, agent2)
                else:
                    starter = PLAYER_1
                    agents_order = (agent2, agent1)
            
            # Jouer la partie
            result = play_single_game(agents_order[0], agents_order[1], 
                                    starter, game_seed)
            
            # Ajuster le nom du gagnant selon l'ordre des agents
            if agents_order == (agent2, agent1):
                # On a inversé les agents, donc on doit inverser le résultat
                if result.winner == 'agent1':
                    result.winner = agent2.name
                elif result.winner == 'agent2':
                    result.winner = agent1.name
                else:
                    result.winner = 'draw'
                
                # Corriger aussi les noms et temps
                result.agent1_name = agent1.name
                result.agent2_name = agent2.name
                result.agent1_avg_time, result.agent2_avg_time = result.agent2_avg_time, result.agent1_avg_time
                result.agent1_max_time, result.agent2_max_time = result.agent2_max_time, result.agent1_max_time
            else:
                if result.winner == 'agent1':
                    result.winner = agent1.name
                elif result.winner == 'agent2':
                    result.winner = agent2.name
            
            result.game_id = i
            results.append(result.to_dict())
            
            # Affichage de progression
            if (i + 1) % 50 == 0:
                print(f"  Parties jouées: {i+1}/{n_games}")
        
        # Créer DataFrame
        df = pd.DataFrame(results)
        self.results_df = df
        
        # Sauvegarder les résultats bruts
        csv_path = os.path.join(self.output_dir, f"{agent1.name}_vs_{agent2.name}_raw.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nRésultats sauvegardés dans: {csv_path}")
        
        return df
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calcule toutes les statistiques importantes
        """
        stats = {}
        
        agent1_name = df['agent1_name'].iloc[0]
        agent2_name = df['agent2_name'].iloc[0]
        
        # ===== 1. WIN RATES =====
        total_games = len(df)
        agent1_wins = (df['winner'] == agent1_name).sum()
        agent2_wins = (df['winner'] == agent2_name).sum()
        draws = (df['winner'] == 'draw').sum()
        
        # Win rate avec intervalle de confiance Wilson
        agent1_wr, agent1_ci_low, agent1_ci_high = self._wilson_ci(agent1_wins, total_games)
        agent2_wr, agent2_ci_low, agent2_ci_high = self._wilson_ci(agent2_wins, total_games)
        
        stats['win_rates'] = {
            agent1_name: {
                'wins': int(agent1_wins),
                'rate': agent1_wr,
                'ci_low': agent1_ci_low,
                'ci_high': agent1_ci_high
            },
            agent2_name: {
                'wins': int(agent2_wins),
                'rate': agent2_wr,
                'ci_low': agent2_ci_low,
                'ci_high': agent2_ci_high
            },
            'draws': {
                'count': int(draws),
                'rate': draws / total_games
            }
        }
        
        # ===== 2. TEST DE PERMUTATION =====
        # Pour la différence de win rate
        outcomes_a1 = (df['winner'] == agent1_name).astype(int).values
        outcomes_a2 = (df['winner'] == agent2_name).astype(int).values
        
        diff_obs, p_value = self._permutation_test(outcomes_a1, outcomes_a2)
        
        stats['permutation_test'] = {
            'observed_diff': diff_obs,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # ===== 3. BOOTSTRAP CI pour la différence =====
        diff_ci = self._bootstrap_diff_ci(outcomes_a1, outcomes_a2)
        stats['bootstrap_diff_ci'] = diff_ci
        
        # ===== 4. ANALYSE PAR PREMIER JOUEUR =====
        starter_stats = {}
        for starter in ['agent1', 'agent2']:
            starter_games = df[df['starter'] == starter]
            if len(starter_games) > 0:
                wins = (starter_games['winner'] == agent1_name).sum()
                total = len(starter_games)
                wr, ci_low, ci_high = self._wilson_ci(wins, total)
                starter_stats[f"{agent1_name}_as_{starter}"] = {
                    'games': int(total),
                    'wins': int(wins),
                    'win_rate': wr,
                    'ci': (ci_low, ci_high)
                }
        stats['by_starter'] = starter_stats
        
        # ===== 5. STATISTIQUES DE TEMPS =====
        stats['time_stats'] = {
            agent1_name: {
                'avg_move_time': float(df['agent1_avg_time'].mean()),
                'std_move_time': float(df['agent1_avg_time'].std()),
                'max_move_time': float(df['agent1_max_time'].max()),
                'median_move_time': float(df['agent1_avg_time'].median())
            },
            agent2_name: {
                'avg_move_time': float(df['agent2_avg_time'].mean()),
                'std_move_time': float(df['agent2_avg_time'].std()),
                'max_move_time': float(df['agent2_max_time'].max()),
                'median_move_time': float(df['agent2_avg_time'].median())
            }
        }
        
        # ===== 6. DURÉE DES PARTIES =====
        stats['game_length'] = {
            'mean_moves': float(df['total_moves'].mean()),
            'std_moves': float(df['total_moves'].std()),
            'min_moves': int(df['total_moves'].min()),
            'max_moves': int(df['total_moves'].max()),
            'median_moves': float(df['total_moves'].median())
        }
        
        # ===== 7. ELO ESTIMATION =====
        elo_scores = self._compute_elo(df)
        stats['elo'] = elo_scores
        
        return stats
    
    def _wilson_ci(self, wins: int, n: int, alpha: float = 0.05):
        """Calcule l'intervalle de confiance Wilson pour une proportion"""
        if n == 0:
            return 0, 0, 0
        
        ci_low, ci_high = proportion_confint(wins, n, alpha=alpha, method='wilson')
        return wins/n, ci_low, ci_high
    
    def _permutation_test(self, outcomes1: np.ndarray, outcomes2: np.ndarray, 
                         n_perms: int = 10000) -> Tuple[float, float]:
        """Test de permutation pour différence de proportions"""
        obs_diff = outcomes1.mean() - outcomes2.mean()
        pooled = np.concatenate([outcomes1, outcomes2])
        n1 = len(outcomes1)
        
        perm_diffs = []
        rng = np.random.default_rng(42)
        
        for _ in range(n_perms):
            rng.shuffle(pooled)
            new1 = pooled[:n1]
            new2 = pooled[n1:]
            perm_diffs.append(new1.mean() - new2.mean())
        
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(np.abs(perm_diffs) >= abs(obs_diff))
        
        return obs_diff, p_value
    
    def _bootstrap_diff_ci(self, outcomes1: np.ndarray, outcomes2: np.ndarray,
                           n_boot: int = 10000, alpha: float = 0.05):
        """Bootstrap CI pour différence de win rates"""
        diffs = []
        rng = np.random.default_rng(42)
        n1, n2 = len(outcomes1), len(outcomes2)
        
        for _ in range(n_boot):
            sample1 = rng.choice(outcomes1, size=n1, replace=True)
            sample2 = rng.choice(outcomes2, size=n2, replace=True)
            diffs.append(sample1.mean() - sample2.mean())
        
        diffs = np.array(diffs)
        lo = np.percentile(diffs, 100 * alpha/2)
        hi = np.percentile(diffs, 100 * (1 - alpha/2))
        
        return {
            'mean_diff': float(outcomes1.mean() - outcomes2.mean()),
            'ci_low': float(lo),
            'ci_high': float(hi),
            'includes_zero': lo <= 0 <= hi
        }
    
    def _compute_elo(self, df: pd.DataFrame, initial: int = 1500, k: int = 20):
        """Calcule les scores Elo basés sur les résultats"""
        agent1_name = df['agent1_name'].iloc[0]
        agent2_name = df['agent2_name'].iloc[0]
        
        elos = {agent1_name: initial, agent2_name: initial}
        history = {'games': [], agent1_name: [], agent2_name: []}
        
        for idx, row in df.iterrows():
            winner = row['winner']
            
            # Score pour agent1
            if winner == agent1_name:
                score1 = 1.0
            elif winner == agent2_name:
                score1 = 0.0
            else:  # draw
                score1 = 0.5
            
            # Mise à jour Elo
            E1 = 1 / (1 + 10 ** ((elos[agent2_name] - elos[agent1_name]) / 400))
            
            elos[agent1_name] += k * (score1 - E1)
            elos[agent2_name] += k * ((1 - score1) - (1 - E1))
            
            # Historique
            history['games'].append(idx)
            history[agent1_name].append(elos[agent1_name])
            history[agent2_name].append(elos[agent2_name])
        
        return {
            'final_scores': elos,
            'history': history,
            'difference': elos[agent1_name] - elos[agent2_name]
        }
    
    def generate_visualizations(self, df: pd.DataFrame, stats: Dict):
        """Génère tous les graphiques d'analyse"""
        agent1_name = df['agent1_name'].iloc[0]
        agent2_name = df['agent2_name'].iloc[0]
        
        # Configuration matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 12))
        
        # ===== 1. WIN RATES avec CI =====
        ax1 = plt.subplot(2, 3, 1)
        agents = [agent1_name, agent2_name, 'Draws']
        wins = [
            stats['win_rates'][agent1_name]['wins'],
            stats['win_rates'][agent2_name]['wins'],
            stats['win_rates']['draws']['count']
        ]
        colors = ['#2E7D32', '#C62828', '#757575']
        bars = ax1.bar(agents, wins, color=colors, alpha=0.7, edgecolor='black')
        
        # Ajouter les CI pour win rates
        for i, agent in enumerate([agent1_name, agent2_name]):
            if agent in stats['win_rates']:
                ci_low = stats['win_rates'][agent]['ci_low'] * len(df)
                ci_high = stats['win_rates'][agent]['ci_high'] * len(df)
                ax1.errorbar(i, wins[i], 
                           yerr=[[wins[i] - ci_low], [ci_high - wins[i]]],
                           fmt='none', color='black', capsize=5, capthick=2)
        
        ax1.set_ylabel('Nombre de victoires', fontsize=12)
        ax1.set_title('Résultats avec IC Wilson 95%', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Ajouter les pourcentages sur les barres
        for bar, win in zip(bars, wins):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{win}\n({100*win/len(df):.1f}%)',
                    ha='center', va='bottom', fontsize=10)
        
        # ===== 2. ÉVOLUTION ELO =====
        ax2 = plt.subplot(2, 3, 2)
        elo_history = stats['elo']['history']
        ax2.plot(elo_history['games'], elo_history[agent1_name], 
                label=agent1_name, linewidth=2, color='#2E7D32')
        ax2.plot(elo_history['games'], elo_history[agent2_name], 
                label=agent2_name, linewidth=2, color='#C62828')
        ax2.axhline(y=1500, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Partie #', fontsize=12)
        ax2.set_ylabel('Score Elo', fontsize=12)
        ax2.set_title('Évolution Elo', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # ===== 3. DISTRIBUTION DES TEMPS =====
        ax3 = plt.subplot(2, 3, 3)
        times_data = [df['agent1_avg_time'], df['agent2_avg_time']]
        bp = ax3.boxplot(times_data, labels=[agent1_name, agent2_name],
                         patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], ['#2E7D32', '#C62828']):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        ax3.set_ylabel('Temps moyen par coup (s)', fontsize=12)
        ax3.set_title('Distribution des temps de réflexion', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ===== 4. LONGUEUR DES PARTIES =====
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(df['total_moves'], bins=30, color='#1976D2', alpha=0.7, edgecolor='black')
        ax4.axvline(df['total_moves'].mean(), color='red', linestyle='--', 
                   label=f'Moyenne: {df["total_moves"].mean():.1f}')
        ax4.axvline(df['total_moves'].median(), color='green', linestyle='--',
                   label=f'Médiane: {df["total_moves"].median():.1f}')
        ax4.set_xlabel('Nombre de coups', fontsize=12)
        ax4.set_ylabel('Fréquence', fontsize=12)
        ax4.set_title('Distribution de la longueur des parties', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # ===== 5. WIN RATE PAR POSITION DE DÉPART =====
        ax5 = plt.subplot(2, 3, 5)
        starter_data = []
        starter_labels = []
        
        for starter in ['agent1', 'agent2']:
            key = f"{agent1_name}_as_{starter}"
            if key in stats['by_starter']:
                starter_data.append(stats['by_starter'][key]['win_rate'])
                starter_labels.append(f"{agent1_name}\ncommence" if starter == 'agent1' 
                                    else f"{agent2_name}\ncommence")
        
        if starter_data:
            bars = ax5.bar(starter_labels, starter_data, color=['#4CAF50', '#FF9800'], 
                          alpha=0.7, edgecolor='black')
            ax5.set_ylabel(f'Win rate de {agent1_name}', fontsize=12)
            ax5.set_title('Influence du premier joueur', fontsize=14, fontweight='bold')
            ax5.set_ylim([0, 1])
            ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax5.grid(True, alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for bar, val in zip(bars, starter_data):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # ===== 6. BOOTSTRAP DISTRIBUTION =====
        ax6 = plt.subplot(2, 3, 6)
        
        # Simuler la distribution bootstrap
        outcomes1 = (df['winner'] == agent1_name).astype(int).values
        outcomes2 = (df['winner'] == agent2_name).astype(int).values
        
        boot_diffs = []
        rng = np.random.default_rng(42)
        for _ in range(1000):  # Moins pour la visualisation
            s1 = rng.choice(outcomes1, size=len(outcomes1), replace=True)
            s2 = rng.choice(outcomes2, size=len(outcomes2), replace=True)
            boot_diffs.append(s1.mean() - s2.mean())
        
        ax6.hist(boot_diffs, bins=30, color='#9C27B0', alpha=0.7, edgecolor='black')
        ax6.axvline(stats['bootstrap_diff_ci']['mean_diff'], color='red', 
                   linestyle='-', linewidth=2, label='Différence observée')
        ax6.axvline(stats['bootstrap_diff_ci']['ci_low'], color='orange', 
                   linestyle='--', label='IC 95%')
        ax6.axvline(stats['bootstrap_diff_ci']['ci_high'], color='orange', 
                   linestyle='--')
        ax6.axvline(0, color='black', linestyle=':', alpha=0.5)
        ax6.set_xlabel('Différence de win rate', fontsize=12)
        ax6.set_ylabel('Fréquence', fontsize=12)
        ax6.set_title('Distribution Bootstrap de la différence', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Analyse Statistique: {agent1_name} vs {agent2_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = os.path.join(self.output_dir, f"{agent1_name}_vs_{agent2_name}_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Graphiques sauvegardés dans: {plot_path}")
    
    def generate_report(self, df: pd.DataFrame, stats: Dict):
        """Génère un rapport textuel complet"""
        agent1_name = df['agent1_name'].iloc[0]
        agent2_name = df['agent2_name'].iloc[0]
        
        report = []
        report.append("=" * 80)
        report.append(f"RAPPORT D'ANALYSE STATISTIQUE COMPLET")
        report.append(f"{agent1_name} vs {agent2_name}")
        report.append(f"Nombre de parties: {len(df)}")
        report.append("=" * 80)
        report.append("")
        
        # 1. RÉSULTATS PRINCIPAUX
        report.append("1. RÉSULTATS PRINCIPAUX")
        report.append("-" * 40)
        wr1 = stats['win_rates'][agent1_name]
        wr2 = stats['win_rates'][agent2_name]
        draws = stats['win_rates']['draws']
        
        report.append(f"  {agent1_name}:")
        report.append(f"    - Victoires: {wr1['wins']}/{len(df)} ({wr1['rate']:.3f})")
        report.append(f"    - IC Wilson 95%: [{wr1['ci_low']:.3f}, {wr1['ci_high']:.3f}]")
        report.append(f"  {agent2_name}:")
        report.append(f"    - Victoires: {wr2['wins']}/{len(df)} ({wr2['rate']:.3f})")
        report.append(f"    - IC Wilson 95%: [{wr2['ci_low']:.3f}, {wr2['ci_high']:.3f}]")
        report.append(f"  Matchs nuls: {draws['count']} ({draws['rate']:.3f})")
        report.append("")
        
        # 2. TESTS STATISTIQUES
        report.append("2. TESTS STATISTIQUES")
        report.append("-" * 40)
        perm = stats['permutation_test']
        report.append(f"  Test de permutation:")
        report.append(f"    - Différence observée: {perm['observed_diff']:.4f}")
        report.append(f"    - p-value: {perm['p_value']:.4f}")
        report.append(f"    - Significatif (α=0.05): {'OUI ✓' if perm['significant'] else 'NON ✗'}")
        
        boot = stats['bootstrap_diff_ci']
        report.append(f"  Bootstrap CI pour différence:")
        report.append(f"    - Différence moyenne: {boot['mean_diff']:.4f}")
        report.append(f"    - IC 95%: [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
        report.append(f"    - Contient zéro: {'OUI (pas de différence significative)' if boot['includes_zero'] else 'NON (différence significative)'}")
        report.append("")
        
        # 3. SCORES ELO
        report.append("3. SCORES ELO")
        report.append("-" * 40)
        elo = stats['elo']
        report.append(f"  Scores finaux:")
        report.append(f"    - {agent1_name}: {elo['final_scores'][agent1_name]:.0f}")
        report.append(f"    - {agent2_name}: {elo['final_scores'][agent2_name]:.0f}")
        report.append(f"  Différence Elo: {elo['difference']:.0f}")
        
        # Interprétation de la différence Elo
        abs_diff = abs(elo['difference'])
        if abs_diff < 50:
            interpretation = "Forces équivalentes"
        elif abs_diff < 100:
            interpretation = "Légère supériorité"
        elif abs_diff < 200:
            interpretation = "Supériorité claire"
        else:
            interpretation = "Domination nette"
        report.append(f"  Interprétation: {interpretation}")
        report.append("")
        
        # 4. ANALYSE TEMPORELLE
        report.append("4. ANALYSE TEMPORELLE")
        report.append("-" * 40)
        time1 = stats['time_stats'][agent1_name]
        time2 = stats['time_stats'][agent2_name]
        
        report.append(f"  {agent1_name}:")
        report.append(f"    - Temps moyen/coup: {time1['avg_move_time']:.3f}s")
        report.append(f"    - Écart-type: {time1['std_move_time']:.3f}s")
        report.append(f"    - Médiane: {time1['median_move_time']:.3f}s")
        report.append(f"    - Maximum: {time1['max_move_time']:.3f}s")
        
        report.append(f"  {agent2_name}:")
        report.append(f"    - Temps moyen/coup: {time2['avg_move_time']:.3f}s")
        report.append(f"    - Écart-type: {time2['std_move_time']:.3f}s")
        report.append(f"    - Médiane: {time2['median_move_time']:.3f}s")
        report.append(f"    - Maximum: {time2['max_move_time']:.3f}s")
        report.append("")
        
        # 5. LONGUEUR DES PARTIES
        report.append("5. LONGUEUR DES PARTIES")
        report.append("-" * 40)
        length = stats['game_length']
        report.append(f"  Nombre de coups moyen: {length['mean_moves']:.1f} (±{length['std_moves']:.1f})")
        report.append(f"  Médiane: {length['median_moves']:.0f}")
        report.append(f"  Min-Max: [{length['min_moves']}, {length['max_moves']}]")
        report.append("")
        
        # 6. INFLUENCE DU PREMIER JOUEUR
        report.append("6. INFLUENCE DU PREMIER JOUEUR")
        report.append("-" * 40)
        for key, data in stats['by_starter'].items():
            report.append(f"  {key}:")
            report.append(f"    - Parties: {data['games']}")
            report.append(f"    - Victoires: {data['wins']}")
            report.append(f"    - Win rate: {data['win_rate']:.3f}")
            report.append(f"    - IC 95%: [{data['ci'][0]:.3f}, {data['ci'][1]:.3f}]")
        report.append("")
        
        # 7. CONCLUSION
        report.append("7. CONCLUSION")
        report.append("-" * 40)
        
        # Déterminer le vainqueur statistique
        if not boot['includes_zero']:
            if boot['mean_diff'] > 0:
                winner = agent1_name
                conf = "avec différence statistiquement significative"
            else:
                winner = agent2_name
                conf = "avec différence statistiquement significative"
        else:
            if wr1['rate'] > wr2['rate']:
                winner = agent1_name
                conf = "mais sans différence statistiquement significative"
            elif wr2['rate'] > wr1['rate']:
                winner = agent2_name
                conf = "mais sans différence statistiquement significative"
            else:
                winner = "Égalité"
                conf = ""
        
        if winner != "Égalité":
            report.append(f"  → {winner} est supérieur {conf}")
            report.append(f"  → p-value = {perm['p_value']:.4f}")
            report.append(f"  → Taille d'effet = {abs(boot['mean_diff']):.3f}")
        else:
            report.append(f"  → Les deux agents sont de force équivalente")
        
        report.append("")
        report.append("=" * 80)
        
        # Sauvegarder le rapport
        report_text = "\n".join(report)
        report_path = os.path.join(self.output_dir, f"{agent1_name}_vs_{agent2_name}_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nRapport sauvegardé dans: {report_path}")
        
        return report_text

# ============= ANALYSES SPÉCIFIQUES POUR MCTS =============

class MCTSSpecificAnalysis:
    """Analyses spécifiques pour les paramètres MCTS"""
    
    def __init__(self, output_dir: str = "mcts_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def exploration_constant_analysis(self, time_limit: float = 2.5, 
                                     c_values: List[float] = None,
                                     baseline_agent: AgentBase = None,
                                     n_games_per_config: int = 100):
        """
        Analyse l'impact de la constante d'exploration C
        """
        if c_values is None:
            c_values = [0.5, 0.7, 1.0, 1.41, 2.0, 2.5]
        
        if baseline_agent is None:
            baseline_agent = MinimaxAgent(depth=2, time_budget=1.0, label="Minimax_d2_tb1s")
        
        results = []
        
        for c in c_values:
            print(f"\nTest avec C = {c}")
            mcts = MCTSAgent(time_limit=time_limit, exploration_constant=c,
                            label=f"MCTS_C{c}")
            
            exp = StatisticalExperiment(self.output_dir)
            df = exp.run_experiment(mcts, baseline_agent, 
                                  n_games=n_games_per_config)
            stats = exp.calculate_statistics(df)
            
            results.append({
                'C': c,
                'win_rate': stats['win_rates'][mcts.name]['rate'],
                'ci_low': stats['win_rates'][mcts.name]['ci_low'],
                'ci_high': stats['win_rates'][mcts.name]['ci_high'],
                'avg_time': stats['time_stats'][mcts.name]['avg_move_time']
            })
        
        # Créer graphique
        self._plot_exploration_analysis(results)
        
        return pd.DataFrame(results)
    
    def time_budget_analysis(self, time_values: List[float] = None,
                            exploration_constant: float = 1.41,
                            baseline_agent: AgentBase = None,
                            n_games_per_config: int = 100):
        """
        Analyse l'impact du budget temporel
        """
        if time_values is None:
            time_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        
        if baseline_agent is None:
            baseline_agent = MinimaxAgent(depth=2, time_budget=1.0, label="Minimax_d2_tb1s")
        
        results = []
        
        for t in time_values:
            print(f"\nTest avec time_limit = {t}s")
            mcts = MCTSAgent(time_limit=t, exploration_constant=exploration_constant,
                            label=f"MCTS_t{t}s")
            
            exp = StatisticalExperiment(self.output_dir)
            df = exp.run_experiment(mcts, baseline_agent, 
                                  n_games=n_games_per_config)
            stats = exp.calculate_statistics(df)
            
            results.append({
                'time_limit': t,
                'win_rate': stats['win_rates'][mcts.name]['rate'],
                'ci_low': stats['win_rates'][mcts.name]['ci_low'],
                'ci_high': stats['win_rates'][mcts.name]['ci_high'],
                'avg_moves': stats['game_length']['mean_moves']
            })
        
        # Créer graphique
        self._plot_time_analysis(results)
        
        return pd.DataFrame(results)
    
    def _plot_exploration_analysis(self, results: List[Dict]):
        """Graphique pour l'analyse de C"""
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['C'], df['win_rate'], 
                    yerr=[df['win_rate'] - df['ci_low'], 
                          df['ci_high'] - df['win_rate']],
                    marker='o', markersize=8, capsize=5, capthick=2,
                    linewidth=2, color='#1976D2')
        
        plt.xlabel('Constante d\'exploration C', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.title('Impact de la constante d\'exploration sur les performances', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Marquer la valeur optimale
        best_idx = df['win_rate'].idxmax()
        best_c = df.loc[best_idx, 'C']
        best_wr = df.loc[best_idx, 'win_rate']
        plt.scatter([best_c], [best_wr], color='red', s=100, zorder=5)
        plt.annotate(f'Optimal: C={best_c:.2f}\nWR={best_wr:.3f}',
                    xy=(best_c, best_wr), xytext=(10, 10),
                    textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'exploration_constant_analysis.png'), dpi=150)
        plt.show()
    
    def _plot_time_analysis(self, results: List[Dict]):
        """Graphique pour l'analyse du temps"""
        df = pd.DataFrame(results)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Win rate vs temps
        ax1.errorbar(df['time_limit'], df['win_rate'],
                    yerr=[df['win_rate'] - df['ci_low'],
                          df['ci_high'] - df['win_rate']],
                    marker='o', markersize=8, capsize=5, capthick=2,
                    linewidth=2, color='#2E7D32')
        ax1.set_xlabel('Budget temporel (s)', fontsize=12)
        ax1.set_ylabel('Win Rate', fontsize=12)
        ax1.set_title('Performance vs Budget temporel', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax1.set_xscale('log')
        
        # Longueur moyenne des parties vs temps
        ax2.plot(df['time_limit'], df['avg_moves'], marker='s', 
                markersize=8, linewidth=2, color='#E65100')
        ax2.set_xlabel('Budget temporel (s)', fontsize=12)
        ax2.set_ylabel('Nombre moyen de coups', fontsize=12)
        ax2.set_title('Longueur des parties vs Budget temporel', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_budget_analysis.png'), dpi=150)
        plt.show()

# ============= SCRIPT PRINCIPAL =============

def main():
    """
    Script principal pour lancer l'analyse complète
    """
    print("\n" + "="*80)
    print("ANALYSE STATISTIQUE COMPLÈTE POUR MCTS PENTAGO")
    print("="*80 + "\n")
    
    # 1. ANALYSE PRINCIPALE : MCTS vs Adversaires variés
    print("\n>>> PHASE 1: Analyse principale MCTS vs différents adversaires\n")
    
    # Configuration MCTS principal à tester
    mcts_main = MinimaxAgent(depth=2, time_budget=1.0,label='MINIMAX_d2_tb1s')
    
    # Adversaires à tester
    opponents = [MinimaxAgent(depth=2, time_budget=1.0,label='MINIMAX_d2_tb1s_bis')]
     
    exp = StatisticalExperiment("mcts_analysis_results")
    
    for opponent in opponents:
        print(f"\n--- Test: {mcts_main.name} vs {opponent.name} ---\n")
        
        
        # Lancer l'expérience
        df = exp.run_experiment(mcts_main, opponent, n_games=200, base_seed=42)
        
        # Calculer les statistiques
        stats = exp.calculate_statistics(df)
        
        # Générer visualisations
        exp.generate_visualizations(df, stats)
        
        # Générer rapport
        exp.generate_report(df, stats)
    
    # 2. ANALYSE DES HYPERPARAMÈTRES
    print("\n>>> PHASE 2: Analyse des hyperparamètres MCTS\n")
    
    mcts_analysis = MCTSSpecificAnalysis("mcts_hyperparam_analysis")
    
    # Tester différentes valeurs de C
    print("\n--- Analyse de la constante d'exploration C ---\n")
    c_results = mcts_analysis.exploration_constant_analysis(
        time_limit=1.0,
        c_values=[0.5, 0.7, 1.0, 1.41, 2.0, 2.5],
        n_games_per_config=100
    )
    print("\nRésultats pour C:")
    print(c_results)
    
    # Tester différents budgets temporels
    print("\n--- Analyse du budget temporel ---\n")
    time_results = mcts_analysis.time_budget_analysis(
        time_values=[0.5, 1.0, 2.0, 5.0],
        exploration_constant=1.41,
        n_games_per_config=100
    )
    print("\nRésultats pour time_limit:")
    print(time_results)
    
    print("\n" + "="*80)
    print("ANALYSE TERMINÉE - Vérifiez les dossiers de résultats")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()