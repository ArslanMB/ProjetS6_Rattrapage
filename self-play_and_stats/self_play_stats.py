import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2
from alphabeta_ia.alpha_beta import (
    timed_find_best_move_minimax,  
    apply_move_cached,            
    check_win_cached,             
    get_legal_moves              
)
from mtcs_ia.optimized_mcts import OptimizedMCTS


# Agents unifiés

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
        mv, dt, _nodes = timed_find_best_move_minimax(game, depth=self.depth, time_budget=self.time_budget, BOOKING=self.BOOKING)
        return mv, dt

class MCTSAgent(AgentBase):
    def __init__(self, time_limit: float = 2.5, exploration_constant: float = 1.41, label: Optional[str] = None):
        self.bot = OptimizedMCTS(time_limit=float(time_limit), exploration_constant=float(exploration_constant))
        self.name = label or f"MCTS_t{time_limit:g}s_C{exploration_constant:g}"
    def choose_move(self, game: PentagoGame):
        t0 = time.perf_counter()
        mv = self.bot.find_best_move(game)
        dt = time.perf_counter() - t0
        return mv, dt


# ---- Boucle d'une partie pour 1 paire ----

@dataclass
class GameResult:
    winner: int                 # 1, -1, 0 (nul)
    plies: int
    p1_times: List[float]
    p2_times: List[float]

def _apply_move(game: PentagoGame, mv: Tuple[int,int,int,int]):
    player = game.current_player
    game.board = apply_move_cached(game.board, mv, player)
    game.current_player = -player

def play_one_game(agent1: AgentBase, agent2: AgentBase, starter: int) -> GameResult:
    game = PentagoGame()
    game.current_player = starter
    agent1.on_game_start()
    agent2.on_game_start()

    p1_times, p2_times = [], []

    for ply in range(200):
        print(ply)
        # terminal avant coup
        w = check_win_cached(game.board)
        if w != 0:
            return GameResult(winner=0 if w == 2 else w, plies=ply, p1_times=p1_times, p2_times=p2_times)
        if not np.any(game.board == 0):
            return GameResult(winner=0, plies=ply, p1_times=p1_times, p2_times=p2_times)

        agent = agent1 if game.current_player == PLAYER_1 else agent2
        mv, dt = agent.choose_move(game)

        if mv is None:
            legal = get_legal_moves(game.board)
            if not legal:
                return GameResult(winner=0, plies=ply, p1_times=p1_times, p2_times=p2_times)
            mv = legal[0]
        
        _apply_move(game, mv)

        if agent is agent1: p1_times.append(float(dt))
        else:               p2_times.append(float(dt))

        w = check_win_cached(game.board)
        if w != 0:
            return GameResult(winner=0 if w == 2 else w, plies=ply+1, p1_times=p1_times, p2_times=p2_times)

    return GameResult(winner=0, plies=200, p1_times=p1_times, p2_times=p2_times)

# ---- Statistiques de temps ----

def summarize_times(ts: List[float]) -> Dict[str, float]:
    if not ts: return {"avg": 0.0, "med": 0.0, "p95": 0.0, "max": 0.0}
    s = sorted(ts); n = len(s)
    med = s[n//2] if n % 2 else 0.5*(s[n//2 - 1] + s[n//2])
    p95 = s[min(n-1, max(0, int(math.ceil(0.95*n))-1))]
    return {"avg": sum(s)/n, "med": med, "p95": p95, "max": s[-1]}

def run_series(agent1: AgentBase, agent2: AgentBase, games: int = 50, seed: int = 42, record_file = os.path.join(os.path.dirname(__file__), "records.txt")):
    random.seed(seed); np.random.seed(seed)
    wins1 = wins2 = draws = 0
    all_p1, all_p2 = [], []

    for i in range(games):
        starter = PLAYER_1 if (i % 2 == 0) else PLAYER_2
        res = play_one_game(agent1, agent2, starter)
        if res.winner == PLAYER_1: wins1 += 1
        elif res.winner == PLAYER_2: wins2 += 1
        else: draws += 1
        all_p1 += res.p1_times
        all_p2 += res.p2_times

    p1s, p2s = summarize_times(all_p1), summarize_times(all_p2)

    result_text = (
        f"\n=== Série {agent1.name}  vs  {agent2.name}  (N={games}) ===\n"
        f"{agent1.name:>24}: {wins1:3d} victoires | t(avg/med/p95/max)={p1s['avg']:.3f}/{p1s['med']:.3f}/{p1s['p95']:.3f}/{p1s['max']:.3f}s\n"
        f"{agent2.name:>24}: {wins2:3d} victoires | t(avg/med/p95/max)={p2s['avg']:.3f}/{p2s['med']:.3f}/{p2s['p95']:.3f}/{p2s['max']:.3f}s\n"
        f"{'Nuls':>24}: {draws:3d}  ({(draws/games):.1%})\n"
    )

    print(result_text)

    with open(record_file, "a", encoding="utf-8") as f:
        f.write(result_text)

    return {
        "wins": {agent1.name: wins1, agent2.name: wins2, "draws": draws},
        "p1_times": p1s,
        "p2_times": p2s
    }


if __name__ == "__main__":
    
    a1 = MinimaxAgent(depth=1, time_budget=10, label="MM_d1_10s_BOOK", BOOKING=True)
    a2 = MinimaxAgent(depth=1,   time_budget=10, label="MM_d1_10s", BOOKING=False)
    run_series(a1, a2, games=1, seed=2003)

   
    a3 = MinimaxAgent(depth=1, time_budget=2.5, label="MM_A_2.5s")
    a4 = MCTSAgent(time_limit=2.5, exploration_constant=1.41, label="MCTS_2.5s")
    #run_series(a3, a4, games=150, seed=2003)

    # Ex 3) MCTS 1s vs MCTS 2.5s
    # b1 = MCTSAgent(time_limit=1.0, label="MCTS_1s")
    # b2 = MCTSAgent(time_limit=2.5, label="MCTS_2.5s")
    # run_series(b1, b2, games=30)
