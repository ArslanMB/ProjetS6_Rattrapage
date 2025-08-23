import numpy as np
from collections import defaultdict
from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2, QUADRANT_SIZE, BOARD_ROWS, BOARD_COLS, ROTATION_SPEED
from alphabeta_ia.alpha_beta import timed_find_best_move_minimax  # ton bot retourne (move, dt)



# --- Depth adaptative locale (sans dépendre du bot) ---
def pick_adaptive_depth(board, base=2, hard_cap=5):
    empties = int(np.count_nonzero(board == 0))
    d = base
    if empties <= 24: d = max(d, 3)
    if empties <= 16: d = max(d, 4)
    if empties <= 10: d = max(d, 5)
    return min(d, hard_cap)

def _resolve_depth(mode, board):
    if isinstance(mode, str) and mode.upper() == "A":
        return pick_adaptive_depth(board, base=2, hard_cap=5)
    return int(mode)

# helpers
def _apply_move_headless(board, move, player):
    r, c, q, d = move
    b1 = PentagoGame.get_board_after_placement(board, r, c, player)
    if PentagoGame.check_win_on_board(b1) == player:
        return b1
    return PentagoGame.get_board_after_rotation(b1, q, d)

class _State:
    __slots__ = ("board", "current_player")
    def __init__(self, board, current_player):
        self.board = board
        self.current_player = current_player

def play_one_game(ai1_depth=2, ai2_depth=2, starter=PLAYER_1):
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
    cur = starter
    map_player_to_ai = {starter: "AI1", -starter: "AI2"}
    times = {"AI1": [], "AI2": []}
    plies = 0
    winner = 0

    while True:
        plies += 1
        which = map_player_to_ai[cur]
        depth_mode = ai1_depth if which == "AI1" else ai2_depth
        depth_used = _resolve_depth(depth_mode, board)

        state = _State(board, cur)
        mv, dt = timed_find_best_move_minimax(state, depth=depth_used)
        if mv is None:
            winner = 2
            break
        times[which].append(dt)

        board = _apply_move_headless(board, mv, cur)
        w = PentagoGame.check_win_on_board(board)
        if w != 0:
            winner = w if w in (PLAYER_1, PLAYER_2) else 2
            break

        cur = -cur

    if winner == PLAYER_1:
        winner_ai = map_player_to_ai[PLAYER_1]
    elif winner == PLAYER_2:
        winner_ai = map_player_to_ai[PLAYER_2]
    else:
        winner_ai = "DRAW"

    return {
        "winner_ai": winner_ai,
        "plies": plies,
        "t_ai1": times["AI1"],
        "t_ai2": times["AI2"],
    }

def run_selfplay(n_games=20, ai1_depth=2, ai2_depth=2):
    stats = defaultdict(int)
    all_plies = []
    t_ai1_all, t_ai2_all = [], []
    long_ai1 = 0.0
    long_ai2 = 0.0

    for g in range(n_games):
        starter = PLAYER_1 if (g % 2 == 0) else PLAYER_2
        res = play_one_game(ai1_depth, ai2_depth, starter=starter)
        winner_ai = res["winner_ai"]
        stats[winner_ai] += 1
        all_plies.append(res["plies"])

        t_ai1_all.extend(res["t_ai1"])
        t_ai2_all.extend(res["t_ai2"])
        if res["t_ai1"]:
            long_ai1 = max(long_ai1, max(res["t_ai1"]))
        if res["t_ai2"]:
            long_ai2 = max(long_ai2, max(res["t_ai2"]))

        print(f"[G{g+1}/{n_games}] starter={'AI1' if starter==PLAYER_1 else 'AI2'} | winner={winner_ai} | plies={res['plies']}")

    def _avg(lst): return (sum(lst)/len(lst)) if lst else 0.0

    ai1_w = stats["AI1"]
    ai2_w = stats["AI2"]
    draws = stats["DRAW"]
    print("\n===== RÉSUMÉ =====")
    label1 = f"A (adaptative)" if (isinstance(ai1_depth,str) and ai1_depth.upper()=="A") else str(ai1_depth)
    label2 = f"A (adaptative)" if (isinstance(ai2_depth,str) and ai2_depth.upper()=="A") else str(ai2_depth)
    print(f"Parties: {n_games}")
    print(f"AI1 (depth={label1})  victoires: {ai1_w}")
    print(f"AI2 (depth={label2})  victoires: {ai2_w}")
    print(f"Nulles: {draws}")
    print(f"Winrate AI1: { (ai1_w / n_games * 100):.1f}% | Winrate AI2: { (ai2_w / n_games * 100):.1f}%")
    print(f"Plies moyens: {_avg(all_plies):.1f}")
    print(f"Temps moyen/coup AI1: {_avg(t_ai1_all):.3f}s | plus long: {long_ai1:.3f}s")
    print(f"Temps moyen/coup AI2: {_avg(t_ai2_all):.3f}s | plus long: {long_ai2:.3f}s")

    return {
        "games": n_games,
        "ai1_wins": ai1_w,
        "ai2_wins": ai2_w,
        "draws": draws,
        "avg_plies": _avg(all_plies),
        "ai1_avg_t": _avg(t_ai1_all),
        "ai2_avg_t": _avg(t_ai2_all),
        "ai1_max_t": long_ai1,
        "ai2_max_t": long_ai2,
    }

if __name__ == "__main__":
    run_selfplay(n_games=2, ai1_depth="A", ai2_depth=2)
