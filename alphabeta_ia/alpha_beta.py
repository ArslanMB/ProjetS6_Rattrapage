import numpy as np
import time
from collections import OrderedDict, defaultdict
from core.pentago_logic import PentagoGame
from core.constants import PLAYER_1, PLAYER_2, QUADRANT_SIZE, BOARD_ROWS, BOARD_COLS, ROTATION_SPEED
from alphabeta_ia.opening_book import probe_opening_move


# CACHE et CLES
APPLY_CACHE_MAX = 300_000
WIN_CACHE_MAX   = 400_000
MOVE_CACHE_MAX  = 150_000
MAX_EVAL_CACHE  = 150_000

_APPLY_CACHE = OrderedDict()
_WIN_CACHE   = OrderedDict()
_MOVE_CACHE  = OrderedDict()
_EVAL_CACHE  = OrderedDict()

def bkey(board): 
    return board.astype(np.int8, copy=False).tobytes()

def check_win_cached(board):
    k = bkey(board)
    try:
        v = _WIN_CACHE.pop(k); _WIN_CACHE[k] = v
        return v
    except KeyError:
        v = PentagoGame.check_win_on_board(board)
        _WIN_CACHE[k] = v
        if len(_WIN_CACHE) > WIN_CACHE_MAX:
            _WIN_CACHE.popitem(last=False)
        return v

def _apply_key(board, move, player):
    return (bkey(board), move, int(player))

def apply_move_cached(board, move, player):
    k = _apply_key(board, move, player)
    try:
        b = _APPLY_CACHE.pop(k); _APPLY_CACHE[k] = b
        return b
    except KeyError:
        r, c, q, d = move
        b1 = PentagoGame.get_board_after_placement(board, r, c, player)
        if check_win_cached(b1) == player:
            res = b1
        else:
            res = PentagoGame.get_board_after_rotation(b1, q, d)
        _APPLY_CACHE[k] = res
        if len(_APPLY_CACHE) > APPLY_CACHE_MAX:
            _APPLY_CACHE.popitem(last=False)
        return res

def _board_eval_key(board, player):
    return (board.dtype.str, board.shape, bkey(board), int(player))

def _eval_cache_get(k):
    try:
        v = _EVAL_CACHE.pop(k); _EVAL_CACHE[k] = v
        return v
    except KeyError:
        return None

def _eval_cache_put(k, v):
    _EVAL_CACHE[k] = v
    if len(_EVAL_CACHE) > MAX_EVAL_CACHE:
        _EVAL_CACHE.popitem(last=False)

# ================== TRANSPO / HEURISTIQUES ==================
TT = {}  # key -> (depth, value, flag, best_move)
def _tt_key(board, maximizing, root_player, depth):
    b8 = board.astype(np.int8, copy=False)
    return (b8.tobytes(), 1 if maximizing else 0, int(root_player))  # sans depth

MAX_PLY = 64
KILLER1 = [None]*MAX_PLY
KILLER2 = [None]*MAX_PLY
HISTORY = defaultdict(int)

# Time control 
class _TimeUp(Exception): pass
TIME_DEADLINE = [None]
NODES = [0]
CHECK_INTERVAL = 2048
def _time_check():
    NODES[0] += 1
    if (NODES[0] & (CHECK_INTERVAL-1)) == 0 and TIME_DEADLINE[0] is not None:
        if time.perf_counter() > TIME_DEADLINE[0]:
            raise _TimeUp()

# COUPS LÉGAUX OPTI (Cache)
def get_legal_moves(board):
    k = bkey(board)
    try:
        mv = _MOVE_CACHE.pop(k); _MOVE_CACHE[k] = mv
        return mv
    except KeyError:
        pass

    empties = np.argwhere(board == 0)
    stones  = np.argwhere(board != 0)

    use_local = len(stones) > 4
    local = set()
    if use_local and len(stones) > 0:
        R, C = board.shape
        for (sr, sc) in stones:
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    rr, cc = int(sr+dr), int(sc+dc)
                    if 0 <= rr < R and 0 <= cc < C and board[rr, cc] == 0:
                        local.add((rr, cc))

    moves = []
    QS = QUADRANT_SIZE
    for (r, c) in empties:
        r = int(r); c = int(c)
        if use_local and (r, c) not in local:
            continue
        for q in range(4):
            qr = (q // 2) * QS
            qc = (q %  2) * QS
            plays_inside = (qr <= r < qr+QS) and (qc <= c < qc+QS)
            quad = board[qr:qr+QS, qc:qc+QS]
            quad_empty = (np.count_nonzero(quad) == 0)
            if quad_empty and not plays_inside:
                moves.append((r, c, q, 1))      # une seule direction
            else:
                moves.append((r, c, q, 1))
                moves.append((r, c, q, -1))

    _MOVE_CACHE[k] = moves
    if len(_MOVE_CACHE) > MOVE_CACHE_MAX:
        _MOVE_CACHE.popitem(last=False)
    return moves

#  ORDO / WINS 
def _winning_moves_fast(board, player, first_only=False):
    wins = []
    empties = np.argwhere(board == 0)
    for (r, c) in empties:
        r = int(r); c = int(c)
        for q in range(4):
            for d in (1, -1):
                mv = (r, c, q, d)
                nb = apply_move_cached(board, mv, player)
                if check_win_cached(nb) == player:
                    wins.append(mv)
                    if first_only:
                        return wins
    return wins

def _order_moves(board, moves, player, ply):
    center_r, center_c = 2.5, 2.5
    opp = -player
    opp_wins_before = _winning_moves_fast(board, opp, first_only=True)

    ordered = []
    for (r, c, q, d) in moves:
        mv = (r, c, q, d)
        nb = apply_move_cached(board, mv, player)

        win_now = 1 if check_win_cached(nb) == player else 0
        block_now = 0
        if opp_wins_before:
            if not _winning_moves_fast(nb, opp, first_only=True):
                block_now = 1

        dist_center = abs(r - center_r) + abs(c - center_c)
        center_score = -dist_center

        neigh = 0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr, cc = r+dr, c+dc
                if 0 <= rr < BOARD_ROWS and 0 <= cc < BOARD_COLS and board[rr, cc] == player:
                    neigh += 1

        killer = 1 if (mv == KILLER1[ply] or mv == KILLER2[ply]) else 0
        hist = HISTORY[(player, mv)]

        ordered.append((win_now, block_now, killer, hist, neigh, center_score, mv))

    ordered.sort(reverse=True)
    return [t[-1] for t in ordered]

# ================== ÉVALUATION (fenêtres pré-calculées) ==================
PRECOMP_WINDOWS = []
for r in range(BOARD_ROWS):
    for c in range(BOARD_COLS - 4):
        PRECOMP_WINDOWS.append([(r, c+i) for i in range(5)])
for c in range(BOARD_COLS):
    for r in range(BOARD_ROWS - 4):
        PRECOMP_WINDOWS.append([(r+i, c) for i in range(5)])
for r in range(BOARD_ROWS - 4):
    for c in range(BOARD_COLS - 4):
        PRECOMP_WINDOWS.append([(r+i, c+i) for i in range(5)])
for r in range(BOARD_ROWS - 4):
    for c in range(BOARD_COLS - 4):
        PRECOMP_WINDOWS.append([(r+i, c+4-i) for i in range(5)])

W = {4:2500, 3:120, 2:25, 1:4}

def _iter_windows_with_coords(board):
    for coords in PRECOMP_WINDOWS:
        vals = np.fromiter((board[r, c] for (r, c) in coords), dtype=board.dtype, count=5)
        yield vals, coords

def _window_score(vals, player):
    cnt_p = np.count_nonzero(vals == player)
    cnt_o = np.count_nonzero(vals == -player)
    if cnt_p > 0 and cnt_o > 0:
        return 0
    return W.get(cnt_p, 0) - W.get(cnt_o, 0)

CENTRAL_4 = [(2,2),(2,3),(3,2),(3,3)]
QUAD_CENTERS = [(1,1),(1,4),(4,1),(4,4)]
def _center_bonus(board, player):
    b = 0
    for r, c in CENTRAL_4:
        v = board[r, c]
        if v == player: b += 6
        elif v == -player: b -= 6
    for r, c in QUAD_CENTERS:
        v = board[r, c]
        if v == player: b += 3
        elif v == -player: b -= 3
    return b

def _winning_empty_cells(board, player):
    winning = set()
    for vals, coords in _iter_windows_with_coords(board):
        cnt_p = np.count_nonzero(vals == player)
        cnt_o = np.count_nonzero(vals == -player)
        if cnt_o == 0 and cnt_p == 4:
            for (v, xy) in zip(vals, coords):
                if v == 0:
                    winning.add(xy); break
    return winning

def evaluate(board, player):
    k = _board_eval_key(board, player)
    c = _eval_cache_get(k)
    if c is not None:
        return c

    winner = check_win_cached(board)
    if winner == player:
        _eval_cache_put(k, float('inf')); return float('inf')
    elif winner == -player:
        _eval_cache_put(k, float('-inf')); return float('-inf')
    elif winner == 2:
        _eval_cache_put(k, 0.0); return 0.0

    score = 0.0
    for vals, _coords in _iter_windows_with_coords(board):
        score += _window_score(vals, player)

    my_wins = _winning_empty_cells(board, player)
    op_wins = _winning_empty_cells(board, -player)
    FORK_BONUS = 1800
    if len(my_wins) >= 2: score += FORK_BONUS * (len(my_wins) - 1)
    if len(op_wins) >= 2: score -= FORK_BONUS * (len(op_wins) - 1)

    score += _center_bonus(board, player)

    res = float(score)
    _eval_cache_put(k, res)
    return res

# Adaptative depth
A = 0  
MODE_IA = "A"
def pick_adaptive_depth(board, base=2, hard_cap=5):
    n_moves = len(get_legal_moves(board))
    empties = int(np.count_nonzero(board == 0))
    d = base
    if empties <= 24 or n_moves <= 100: d = max(d, 3)
    if empties <= 16 or n_moves <= 60:  d = max(d, 4)
    if empties <= 10 or n_moves <= 25:  d = max(d, 5)
    return min(d, hard_cap)

def get_adaptive_depth():
    return A

# ================== ALPHABÊTA (PVS + LMR + killers/history + TT) ==================
FUT_DEPTH  = 2
FUT_MARGIN = 300.0

def alphabeta(board, depth, alpha, beta, maximizing, root_player, ply=0):
    _time_check()
    key = _tt_key(board, maximizing, root_player, depth)
    winner = check_win_cached(board)
    if depth == 0 or winner != 0 or np.all(board != 0):
        val = evaluate(board, root_player)
        TT[key] = (depth, val, 'EXACT', None)
        return val, None

    entry = TT.get(key)
    if entry is not None:
        d_stored, v_stored, flag, mv_stored = entry
        if d_stored >= depth:
            if flag == 'EXACT':
                return v_stored, mv_stored
            elif flag == 'LOWER':
                if v_stored > alpha: alpha = v_stored
            elif flag == 'UPPER':
                if v_stored < beta:  beta = v_stored
            if alpha >= beta:
                return v_stored, mv_stored

    alpha0, beta0 = alpha, beta
    player_to_move = root_player if maximizing else -root_player
    moves = get_legal_moves(board)
    if not moves:
        val = evaluate(board, root_player)
        TT[key] = (depth, val, 'EXACT', None)
        return val, None

    wins = _winning_moves_fast(board, player_to_move, first_only=True)
    if wins:
        best = wins[0]
        val  = evaluate(apply_move_cached(board, best, player_to_move), root_player)
        TT[key] = (depth, val, 'EXACT', best)
        return val, best

    moves = _order_moves(board, moves, player_to_move, ply)
    pv_move = entry[3] if entry is not None else None
    if pv_move is not None and pv_move in moves:
        i = moves.index(pv_move)
        if i != 0: moves[0], moves[i] = moves[i], moves[0]

    static_eval = None
    if depth <= FUT_DEPTH:
        static_eval = evaluate(board, root_player)

    best_move = None

    # PVS
    if maximizing:
        value = float('-inf')
        for i, mv in enumerate(moves):
            nb = apply_move_cached(board, mv, player_to_move)

            if static_eval is not None and i >= 6:
                if static_eval + FUT_MARGIN <= alpha:
                    # prune tardif non-forçant
                    pass  # continue sans descendre
                # (on descend quand même si ça peut dépasser alpha)
            reduced = (depth >= 2 and i >= 8)
            child_depth = depth - 1

            if i == 0:
                score, _ = alphabeta(nb, child_depth, alpha, beta, False, root_player, ply+1)
            else:
                if reduced:
                    score, _ = alphabeta(nb, child_depth - 1, alpha, alpha+1, False, root_player, ply+1)
                    if score > alpha:
                        score, _ = alphabeta(nb, child_depth, alpha, alpha+1, False, root_player, ply+1)
                        if score > alpha and score < beta:
                            score, _ = alphabeta(nb, child_depth, alpha, beta, False, root_player, ply+1)
                else:
                    score, _ = alphabeta(nb, child_depth, alpha, alpha+1, False, root_player, ply+1)
                    if score > alpha and score < beta:
                        score, _ = alphabeta(nb, child_depth, alpha, beta, False, root_player, ply+1)

            if score > value:
                value, best_move = score, mv
            if value > alpha: alpha = value
            if alpha >= beta:
                if mv != KILLER1[ply]:
                    KILLER2[ply] = KILLER1[ply]
                    KILLER1[ply] = mv
                HISTORY[(player_to_move, mv)] += depth*depth
                break

        flag = 'EXACT'
        if value <= alpha0: flag = 'UPPER'
        elif value >= beta: flag = 'LOWER'
        TT[key] = (depth, value, flag, best_move)
        return value, best_move

    else:
        value = float('inf')
        for i, mv in enumerate(moves):
            nb = apply_move_cached(board, mv, player_to_move)
            if static_eval is not None and i >= 6:
                if static_eval - FUT_MARGIN >= beta:
                    pass  # prune tardif
            reduced = (depth >= 2 and i >= 8)
            child_depth = depth - 1

            if i == 0:
                score, _ = alphabeta(nb, child_depth, alpha, beta, True, root_player, ply+1)
            else:
                if reduced:
                    score, _ = alphabeta(nb, child_depth - 1, alpha, beta, True, root_player, ply+1)
                    if score < beta:
                        score, _ = alphabeta(nb, child_depth, alpha, beta, True, root_player, ply+1)
                else:
                    score, _ = alphabeta(nb, child_depth, alpha, beta, True, root_player, ply+1)

            if score < value:
                value, best_move = score, mv
            if value < beta: beta = value
            if alpha >= beta:
                if mv != KILLER1[ply]:
                    KILLER2[ply] = KILLER1[ply]
                    KILLER1[ply] = mv
                HISTORY[(player_to_move, mv)] += depth*depth
                break

        flag = 'EXACT'
        if value <= alpha:  flag = 'UPPER'
        elif value >= beta0: flag = 'LOWER'
        TT[key] = (depth, value, flag, best_move)
        return value, best_move

# MINIMAX SIMPLE (non utilisé par l'UI, conservé)
def minimax(board, depth, maximizing, player):
    winner = check_win_cached(board)
    if depth == 0 or winner != 0 or np.all(board != 0):
        return evaluate(board, player), None
    moves = get_legal_moves(board)
    if not moves:
        return evaluate(board, player), None
    if maximizing:
        best_score, best_move = float('-inf'), None
        for mv in moves:
            nb = apply_move_cached(board, mv, player)
            sc, _ = minimax(nb, depth-1, False, player)
            if sc > best_score: best_score, best_move = sc, mv
        return best_score, best_move
    else:
        best_score, best_move = float('inf'), None
        for mv in moves:
            nb = apply_move_cached(board, mv, -player)
            sc, _ = minimax(nb, depth-1, True, player)
            if sc < best_score: best_score, best_move = sc, mv
        return best_score, best_move

# ================== ITERATIVE DEEPENING + ASPIRATION ==================
def find_best_move_iterative(game_instance, max_depth=4, time_budget=2.5):
    board = np.copy(game_instance.board)
    root_player = game_instance.current_player
    TIME_DEADLINE[0] = time.perf_counter() + (time_budget if time_budget else 1e9)
    NODES[0] = 0
    best_move = None
    last_score = 0.0
    ASP = 500.0
    for d in range(1, max_depth+1):
        alpha = last_score - ASP if d > 1 else float('-inf')
        beta  = last_score + ASP if d > 1 else float('inf')
        while True:
            try:
                score, mv = alphabeta(board, d, alpha, beta, True, root_player, ply=0)
            except _TimeUp:
                return best_move if best_move is not None else mv
            if mv is None and best_move is not None:
                mv = best_move
            if score <= alpha:
                ASP *= 2.0; alpha = score - ASP; beta = score + ASP; continue
            if score >= beta:
                ASP *= 2.0; alpha = score - ASP; beta = score + ASP; continue
            best_move = mv; last_score = score
            break
    return best_move

def find_best_move_minimax(game_instance, depth=2, time_budget=2.5):
    mv_book = probe_opening_move(game_instance.board, game_instance.current_player)
    if mv_book is not None:
        return mv_book  
    global A
    if MODE_IA == "A":
        A = pick_adaptive_depth(game_instance.board, base=depth, hard_cap=max(5, depth))
        return find_best_move_iterative(game_instance, max_depth=A, time_budget=time_budget)
    else:
        fixed_depth = int(MODE_IA)
        A = fixed_depth
        return find_best_move_iterative(game_instance, max_depth=fixed_depth, time_budget=time_budget)


# Temps
SEARCH_TIMES = []
_LONGEST = [0.0, None]

def reset_timing():
    SEARCH_TIMES.clear(); _LONGEST[0]=0.0; _LONGEST[1]=None

def record_search_time(sec: float):
    SEARCH_TIMES.append(float(sec))
    i = len(SEARCH_TIMES)
    if sec > _LONGEST[0]:
        _LONGEST[0], _LONGEST[1] = float(sec), i

def get_timing_stats():
    n = len(SEARCH_TIMES)
    avg = (sum(SEARCH_TIMES) / n) if n else 0.0
    last = SEARCH_TIMES[-1] if n else 0.0
    return {"moves": n, "avg": avg, "last": last, "longest": _LONGEST[0], "longest_move_index": _LONGEST[1]}

def print_timing_summary(prefix="[IA]"):
    s = get_timing_stats()
    print(f"{prefix} Coups:{s['moves']} | Moy:{s['avg']:.3f}s | Dernier:{s['last']:.3f}s | Max:{s['longest']:.3f}s (#{s['longest_move_index']})", flush=True)

def timed_find_best_move_minimax(game_instance, depth=2, time_budget=2.5):
    t0 = time.perf_counter()
    mv = find_best_move_minimax(game_instance, depth=depth, time_budget=time_budget)
    dt = time.perf_counter() - t0
    record_search_time(dt)
    return mv, dt
