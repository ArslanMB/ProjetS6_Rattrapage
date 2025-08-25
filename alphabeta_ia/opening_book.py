from typing import Optional, Tuple, List
import numpy as np

Move = Tuple[int, int, int, int]


OPENING_FIRST_MOVES: List[Move] = [
    (2, 2, 0, +1),  
    (3, 3, 3, -1),
    (2, 3, 1, +1),
    (3, 2, 2, -1),
]

SECOND_MOVE_BOOK: dict = {
    # central plateau ==> centre quadrant
    ("OPP_AT", (1, 1)): (3, 2, 2, +1),
    ("OPP_AT", (1, 4)): (2, 3, 1, -1),
    ("OPP_AT", (4, 1)): (3, 2, 2, -1),
    ("OPP_AT", (4, 4)): (2, 3, 1, +1),

    # Contestation de centre
    ("OPP_AT", (3, 3)): (2, 3, 1, +1),

    # Bords centraux 
    ("OPP_AT", (2, 0)): (2, 3, 1, +1),
    ("OPP_AT", (0, 2)): (3, 2, 2, -1),
    ("OPP_AT", (3, 5)): (3, 2, 2, +1),
    ("OPP_AT", (5, 3)): (2, 3, 1, -1),

    # Si coin centre agressif
    ("OPP_AT", (0, 0)): (2, 3, 1, +1),
    ("OPP_AT", (0, 5)): (3, 2, 2, -1),
    ("OPP_AT", (5, 0)): (2, 3, 1, -1),
    ("OPP_AT", (5, 5)): (3, 2, 2, +1),
}

def _board_is_empty(board: np.ndarray) -> bool:
    return np.count_nonzero(board) == 0

def _find_single_opponent_stone(board: np.ndarray, me: int) -> Optional[Tuple[int,int]]:
    """Renvoie (r,c) d’une unique pierre adverse si et seulement si le plateau contient
    exactement 1 pierre adverse et 1 pierre à moi (cas typique après deux plies).
    Retourne None sinon."""
    # Après ton 1er coup, l’adversaire joue: on attend 1 pierre à toi et 1 à lui.
    if np.count_nonzero(board == me) != 1:
        return None
    if np.count_nonzero(board == -me) != 1:
        return None
    pos = np.argwhere(board == -me)
    if len(pos) != 1:
        return None
    r, c = int(pos[0][0]), int(pos[0][1])
    return (r, c)

def probe_opening_move(board: np.ndarray, current_player: int) -> Optional[Move]:

    if _board_is_empty(board) and current_player == 1:
        return OPENING_FIRST_MOVES[0]


    if current_player == 1 and np.count_nonzero(board) == 2:
        opp = _find_single_opponent_stone(board, me=1)
        if opp is not None:
            key = ("OPP_AT", opp)
            mv = SECOND_MOVE_BOOK.get(key)
            if mv is not None:
                return mv


    return None
