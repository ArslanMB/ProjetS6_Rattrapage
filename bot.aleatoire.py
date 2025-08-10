import random
from typing import Tuple
from pentago_game import Pentago  # assure-toi que Pentago est dans pentago_base.py

Coord = Tuple[int, int]
Rotation = Tuple[int, str]
Move = Tuple[Coord, Rotation]

class RandomBot:
    def __init__(self, symbol: str):
        self.symbol = symbol  # 'X' ou 'O'

    def select_move(self, game: Pentago) -> Move:
        moves = game.legal_moves()
        return random.choice(moves) if moves else None

    def play_one(self, game: Pentago) -> bool:
        if game.current != self.symbol or game.is_terminal():
            return False
        mv = self.select_move(game)
        if not mv: 
            return False
        placement, rotation = mv
        return game.play(placement, rotation)

# --- Démo rapide bot vs bot (console) ---
if __name__ == "__main__":
    g = Pentago()
    xbot = RandomBot("X")
    obot = RandomBot("O")

    while not g.is_terminal():
        (xbot if g.current == "X" else obot).play_one(g)

    print(g.render())
    print("Résultat:", g.winner())

