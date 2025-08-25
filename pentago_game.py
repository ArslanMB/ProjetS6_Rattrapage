from __future__ import annotations
import tkinter as tk
import random
from typing import Optional, Tuple


Coord = Tuple[int, int]
Rotation = Tuple[int, str]  # (quadrant in 0..3, direction in {"CW","CCW"})
Move = Tuple[Coord, Rotation]


class Pentago:
    def __init__(self) -> None:
        self.board = [["." for _ in range(6)] for _ in range(6)]
        self.current = "X"
        self._last_move: Optional[Move] = None

    def clone(self) -> "Pentago":
        g = Pentago()
        g.board = [row[:] for row in self.board]
        g.current = self.current
        g._last_move = self._last_move
        return g

    def render(self) -> str:
        lines = []
        for r in range(6):
            row = " ".join(self.board[r])
            if r == 2:
                lines.append(row + "\n" + "-" * 11)
            else:
                lines.append(row)
        return "\n".join(lines)

    def legal_placements(self):
        return [(r, c) for r in range(6) for c in range(6) if self.board[r][c] == "."]

    def legal_rotations(self):
        rots = []
        for q in range(4):
            rots.append((q, "CW"))
            rots.append((q, "CCW"))
        return rots

    def legal_moves(self):
        placements = self.legal_placements()
        if not placements:
            return []
        rots = self.legal_rotations()
        return [((r, c), rot) for (r, c) in placements for rot in rots]

    def play(self, placement: Coord, rotation: Rotation) -> bool:
        (r, c) = placement
        (q, d) = rotation
        if not (0 <= r < 6 and 0 <= c < 6):
            return False
        if self.board[r][c] != ".":
            return False
        if q not in (0, 1, 2, 3) or d not in ("CW", "CCW"):
            return False
        self.board[r][c] = self.current
        self._rotate_quadrant(q, d)
        self._last_move = ((r, c), (q, d))
        if not self.is_terminal():
            self.current = "O" if self.current == "X" else "X"
        return True

    def _quad_bounds(self, q: int):
        if q == 0:
            return 0, 3, 0, 3
        if q == 1:
            return 0, 3, 3, 6
        if q == 2:
            return 3, 6, 0, 3
        if q == 3:
            return 3, 6, 3, 6
        raise ValueError("Quadrant invalide")

    def _rotate_quadrant(self, q: int, direction: str) -> None:
        r0, r1, c0, c1 = self._quad_bounds(q)
        sub = [row[c0:c1] for row in self.board[r0:r1]]  # 3x3
        n = 3
        rotated = [["" for _ in range(n)] for _ in range(n)]
        if direction == "CW":
            for i in range(n):
                for j in range(n):
                    rotated[j][n - 1 - i] = sub[i][j]
        else:  # CCW
            for i in range(n):
                for j in range(n):
                    rotated[n - 1 - j][i] = sub[i][j]
        for i in range(n):
            for j in range(n):
                self.board[r0 + i][c0 + j] = rotated[i][j]

    def winner(self) -> Optional[str]:
        x5 = self._has_five("X")
        o5 = self._has_five("O")
        if x5 and o5:
            return "Draw"
        if x5:
            return "X"
        if o5:
            return "O"
        if all(self.board[r][c] != "." for r in range(6) for c in range(6)):
            return "Draw"
            return None
        

    def is_terminal(self) -> bool:
        return self.winner() is not None

    def _has_five(self, player: str) -> bool:
        B = self.board
        n = 6
        target = player
        dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(n):
            for c in range(n):
                if B[r][c] != target:
                    continue
                for dr, dc in dirs:
                    rr, cc = r, c
                    ok = True
                    for _ in range(1, 5):
                        rr += dr
                        cc += dc
                        if not (0 <= rr < n and 0 <= cc < n):
                            ok = False
                            break
                        if B[rr][cc] != target:
                            ok = False
                            break
                    if ok:
                        return True
        return False


# --- GUI Tkinter ---
CELL_SIZE = 80
PADDING = 20
BOARD_PIX = CELL_SIZE * 6
WIDTH = BOARD_PIX + 2 * PADDING
HEIGHT = BOARD_PIX + 160  # un peu plus pour les contrôles


class PentagoGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pentago — GUI")
        self.game = Pentago()

        # état d'UI
        self.awaiting_rotation = False
        self.last_placement: Optional[Coord] = None
        self.bot_thinking = False

        # mode de jeu: PVP ou BOT_O (le bot joue O)
        self.mode_var = tk.StringVar(value="PVP")  # "PVP" | "BOT_O"
        self.bot_symbol = "O"

        # Canvas
        self.canvas = tk.Canvas(root, width=WIDTH, height=BOARD_PIX + 2 * PADDING)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click_board)

        # Panneau du haut: statut + mode
        top_frame = tk.Frame(root)
        top_frame.pack(pady=6)

        self.status = tk.Label(top_frame, text="Joueur X — placez un pion", font=("Arial", 12))
        self.status.pack(side=tk.LEFT, padx=10)

        mode_frame = tk.Frame(top_frame)
        mode_frame.pack(side=tk.RIGHT)
        tk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="JcJ", variable=self.mode_var, value="PVP",
                       command=self.on_mode_change).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Joueur vs Bot (bot=O)", variable=self.mode_var, value="BOT_O",
                       command=self.on_mode_change).pack(side=tk.LEFT)

        # Boutons de rotation
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        self.rot_buttons = []
        for q in range(4):
            f = tk.Frame(btn_frame, padx=4, pady=2, bd=0)
            f.pack(side=tk.LEFT)
            tk.Label(f, text=f"Q{q}").pack()
            b1 = tk.Button(f, text="CW", width=4, command=lambda q=q: self.rotate(q, "CW"))
            b2 = tk.Button(f, text="CCW", width=4, command=lambda q=q: self.rotate(q, "CCW"))
            b1.pack(padx=1, pady=1)
            b2.pack(padx=1, pady=1)
            self.rot_buttons.extend([b1, b2])

        # Reset
        self.reset_btn = tk.Button(root, text="Reset", command=self.reset)
        self.reset_btn.pack(pady=6)

        self.set_rotation_buttons_state(False)
        self.draw()

    # --- Helpers UI ---
    def set_rotation_buttons_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for b in self.rot_buttons:
            b.config(state=state)

    def block_inputs(self, blocked: bool):
        # Bloque les clics/rotations pendant que le bot réfléchit
        if blocked:
            self.canvas.unbind("<Button-1>")
            self.set_rotation_buttons_state(False)
        else:
            self.canvas.bind("<Button-1>", self.on_click_board)
            if self.awaiting_rotation:
                self.set_rotation_buttons_state(True)

    def board_to_xy(self, r: int, c: int):
        x = PADDING + c * CELL_SIZE + CELL_SIZE // 2
        y = PADDING + r * CELL_SIZE + CELL_SIZE // 2
        return x, y

    def xy_to_cell(self, x: int, y: int) -> Optional[Coord]:
        x0 = PADDING
        y0 = PADDING
        x1 = x0 + BOARD_PIX
        y1 = y0 + BOARD_PIX
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        c = (x - x0) // CELL_SIZE
        r = (y - y0) // CELL_SIZE
        r = int(r)
        c = int(c)
        if 0 <= r < 6 and 0 <= c < 6:
            return (r, c)
        return None

    # --- Dessin ---
    def draw(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, WIDTH, BOARD_PIX + 2 * PADDING, fill="#f3f3f3", width=0)

        x0 = PADDING
        y0 = PADDING
        x1 = x0 + BOARD_PIX
        y1 = y0 + BOARD_PIX

        self.canvas.create_rectangle(x0, y0, x1, y1)
        self.canvas.create_line(x0 + 3 * CELL_SIZE, y0, x0 + 3 * CELL_SIZE, y1, width=3)
        self.canvas.create_line(x0, y0 + 3 * CELL_SIZE, x1, y0 + 3 * CELL_SIZE, width=3)

        for i in range(7):
            self.canvas.create_line(x0 + i * CELL_SIZE, y0, x0 + i * CELL_SIZE, y1, dash=(2,))
            self.canvas.create_line(x0, y0 + i * CELL_SIZE, x1, y0 + i * CELL_SIZE, dash=(2,))

        for r in range(6):
            for c in range(6):
                v = self.game.board[r][c]
                if v == ".":
                    continue
                cx, cy = self.board_to_xy(r, c)
                radius = CELL_SIZE * 0.35
                self.canvas.create_oval(
                    cx - radius, cy - radius, cx + radius, cy + radius,
                    fill="#222" if v == "X" else "white", outline="#222", width=2
                )

    # --- Gestion des événements ---
    def on_click_board(self, event):
        if self.game.is_terminal() or self.bot_thinking:
            return
        if self.awaiting_rotation:
            return
        cell = self.xy_to_cell(event.x, event.y)
        if cell is None:
            return
        r, c = cell
        if self.game.board[r][c] != ".":
            return

        self.game.board[r][c] = self.game.current
        self.last_placement = (r, c)
        self.awaiting_rotation = True
        self.set_rotation_buttons_state(True)
        self.status.config(text=f"Rotation requise — choisissez Q0..Q3 + (CW/CCW)")
        self.draw()

    def rotate(self, q: int, direction: str):
        if not self.awaiting_rotation or self.last_placement is None or self.bot_thinking:
            return

        self.game._rotate_quadrant(q, direction)
        self.game._last_move = (self.last_placement, (q, direction))
        winner = self.game.winner()
        self.awaiting_rotation = False
        self.last_placement = None
        self.set_rotation_buttons_state(False)
        self.draw()

        if winner is None:
            self.game.current = "O" if self.game.current == "X" else "X"
            self.status.config(text=f"Joueur {self.game.current} — placez un pion")
            # Si on est en mode bot et que c'est au bot de jouer, lancer le bot
            if self.mode_var.get() == "BOT_O" and self.game.current == self.bot_symbol:
                self.root.after(200, self.bot_play)
        else:
            if winner == "Draw":
                self.status.config(text="Match nul ! Rejouez si vous voulez.")
            else:
                self.status.config(text=f"Victoire de {winner} !")
        self.draw()

    # --- Bot aléatoire (joue O) ---
    def bot_play(self):
        if self.game.is_terminal() or self.mode_var.get() != "BOT_O" or self.game.current != self.bot_symbol:
            return
        self.bot_thinking = True
        self.block_inputs(True)
        self.status.config(text="Le bot joue...")

        from greedy_ai_eloi import GreedyAI
        ai = GreedyAI(self.bot_symbol)
        move = ai.choose_move(self.game)

        if move is None:
            self.bot_thinking = False
            self.block_inputs(False)
            return

        placement, rotation = move
        self.game.play(placement, rotation)


        winner = self.game.winner()
        self.draw()

        if winner is None:
            self.status.config(text=f"Joueur {self.game.current} — placez un pion")
        else:
            if winner == "Draw":
                self.status.config(text="Match nul ! Rejouez si vous voulez.")
            else:
                self.status.config(text=f"Victoire de {winner} !")

        self.bot_thinking = False
        self.block_inputs(False)

    # --- Contrôles ---
    def on_mode_change(self):
        # Si on passe en mode bot et que c'est déjà au bot de jouer, on le déclenche
        if self.mode_var.get() == "BOT_O" and self.game.current == self.bot_symbol and not self.game.is_terminal():
            self.root.after(100, self.bot_play)

    def reset(self):
        self.game = Pentago()
        self.awaiting_rotation = False
        self.last_placement = None
        self.bot_thinking = False
        self.set_rotation_buttons_state(False)
        self.block_inputs(False)
        self.status.config(text="Joueur X — placez un pion")
        self.draw()


if __name__ == "__main__":
    root = tk.Tk()
    gui = PentagoGUI(root)
    root.mainloop()
