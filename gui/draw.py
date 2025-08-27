# gui/draw.py
import pygame
from core.constants import PLAYER_1, PLAYER_2, QUADRANT_SIZE, BOARD_ROWS, BOARD_COLS

# ==== Constantes UI (uniquement visuelles) ====
SQUARE_SIZE = 110
BOARD_WIDTH = BOARD_COLS * SQUARE_SIZE
BOARD_HEIGHT = BOARD_ROWS * SQUARE_SIZE
BOARD_PADDING = 55
SCREEN_WIDTH = BOARD_WIDTH + 2 * BOARD_PADDING
SCREEN_HEIGHT = BOARD_HEIGHT + 2 * BOARD_PADDING
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
BOARD_OFFSET_X = BOARD_PADDING
BOARD_OFFSET_Y = BOARD_PADDING
CIRCLE_RADIUS = SQUARE_SIZE // 2 - 10
INFO_FONT_SIZE = 30

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BOARD = (200, 180, 140)
COLOR_GRID = (50, 50, 50)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_TEXT = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_WHITE_SHADOW = (180, 180, 180)
COLOR_BLACK_SHADOW = (20, 20, 20)

# Boutons de rotation (mis à jour à chaque draw)
_ROTATION_BUTTONS = []  # liste de tuples: (rect, quadrant_idx, direction)

def rotation_buttons():
    """Expose une copie des boutons calculés au dernier frame."""
    return list(_ROTATION_BUTTONS)

# ==== Fonctions de dessin ====
def _draw_marble_3d(screen, cx, cy, color, shadow_color, radius):
    pygame.draw.circle(screen, shadow_color, (cx + 3, cy + 3), radius)
    pygame.draw.circle(screen, color, (cx, cy), radius)

def _draw_rotation_controls(screen):
    global _ROTATION_BUTTONS
    _ROTATION_BUTTONS = []
    font = pygame.font.SysFont("arialunicode", 28)

    for i in range(4):
        row_start = (i // 2) * QUADRANT_SIZE * SQUARE_SIZE
        col_start = (i % 2) * QUADRANT_SIZE * SQUARE_SIZE
        quadrant_rect = pygame.Rect(
            BOARD_OFFSET_X + col_start,
            BOARD_OFFSET_Y + row_start,
            QUADRANT_SIZE * SQUARE_SIZE,
            QUADRANT_SIZE * SQUARE_SIZE
        )

        y_btn = quadrant_rect.top - 30 if i in (0, 1) else quadrant_rect.bottom + 30

        # ↺ (anti-horaire = +1)
        ccw_center = (quadrant_rect.centerx - 30, y_btn)
        ccw_rect = pygame.Rect(0, 0, 40, 40); ccw_rect.center = ccw_center
        pygame.draw.circle(screen, COLOR_BUTTON, ccw_center, 20)
        text_ccw = font.render("↺", True, COLOR_BUTTON_TEXT)
        screen.blit(text_ccw, text_ccw.get_rect(center=ccw_center))
        _ROTATION_BUTTONS.append((ccw_rect, i, +1))

        # ↻ (horaire = -1)
        cw_center = (quadrant_rect.centerx + 30, y_btn)
        cw_rect = pygame.Rect(0, 0, 40, 40); cw_rect.center = cw_center
        pygame.draw.circle(screen, COLOR_BUTTON, cw_center, 20)
        text_cw = font.render("↻", True, COLOR_BUTTON_TEXT)
        screen.blit(text_cw, text_cw.get_rect(center=cw_center))
        _ROTATION_BUTTONS.append((cw_rect, i, -1))

def draw_game_board(screen, game_instance, timing_stats=None):
    """
    Dessine le plateau (avec animation si nécessaire) + HUD.
    `timing_stats` est un dict optionnel {"moves", "avg", "last", "longest", "longest_move_index"}.
    """
    screen.fill(COLOR_BLACK)
    pygame.draw.rect(screen, COLOR_BOARD, (BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_WIDTH, BOARD_HEIGHT))

    # Grille + séparations de quadrants
    for i in range(1, BOARD_ROWS):
        line_width = 3 if i % QUADRANT_SIZE == 0 else 1
        # horizontales
        pygame.draw.line(screen, COLOR_GRID,
                         (BOARD_OFFSET_X, BOARD_OFFSET_Y + i * SQUARE_SIZE),
                         (BOARD_OFFSET_X + BOARD_WIDTH, BOARD_OFFSET_Y + i * SQUARE_SIZE), line_width)
        # verticales
        pygame.draw.line(screen, COLOR_GRID,
                         (BOARD_OFFSET_X + i * SQUARE_SIZE, BOARD_OFFSET_Y),
                         (BOARD_OFFSET_X + i * SQUARE_SIZE, BOARD_OFFSET_Y + BOARD_HEIGHT), line_width)

    if game_instance.game_phase == "ANIMATING_ROTATION":
        board_to_draw = game_instance.board_before_rotation
        if board_to_draw is None:
            return  # rien à dessiner de spécifique, sécurité

        q_row_start_idx = (game_instance.animating_quadrant_idx // 2) * QUADRANT_SIZE
        q_col_start_idx = (game_instance.animating_quadrant_idx % 2) * QUADRANT_SIZE

        quadrant_surface = pygame.Surface((QUADRANT_SIZE * SQUARE_SIZE, QUADRANT_SIZE * SQUARE_SIZE), pygame.SRCALPHA)
        quadrant_surface.fill((0, 0, 0, 0))

        # Billes du quadrant animé sur surface dédiée
        for r_q in range(QUADRANT_SIZE):
            for c_q in range(QUADRANT_SIZE):
                marble_val = board_to_draw[q_row_start_idx + r_q, q_col_start_idx + c_q]
                if marble_val != 0:
                    cx = int(c_q * SQUARE_SIZE + SQUARE_SIZE / 2)
                    cy = int(r_q * SQUARE_SIZE + SQUARE_SIZE / 2)
                    color = COLOR_WHITE if marble_val == PLAYER_1 else COLOR_BLACK
                    shadow = COLOR_WHITE_SHADOW if marble_val == PLAYER_1 else COLOR_BLACK_SHADOW
                    _draw_marble_3d(quadrant_surface, cx, cy, color, shadow, CIRCLE_RADIUS)

        rotated_quadrant_surface = pygame.transform.rotate(quadrant_surface, game_instance.animation_angle)

        original_center_x = BOARD_OFFSET_X + (q_col_start_idx * SQUARE_SIZE + 1.5 * SQUARE_SIZE)
        original_center_y = BOARD_OFFSET_Y + (q_row_start_idx * SQUARE_SIZE + 1.5 * SQUARE_SIZE)
        rotated_rect = rotated_quadrant_surface.get_rect(center=(original_center_x, original_center_y))
        screen.blit(rotated_quadrant_surface, rotated_rect.topleft)

        # Billes hors quadrant animé (⚠️ correctif: borne colonne utilisait le mauvais index)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if not (q_row_start_idx <= r < q_row_start_idx + QUADRANT_SIZE and
                        q_col_start_idx <= c < q_col_start_idx + QUADRANT_SIZE):
                    cx = BOARD_OFFSET_X + int(c * SQUARE_SIZE + SQUARE_SIZE / 2)
                    cy = BOARD_OFFSET_Y + int(r * SQUARE_SIZE + SQUARE_SIZE / 2)
                    marble_val = board_to_draw[r, c]
                    if marble_val == PLAYER_1:
                        _draw_marble_3d(screen, cx, cy, COLOR_WHITE, COLOR_WHITE_SHADOW, CIRCLE_RADIUS)
                    elif marble_val == PLAYER_2:
                        _draw_marble_3d(screen, cx, cy, COLOR_BLACK, COLOR_BLACK_SHADOW, CIRCLE_RADIUS)
    else:
        # Rendu normal (pas d’animation)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cx = BOARD_OFFSET_X + int(c * SQUARE_SIZE + SQUARE_SIZE / 2)
                cy = BOARD_OFFSET_Y + int(r * SQUARE_SIZE + SQUARE_SIZE / 2)
                marble_val = game_instance.board[r, c]
                if marble_val == PLAYER_1:
                    _draw_marble_3d(screen, cx, cy, COLOR_WHITE, COLOR_WHITE_SHADOW, CIRCLE_RADIUS)
                elif marble_val == PLAYER_2:
                    _draw_marble_3d(screen, cx, cy, COLOR_BLACK, COLOR_BLACK_SHADOW, CIRCLE_RADIUS)

    # Phase rotation : message + boutons
    if game_instance.game_phase == "ROTATION":
        font = pygame.font.Font(None, INFO_FONT_SIZE)
        msg = "Rotation: cliquez un bouton ↺/↻ sur un quadrant (ou clic haut/bas)."
        text_surface = font.render(msg, True, COLOR_BLACK)
        screen.blit(text_surface, (BOARD_OFFSET_X, 10))
        _draw_rotation_controls(screen)
    else:
        # Quand on n’est pas en phase rotation, on invalide la liste (pas cliquables)
        _ROTATION_BUTTONS = []

    # HUD timing (si fourni)
    if timing_stats:
        moves = timing_stats.get("moves", 0)
        avg = timing_stats.get("avg", 0.0)
        last = timing_stats.get("last", 0.0)
        longest = timing_stats.get("longest", 0.0)
        li = timing_stats.get("longest_move_index", None)
        hud = f"IA: coups {moves} | moy {avg:.2f}s | dernier {last:.2f}s | max {longest:.2f}s"
        if li:
            hud += f" (#{li})"
        font_hud = pygame.font.Font(None, 24)
        text = font_hud.render(hud, True, COLOR_WHITE)
        hud_pos = (BOARD_OFFSET_X, BOARD_OFFSET_Y + BOARD_HEIGHT + 8)
        screen.blit(text, hud_pos)
