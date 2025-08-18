import pygame
import sys
from pentago_logic import PentagoGame, BOARD_ROWS, BOARD_COLS, QUADRANT_SIZE, PLAYER_1, PLAYER_2, ROTATION_SPEED
from pentago_bot_1 import find_best_move_minimax

SQUARE_SIZE = 100
BOARD_WIDTH = BOARD_COLS * SQUARE_SIZE
BOARD_HEIGHT = BOARD_ROWS * SQUARE_SIZE
BOARD_PADDING = 50
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
COLOR_HIGHLIGHT = (255, 255, 0, 100)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_TEXT = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_WHITE_SHADOW = (180, 180, 180)
COLOR_BLACK_SHADOW = (20, 20, 20)

AI_DEPTH = 2 
AI_TIME_BUDGET_SECONDS = 2

ROT_BTN_W = 36
ROT_BTN_H = 36
ROT_BTN_MARGIN = 6
SYMBOL_CCW = '↺'
SYMBOL_CW  = '↻'
ROTATION_BUTTONS = []  

def draw_marble_3d(screen, center_x, center_y, color, shadow_color, radius):
    pygame.draw.circle(screen, shadow_color, (center_x + 3, center_y + 3), radius)
    pygame.draw.circle(screen, color, (center_x, center_y), radius)

def draw_game_board(screen, game_instance):
    screen.fill(COLOR_BLACK)
    pygame.draw.rect(screen, COLOR_BOARD, (BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_WIDTH, BOARD_HEIGHT))

    for i in range(1, BOARD_ROWS):
        line_width = 3 if i % QUADRANT_SIZE == 0 else 1
        pygame.draw.line(screen, COLOR_GRID,
                         (BOARD_OFFSET_X, BOARD_OFFSET_Y + i * SQUARE_SIZE),
                         (BOARD_OFFSET_X + BOARD_WIDTH, BOARD_OFFSET_Y + i * SQUARE_SIZE), line_width)
        pygame.draw.line(screen, COLOR_GRID,
                         (BOARD_OFFSET_X + i * SQUARE_SIZE, BOARD_OFFSET_Y),
                         (BOARD_OFFSET_X + i * SQUARE_SIZE, BOARD_OFFSET_Y + BOARD_HEIGHT), line_width)

    if game_instance.game_phase == "ANIMATING_ROTATION":
        board_to_draw = game_instance.board_before_rotation
        if board_to_draw is None:
            print("Erreur : board_before_rotation n'est pas initialisé.")
            return
        q_row_start_idx = (game_instance.animating_quadrant_idx // 2) * QUADRANT_SIZE
        q_col_start_idx = (game_instance.animating_quadrant_idx % 2) * QUADRANT_SIZE
        
        quadrant_surface = pygame.Surface((QUADRANT_SIZE * SQUARE_SIZE, QUADRANT_SIZE * SQUARE_SIZE), pygame.SRCALPHA)
        quadrant_surface.fill((0,0,0,0))

        for r_q in range(QUADRANT_SIZE):
            for c_q in range(QUADRANT_SIZE):
                marble_val = board_to_draw[q_row_start_idx + r_q, q_col_start_idx + c_q]
                if marble_val != 0:
                    center_x = int(c_q * SQUARE_SIZE + SQUARE_SIZE / 2)
                    center_y = int(r_q * SQUARE_SIZE + SQUARE_SIZE / 2)
                    color = COLOR_WHITE if marble_val == PLAYER_1 else COLOR_BLACK
                    shadow_color = COLOR_WHITE_SHADOW if marble_val == PLAYER_1 else COLOR_BLACK_SHADOW
                    draw_marble_3d(quadrant_surface, center_x, center_y, color, shadow_color, CIRCLE_RADIUS)
        
        rotated_quadrant_surface = pygame.transform.rotate(quadrant_surface, game_instance.animation_angle)
        
        original_center_x = BOARD_OFFSET_X + (q_col_start_idx * SQUARE_SIZE + 1.5 * SQUARE_SIZE)
        original_center_y = BOARD_OFFSET_Y + (q_row_start_idx * SQUARE_SIZE + 1.5 * SQUARE_SIZE)
        
        rotated_rect = rotated_quadrant_surface.get_rect(center=(original_center_x, original_center_y))
        screen.blit(rotated_quadrant_surface, rotated_rect.topleft)

        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if not (q_row_start_idx <= r < q_row_start_idx + QUADRANT_SIZE and
                        q_col_start_idx <= c < q_row_start_idx + QUADRANT_SIZE):  
                    center_x = BOARD_OFFSET_X + int(c * SQUARE_SIZE + SQUARE_SIZE / 2)
                    center_y = BOARD_OFFSET_Y + int(r * SQUARE_SIZE + SQUARE_SIZE / 2)
                    marble_val = board_to_draw[r, c]
                    if marble_val == PLAYER_1:
                        draw_marble_3d(screen, center_x, center_y, COLOR_WHITE, COLOR_WHITE_SHADOW, CIRCLE_RADIUS)
                    elif marble_val == PLAYER_2:
                        draw_marble_3d(screen, center_x, center_y, COLOR_BLACK, COLOR_BLACK_SHADOW, CIRCLE_RADIUS)
    else:
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                center_x = BOARD_OFFSET_X + int(c * SQUARE_SIZE + SQUARE_SIZE / 2)
                center_y = BOARD_OFFSET_Y + int(r * SQUARE_SIZE + SQUARE_SIZE / 2)
                marble_val = game_instance.board[r, c]
                if marble_val == PLAYER_1:
                    draw_marble_3d(screen, center_x, center_y, COLOR_WHITE, COLOR_WHITE_SHADOW, CIRCLE_RADIUS)
                elif marble_val == PLAYER_2:
                    draw_marble_3d(screen, center_x, center_y, COLOR_BLACK, COLOR_BLACK_SHADOW, CIRCLE_RADIUS)

    if game_instance.game_phase == "ROTATION":
        font = pygame.font.Font(None, INFO_FONT_SIZE)
        text_surface = font.render("Rotation: cliquez un bouton ↺/↻ sur un quadrant (ou clic haut/bas).", True, COLOR_BLACK) 
        screen.blit(text_surface, (BOARD_OFFSET_X, 10))
        draw_rotation_controls(screen)  

def draw_rotation_controls(screen):
    global ROTATION_BUTTONS
    ROTATION_BUTTONS = []
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

        if i in (0, 1): 
            y_btn = quadrant_rect.top - 30
        else:  
            y_btn = quadrant_rect.bottom + 30

        ccw_center = (quadrant_rect.centerx - 30, y_btn)
        ccw_rect = pygame.Rect(0, 0, 40, 40)
        ccw_rect.center = ccw_center
        pygame.draw.circle(screen, COLOR_BUTTON, ccw_center, 20)
        text_ccw = font.render("↺", True, COLOR_BUTTON_TEXT)
        screen.blit(text_ccw, text_ccw.get_rect(center=ccw_center))
        ROTATION_BUTTONS.append((ccw_rect, i, +1)) 

        cw_center = (quadrant_rect.centerx + 30, y_btn)
        cw_rect = pygame.Rect(0, 0, 40, 40)
        cw_rect.center = cw_center
        pygame.draw.circle(screen, COLOR_BUTTON, cw_center, 20)
        text_cw = font.render("↻", True, COLOR_BUTTON_TEXT)
        screen.blit(text_cw, text_cw.get_rect(center=cw_center))
        ROTATION_BUTTONS.append((cw_rect, i, -1)) 


def _draw_button_symbol(screen, rect, symbol):
    font = pygame.font.SysFont("arialunicode", 28)
    text_surface = font.render(symbol, True, COLOR_BUTTON_TEXT)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Pentago")
    game = PentagoGame()
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("arialunicode", 28)
    button_font = pygame.font.SysFont("arialunicode", 28)
    
    start_pvp_button_rect = pygame.Rect(SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 - 70, 300, 70)
    start_pva_button_rect = pygame.Rect(SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 + 20, 300, 70)

    game_mode = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if game.game_state == 'START_MENU':
                    if start_pvp_button_rect.collidepoint(event.pos):
                        game_mode = "PVP"
                        game.game_state = 'PLAYING'
                    elif start_pva_button_rect.collidepoint(event.pos):
                        game_mode = "PVA"
                        game.game_state = 'PLAYING'

                elif game.game_state == 'PLAYING':
                    if game.game_phase == "PLACEMENT":
                        if game_mode == "PVP" or (game_mode == "PVA" and game.current_player == PLAYER_1):
                            mouseX, mouseY = event.pos
                            row, col = (mouseY - BOARD_OFFSET_Y) // SQUARE_SIZE, (mouseX - BOARD_OFFSET_X) // SQUARE_SIZE
                            if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
                                game.place_marble(row, col)

                    elif game.game_phase == "ROTATION":
                        if game_mode == "PVP" or (game_mode == "PVA" and game.current_player == PLAYER_1):
                            mouseX, mouseY = event.pos

                            clicked_button = False
                            for rect, q_idx, direction in ROTATION_BUTTONS:
                                if rect.collidepoint(event.pos):
                                    game.start_quadrant_rotation_animation(q_idx, direction)
                                    clicked_button = True
                                    break

                            if not clicked_button:
                                # j'ai gardé le meca d'origine au cas ou (le joueur a le choix)
                                quadrant_x = (mouseX - BOARD_OFFSET_X) // (QUADRANT_SIZE * SQUARE_SIZE)
                                quadrant_y = (mouseY - BOARD_OFFSET_Y) // (QUADRANT_SIZE * SQUARE_SIZE)
                                quadrant_idx = quadrant_y * 2 + quadrant_x
                                quadrant_center_y_on_screen = BOARD_OFFSET_Y + (quadrant_y * QUADRANT_SIZE + 1.5) * SQUARE_SIZE
                                direction = 1 if mouseY < quadrant_center_y_on_screen else -1
                                
                                if 0 <= quadrant_idx < 4:
                                    game.start_quadrant_rotation_animation(quadrant_idx, direction)

                elif game.game_state == 'GAME_OVER':
                    game.reset_game()
                    game_mode = None

        if game.game_state == 'PLAYING':
            if game.game_phase == "ANIMATING_ROTATION":
                game.update_rotation_animation()
            
            if game_mode == "PVA" and game.current_player == PLAYER_2 and game.game_phase != "ANIMATING_ROTATION":
                pygame.time.wait(10)

                if game.game_phase == "PLACEMENT":
                    best_move = find_best_move_minimax(game, depth=2)
                    if best_move:
                        r, c, quad_idx, direction = best_move
                        game.place_marble(r, c)
                        if game.game_state == 'PLAYING':
                            game.start_quadrant_rotation_animation(quad_idx, direction)

        if game.game_state == 'START_MENU':
            screen.fill(COLOR_BOARD)
            
            title_text = title_font.render("PENTAGO", True, COLOR_BLACK)
            title_rect = title_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3 - 100))
            screen.blit(title_text, title_rect)
            
            pygame.draw.rect(screen, COLOR_BUTTON, start_pvp_button_rect, border_radius=15)
            pvp_button_text = button_font.render("Joueur vs Joueur", True, COLOR_BUTTON_TEXT)
            pvp_button_rect = pvp_button_text.get_rect(center=start_pvp_button_rect.center)
            screen.blit(pvp_button_text, pvp_button_rect)

            pygame.draw.rect(screen, COLOR_BUTTON, start_pva_button_rect, border_radius=15)
            pva_button_text = button_font.render("Joueur vs IA", True, COLOR_BUTTON_TEXT)
            pva_button_rect = pva_button_text.get_rect(center=start_pva_button_rect.center)
            screen.blit(pva_button_text, pva_button_rect)

        elif game.game_state == 'PLAYING':
            draw_game_board(screen, game)

        elif game.game_state == 'GAME_OVER':
            draw_game_board(screen, game)
            
            msg_font = pygame.font.Font(None, 60)
            if game.winner == PLAYER_1:
                msg = "Le joueur BLANC gagne !"
            elif game.winner == PLAYER_2:
                msg = "Le joueur NOIR gagne !"
            else:
                msg = "Partie Nulle !"

            text_surface = msg_font.render(msg, True, COLOR_RED)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0,0))
            
            pygame.draw.rect(screen, COLOR_BOARD, text_rect.inflate(20, 20), border_radius=10)
            screen.blit(text_surface, text_rect)
            
            restart_font = pygame.font.Font(None, 30)
            restart_text = restart_font.render("Cliquez pour retourner au menu", True, COLOR_WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 50))
            screen.blit(restart_text, restart_rect)

        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
