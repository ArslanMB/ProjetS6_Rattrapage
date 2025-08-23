# main.py
import sys
import pygame

from core.pentago_logic import PentagoGame   # garde tes chemins actuels
from core.constants import PLAYER_1, PLAYER_2, BOARD_ROWS, BOARD_COLS, QUADRANT_SIZE

from alphabeta_ia.alpha_beta import (
    timed_find_best_move_minimax, reset_timing, get_timing_stats
)

# -- UI --
from gui.draw import (
    draw_game_board, rotation_buttons,
    SCREEN_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT,
    COLOR_BOARD, COLOR_BLACK, COLOR_RED
)

def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Pentago")
    clock = pygame.time.Clock()

    game = PentagoGame()

    title_font = pygame.font.SysFont("arialunicode", 28)
    button_font = pygame.font.SysFont("arialunicode", 28)
    msg_font = pygame.font.Font(None, 60)
    restart_font = pygame.font.Font(None, 30)

    # Boutons du menu
    start_pvp_button_rect = pygame.Rect(SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 - 70, 300, 70)
    start_pva_button_rect = pygame.Rect(SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 + 20, 300, 70)
    start_aiai_button_rect = pygame.Rect(SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 + 110, 300, 70)

    game_mode = None  # "PVP", "PVA", "AIAI"

    while True:
        # === Events ===
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # --- Menu principal ---
                if game.game_state == 'START_MENU':
                    if start_pvp_button_rect.collidepoint(event.pos):
                        game_mode = "PVP"; game.game_state = 'PLAYING'; reset_timing()
                    elif start_pva_button_rect.collidepoint(event.pos):
                        game_mode = "PVA"; game.game_state = 'PLAYING'; reset_timing()
                    elif start_aiai_button_rect.collidepoint(event.pos):
                        game_mode = "AIAI"; game.game_state = 'PLAYING'; reset_timing()
                
                # ----- En cours de partie -----
                elif game.game_state == 'PLAYING':
                    if game.game_phase == "PLACEMENT":
                        if game_mode == "PVP" or (game_mode == "PVA" and game.current_player == PLAYER_1):
                            mouseX, mouseY = event.pos
                            row = (mouseY - 55) // 110  # correspond à BOARD_OFFSET_Y / SQUARE_SIZE (voir draw.py)
                            col = (mouseX - 55) // 110
                            if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
                                game.place_marble(row, col)

                    elif game.game_phase == "ROTATION":
                        if game_mode == "PVP" or (game_mode == "PVA" and game.current_player == PLAYER_1):
                            # D'abord: boutons ↺/↻ 
                            clicked_button = False
                            for rect, q_idx, direction in rotation_buttons():
                                if rect.collidepoint(event.pos):
                                    game.start_quadrant_rotation_animation(q_idx, direction)
                                    clicked_button = True
                                    break

                            if not clicked_button:
                                # Alternative: clic haut/bas du quadrant
                                mouseX, mouseY = event.pos
                                SQUARE_SIZE = 110; BOARD_OFFSET_X = 55; BOARD_OFFSET_Y = 55
                                quadrant_x = (mouseX - BOARD_OFFSET_X) // (QUADRANT_SIZE * SQUARE_SIZE)
                                quadrant_y = (mouseY - BOARD_OFFSET_Y) // (QUADRANT_SIZE * SQUARE_SIZE)
                                quadrant_idx = quadrant_y * 2 + quadrant_x
                                quadrant_center_y_on_screen = BOARD_OFFSET_Y + (quadrant_y * QUADRANT_SIZE + 1.5) * SQUARE_SIZE
                                direction = 1 if mouseY < quadrant_center_y_on_screen else -1
                                if 0 <= quadrant_idx < 4:
                                    game.start_quadrant_rotation_animation(quadrant_idx, direction)

                # --- Fin de partie ---
                elif game.game_state == 'GAME_OVER':
                    game.reset_game()
                    game_mode = None

        # ---- Updates ----
        just_finished_anim = False # pour savoir si on vient de finir une anim
        if game.game_state == 'PLAYING':
            if game.game_phase == "ANIMATING_ROTATION":
                if game.update_rotation_animation():
                    just_finished_anim = True

            # IA vs IA
            if not just_finished_anim and game_mode == "AIAI" and game.game_phase != "ANIMATING_ROTATION":
                pygame.time.wait(10)  # respirer un peu
                if game.game_phase == "PLACEMENT":
                    depth = "A" if game.current_player == PLAYER_1 else 3
                    best_move, _dt = timed_find_best_move_minimax(game, depth=depth)
                    if best_move:
                        r, c, quad_idx, direction = best_move
                        game.place_marble(r, c)
                        if game.game_state == 'PLAYING':
                            game.start_quadrant_rotation_animation(quad_idx, direction)

            # Joueur vs IA (IA = PLAYER_2)
            if not just_finished_anim and game_mode == "PVA" and game.current_player == PLAYER_2 and game.game_phase != "ANIMATING_ROTATION":
                pygame.time.wait(10)
                if game.game_phase == "PLACEMENT":
                    best_move, _dt = timed_find_best_move_minimax(game, depth=3)
                    if best_move:
                        r, c, quad_idx, direction = best_move
                        game.place_marble(r, c)
                        if game.game_state == 'PLAYING':
                            game.start_quadrant_rotation_animation(quad_idx, direction)

        # ----- Draw -----
        if game.game_state == 'START_MENU':
            screen.fill(COLOR_BOARD)
            title_text = title_font.render("PENTAGO", True, COLOR_BLACK)
            title_rect = title_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3 - 100))
            screen.blit(title_text, title_rect)

            pygame.draw.rect(screen, (70, 130, 180), start_pvp_button_rect, border_radius=15)
            pvp_button_text = button_font.render("Joueur vs Joueur", True, (255, 255, 255))
            screen.blit(pvp_button_text, pvp_button_text.get_rect(center=start_pvp_button_rect.center))

            pygame.draw.rect(screen, (70, 130, 180), start_pva_button_rect, border_radius=15)
            pva_button_text = button_font.render("Joueur vs IA", True, (255, 255, 255))
            screen.blit(pva_button_text, pva_button_text.get_rect(center=start_pva_button_rect.center))

            pygame.draw.rect(screen, (70, 130, 180), start_aiai_button_rect, border_radius=15)
            aiai_button_text = button_font.render("IA vs IA", True, (255, 255, 255))
            screen.blit(aiai_button_text, aiai_button_text.get_rect(center=start_aiai_button_rect.center))

        elif game.game_state == 'PLAYING':
            draw_game_board(screen, game, timing_stats=get_timing_stats())

        elif game.game_state == 'GAME_OVER':
            draw_game_board(screen, game, timing_stats=get_timing_stats())
            # Overlay + message
            msg = "Le joueur BLANC gagne !" if game.winner == PLAYER_1 else \
                  "Le joueur NOIR gagne !" if game.winner == PLAYER_2 else "Partie Nulle !"

            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0, 0))

            text_surface = msg_font.render(msg, True, COLOR_RED)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            pygame.draw.rect(screen, (200, 180, 140), text_rect.inflate(20, 20), border_radius=10)
            screen.blit(text_surface, text_rect)

            restart_text = restart_font.render("Cliquez pour retourner au menu", True, (255, 255, 255))
            screen.blit(restart_text, restart_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 50)))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
