import pygame
import sys

class Pong:

    def __init__(self, window_width=1200, window_height=1200, fps=60):
        
        self.window_width = window_width
        self.window_height = window_height

        pygame.init()

        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.fps = fps

        self.background_color = (0, 0, 0)
        


    def game_loop(self):
        while(True):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


            self.step()


    def fill_background(self):
        self.screen.fill(self.background_color)


    def step(self):
        self.fill_background()
        self.clock.tick(self.fps)


