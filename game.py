import pygame
import sys
from paddle import Paddle

class Pong:

    def __init__(self, window_width=1280, window_height=960, fps=60):
        
        self.window_width = window_width
        self.window_height = window_height

        pygame.init()

        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.fps = fps

        self.background_color = (0, 0, 0)

        self.paddle_height = 120
        self.paddle_width = 20
       
        self.player_1_paddle = Paddle(x=window_width - 2 * (window_width / 64), 
                                      y=(window_height / 2) - (self.paddle_height / 2), 
                                      player=1, 
                                      height=self.paddle_height,
                                      width=self.paddle_width); 
        
        self.player_2_paddle = Paddle(x=(window_width / 64), 
                                      y=(window_height / 2) - (self.paddle_height / 2), 
                                      player=2, 
                                      height=self.paddle_height,
                                      width=self.paddle_width); 


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
        self.player_1_paddle.draw(screen=self.screen)
        self.player_2_paddle.draw(screen=self.screen)
        self.clock.tick(self.fps)
        pygame.display.flip()


