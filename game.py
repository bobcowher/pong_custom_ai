import pygame
import sys

from pygame.cursors import ball
from assets import Paddle, Ball
import random

class Pong:

    def __init__(self, window_width=1280, window_height=960, fps=60, player1="human", player2="bot"):
        
        self.window_width = window_width
        self.window_height = window_height

        pygame.init()
        pygame.display.set_caption("Pong")

        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.fps = fps

        self.background_color = (0, 0, 0)

        self.paddle_height = 120
        self.paddle_width = 20
       
        self.player_1_color = (50, 205, 50)
        self.player_2_color = (138, 43, 226)
        
        self.font = pygame.font.SysFont(None, 70)

        self.player1 = player1
        self.player2 = player2

        self.player_1_score = 0
        self.player_2_score = 0

        self.player_1_paddle = Paddle(x=window_width - 2 * (window_width / 64), 
                                      y=(window_height / 2) - (self.paddle_height / 2), 
                                      player_color=self.player_1_color, 
                                      height=self.paddle_height,
                                      width=self.paddle_width,
                                      window_height=window_height); 
        
        self.player_2_paddle = Paddle(x=(window_width / 64), 
                                      y=(window_height / 2) - (self.paddle_height / 2), 
                                      player_color=self.player_2_color, 
                                      height=self.paddle_height,
                                      width=self.paddle_width,
                                      window_height=window_height); 

        self.ball = Ball(window_height=window_height,
                         window_width=window_width,
                         height=20,
                         width=20,
                         player_1_paddle=self.player_1_paddle,
                         player_2_paddle=self.player_2_paddle)

        self.bot_move_queue = []


    def game_loop(self):
        while(True):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            keys = pygame.key.get_pressed()

            if self.player1 == "human": 
                if keys[pygame.K_k]:
                    self.player_1_paddle.move(1)
                elif keys[pygame.K_j]:
                    self.player_1_paddle.move(2)
            if self.player2 == "human":
                if keys[pygame.K_w]:
                    self.player_2_paddle.move(1)
                elif keys[pygame.K_s]:
                    self.player_2_paddle.move(2)

            if self.player1 == "bot":
                move = self.get_bot_move()
                self.player_1_paddle.move(move)
            if self.player2 == "bot":
                move = self.get_bot_move()
                self.player_2_paddle.move(move)

            self.step()


    def get_bot_move(self):
         
        random_target = 0.05
        rqueue = 5

        if self.bot_move_queue.__len__() > 0:
            pass 
        elif random.random() <= random_target:
            next_move = random.randint(0, 2)
            
            for i in range(rqueue):
                self.bot_move_queue.append(next_move)
        else:
            if(self.ball.vy > 0):
                self.bot_move_queue.append(2)
            else:
                self.bot_move_queue.append(1)

        return self.bot_move_queue.pop(0)       



    def fill_background(self):
        self.screen.fill(self.background_color)

        player_1_score_surface = self.font.render(f'Score: {self.player_1_score}', True, self.player_1_color)
        self.screen.blit(player_1_score_surface, ((self.window_width / 2) + 20, 10))
        
        player_2_score_surface = self.font.render(f'Score: {self.player_2_score}', True, self.player_2_color)
        self.screen.blit(player_2_score_surface, ((self.window_width / 2) - player_2_score_surface.get_width() - 20, 10))


    def step(self):
        self.fill_background()
        self.player_1_paddle.draw(screen=self.screen)
        self.player_2_paddle.draw(screen=self.screen)
        self.ball.move()
        self.ball.draw(screen=self.screen)
        self.clock.tick(self.fps)
        pygame.display.flip()

        if(self.ball.x < 0):
            self.player_1_score += 1
            self.ball.spawn()
        elif(self.ball.x > self.window_width):
            self.player_2_score += 1
            self.ball.spawn()




