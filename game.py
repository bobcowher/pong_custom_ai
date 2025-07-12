import pygame
import sys

from pygame.cursors import ball
from assets import Paddle, Ball
import random
import os
import time
import numpy as np
import cv2 
import torch


class Pong:

    def __init__(self, window_width=1280, window_height=960, fps=60, player1="human", player2="bot"):
       
        # Players should be human, bot, or ai

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
        self.announcement_font = pygame.font.SysFont(None, 150)

        self.player1 = player1
        self.player2 = player2

        if(player1 != "human"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.reset()


    def reset(self):
        
        self.player_1_score = 0
        self.player_2_score = 0

        self.top_score = 20 

        self.player_1_paddle = Paddle(x=self.window_width - 2 * (self.window_width / 64), 
                                      y=(self.window_height / 2) - (self.paddle_height / 2), 
                                      player_color=self.player_1_color, 
                                      height=self.paddle_height,
                                      width=self.paddle_width,
                                      window_height=self.window_height); 
        
        self.player_2_paddle = Paddle(x=(self.window_width / 64), 
                                      y=(self.window_height / 2) - (self.paddle_height / 2), 
                                      player_color=self.player_2_color, 
                                      height=self.paddle_height,
                                      width=self.paddle_width,
                                      window_height=self.window_height); 

        self.ball = Ball(window_height=self.window_height,
                         window_width=self.window_width,
                         height=20,
                         width=20,
                         player_1_paddle=self.player_1_paddle,
                         player_2_paddle=self.player_2_paddle)

        self.bot_move_queue = []

        return self._get_obs(), {}


    def game_loop(self):
        # Game loop for human players

        while(True):

            player_1_action = 0
            player_2_action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            keys = pygame.key.get_pressed()

            if self.player1 == "human": 
                if keys[pygame.K_k]:
                    player_1_action = 1
                elif keys[pygame.K_j]:
                    player_1_action = 2
            if self.player2 == "human":
                if keys[pygame.K_w]:
                    player_2_action = 1
                elif keys[pygame.K_s]:
                    player_2_action = 2

            if self.player1 == "bot":
                player_1_action = self.get_bot_move()
            if self.player2 == "bot":
                player_2_action = self.get_bot_move()

            self.step(player_1_action, player_2_action)
    

    def _get_obs(self):

        screen_array = pygame.surfarray.pixels3d(self.screen)

        # Transpose to (height, width, channels)
        screen_array = np.transpose(screen_array, (1, 0, 2))

        # Resize to 128x128
        downscaled_image = cv2.resize(screen_array, (128, 128), interpolation=cv2.INTER_NEAREST)

        # Convert to grayscale
        grayscale = cv2.cvtColor(downscaled_image, cv2.COLOR_RGB2GRAY)

        # Convert to PyTorch tensor
        observation = torch.from_numpy(grayscale).float().unsqueeze(0)

        # observation = observation / 255 # Reducing to decimals at this point doesn't work with a uint 8 replay buffer. 

        return observation


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


    def game_over(self):
        # Render the "You Died" message
        if(self.player_1_score >= self.top_score):
            game_over_surface = self.announcement_font.render('Player 1 Won', True, self.player_1_color)
        elif(self.player_2_score >= self.top_score):
            game_over_surface = self.announcement_font.render('Player 2 Won', True, self.player_2_color)
        
        game_over_rect = game_over_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))

        # Blit the message to the screen
        self.screen.blit(game_over_surface, game_over_rect)

        # Update the display to show the message
        pygame.display.flip()

        self.done = True

        time.sleep(10)

        pygame.quit()
        sys.exit()


    def fill_background(self):
        self.screen.fill(self.background_color)

        player_1_score_surface = self.font.render(f'Score: {self.player_1_score}', True, self.player_1_color)
        self.screen.blit(player_1_score_surface, ((self.window_width / 2) + 20, 10))
        
        player_2_score_surface = self.font.render(f'Score: {self.player_2_score}', True, self.player_2_color)
        self.screen.blit(player_2_score_surface, ((self.window_width / 2) - player_2_score_surface.get_width() - 20, 10))


    def step(self, player_1_action, player_2_action):
        
        done = False
        truncated = False

        player_1_reward = 0
        player_2_reward = 0

        self.player_1_paddle.move(player_1_action)
        self.player_2_paddle.move(player_2_action)

        self.fill_background()
        self.player_1_paddle.draw(screen=self.screen)
        self.player_2_paddle.draw(screen=self.screen)
        self.ball.move()
        self.ball.draw(screen=self.screen)
        self.clock.tick(self.fps)
        pygame.display.flip()

        if(self.ball.x < 0):
            self.player_1_score += 1
            player_1_reward += 1
            player_2_reward -= 1
            self.ball.spawn()
        elif(self.ball.x > self.window_width):
            self.player_2_score += 1
            player_2_reward += 1
            player_1_reward -= 1
            self.ball.spawn()


        if(self.player_1_score >= self.top_score or 
           self.player_2_score >= self.top_score):
            if self.player1 == "human":
                self.game_over()
            else:
                done = True
                truncated = True

        observation = self._get_obs() 
        info = {}

        return observation, player_1_reward, done, truncated, info       
        


