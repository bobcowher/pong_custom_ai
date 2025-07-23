from random import random
from re import error
import pygame
import math
import numpy as np
import random

class Paddle:

    def __init__(self, x, y, player_color, window_height, width=20, height=120):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.speed = 10

        self.paddle_color = player_color

        self.window_height = window_height


    def draw(self, screen):
        pygame.draw.rect(screen, self.paddle_color,(self.rect.x, self.rect.y, self.width, self.height))


    def move(self, direction):
        assert 0 <= direction <= 3
        # Direction 0 is down, 1 is up. 

        if(direction == 0):
            return
        elif(direction == 1):
            new_y = self.y - self.speed
        elif(direction == 2):
            new_y = self.y + self.speed

        if(0 <= new_y <= (self.window_height - self.height)):
            self.y = new_y

    
        self.rect.topleft = (self.x, self.y)

            
class Ball:

    def __init__(self, window_height, window_width, player_1_paddle, player_2_paddle, width=10, height=10):
        self.height = height
        self.width = width
        self.window_height = window_height
        self.window_width = window_width
        self.player_1_paddle = player_1_paddle
        self.player_2_paddle = player_2_paddle
        self.ball_color = (255, 255, 255)
        self.last_serve_left = False

        self.spawn()


    def spawn(self):
        self.x = self.window_width / 2
        self.y = self.window_height / 2

        speed = random.choice([8, 10, 12])

        # Alternate direction
        self.last_serve_left = not self.last_serve_left
        self.vx = -speed if self.last_serve_left else speed

        self.vy = random.choice([-6, -4, -2, 2, 4, 6])
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)


    def generate_new_rect(self):
        new_y = self.y + self.vy
        new_x = self.x + self.vx
    
        new_rect = pygame.Rect(new_x, new_y, self.width, self.height)

        return new_x, new_y, new_rect


    def move(self):
        
        collision = False

        # First pass at new_x and new_y
        new_x, new_y, new_rect = self.generate_new_rect()

        if(not (0 <= new_y <= (self.window_height - self.height))):
            self.vy = np.clip(self.vy * -1, -20, 20)


        for paddle in [self.player_1_paddle, self.player_2_paddle]:
               if(new_rect.colliderect(paddle)):
                    self.vx = (self.vx * 1.1) * -1 # Invert direction and speed up the ball slightly
        
        new_x, new_y, new_rect = self.generate_new_rect()

        self.x = new_x
        self.y = new_y
        self.rect = new_rect

#        self.rect.topleft = (self.x, self.y)
    

    def draw(self, screen):
        pygame.draw.rect(screen, self.ball_color,(self.rect.x, self.rect.y, self.width, self.height))








