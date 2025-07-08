from re import error
import pygame
import math

class Paddle:

    def __init__(self, x, y, player, width=20, height=120):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.rect = pygame.Rect(x, y, self.width, self.height)
        
        if(player == 1):
            self.paddle_color = (50, 205, 50)
        elif(player == 2):
            self.paddle_color = (255, 180, 50)
        else:
            raise ValueError("Invalid player provided in the Paddle class. Player options are 1 and 2")

    def draw(self, screen):
        pygame.draw.rect(screen, self.paddle_color,(self.rect.x, self.rect.y, self.width, self.height))

