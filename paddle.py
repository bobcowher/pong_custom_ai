from re import error
import pygame
import math

class Paddle:

    def __init__(self, x, y, player_color, width=20, height=120):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.speed = 10

        self.paddle_color = player_color


    def draw(self, screen):
        pygame.draw.rect(screen, self.paddle_color,(self.rect.x, self.rect.y, self.width, self.height))


    def move(self, direction):
        if(direction == 0):
            self.y += self.speed
        elif(direction == 1):
            self.y -= self.speed

        self.rect.topleft = (self.x, self.y)

            

