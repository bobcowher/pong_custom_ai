from re import error
import pygame
import math

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

        # Direction 0 is down, 1 is up. 

        if(direction == 0):
            new_y = self.y + self.speed
        elif(direction == 1):
            new_y = self.y - self.speed

        

        if(0 <= new_y <= (self.window_height - self.height)):
            self.y = new_y

    
        print(self.y)

        self.rect.topleft = (self.x, self.y)

            

