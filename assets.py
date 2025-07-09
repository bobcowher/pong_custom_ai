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

    
        self.rect.topleft = (self.x, self.y)

            
class Ball:

    def __init__(self, x, y, window_height, player_1_paddle, player_2_paddle, width=10, height=10):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.window_height = window_height
        self.player_1_paddle = player_1_paddle
        self.player_2_paddle = player_2_paddle
        self.rect = pygame.Rect(x, y, width, height)
        self.vx = 3
        self.vy = 10
        self.ball_color = (255, 255, 255)


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
            self.vy = self.vy * -1
#            self.vx = self.vx * -1
            collision = True


        for paddle in [self.player_1_paddle, self.player_2_paddle]:
               if(new_rect.colliderect(paddle)):
                    # self.vy = self.vy * -1
                    self.vx = self.vx * -1
                    collision = True
        
        new_x, new_y, new_rect = self.generate_new_rect()

        self.x = new_x
        self.y = new_y
        self.rect = new_rect

#        self.rect.topleft = (self.x, self.y)
    

    def draw(self, screen):
        pygame.draw.rect(screen, self.ball_color,(self.rect.x, self.rect.y, self.width, self.height))





