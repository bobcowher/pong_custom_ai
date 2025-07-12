from agent import Agent
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation 
import ale_py
import os
import pygame

total_steps = 0
step_repeat = 4

learning_rate = 0.0001
gamma = 0.99

hidden_layer = 128

# print(observation.shape)

os.environ["SDL_VIDEO_WINDOW_POS"] = "50,50"  # Position it on the screen
os.environ["SDL_VIDEO_CENTERED"] = "0"

# Constants
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

original_render = env.render

def custom_render(mode="human"):
    result = original_render(mode)
    if mode == "human":
        screen = pygame.display.get_surface()
        if screen:  # If the window exists, resize it
            pygame.display.set_mode((500, 400), pygame.RESIZABLE)
    return result

env = ResizeObservation(env, (64, 64))

env = GrayscaleObservation(env, keep_dim=True)

env.render = custom_render

agent = Agent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate, step_repeat=step_repeat,
              gamma=gamma)


# Training Phase 1

agent.test() 

