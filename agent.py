from pygame.event import clear
from buffer import ReplayBuffer
from model import Model, soft_update, hard_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import pygame
import cv2
import numpy as np
from game import Pong
from collections import deque

class Agent():

    def __init__(self, hidden_layer=512, learning_rate=0.0001, gamma=0.99, max_buffer_size=100000, eval=False, frame_stack=3, target_update_interval=10000) -> None:

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.frame_stack = frame_stack
        self.frames_player_1 = deque(maxlen=frame_stack)
        self.frames_player_2 = deque(maxlen=frame_stack)
        self.target_update_interval = target_update_interval
        
        if eval:
            self.model = Model(action_dim=3, hidden_dim=hidden_layer, observation_shape=(frame_stack,84,84), obs_stack=frame_stack).to(self.device)
            self.model.load_the_model()
            return

        self.env = Pong(player1="ai", player2="ai", render_mode="rgbarray")
        self.eval_envs = [Pong(player1="ai", player2="bot", render_mode="rgbarray"),
                          Pong(player1="bot", player2="ai", render_mode="rgbarray")]

        self.gamma = gamma

        obs, info = self.env.reset()

        player_1_obs, player_2_obs = self.process_observation(obs, clear_stack=True)

        print(f"Player1 1 Obs Shape: {player_1_obs.shape}")

        self.player_1_memory = ReplayBuffer(max_size=max_buffer_size, input_shape=player_1_obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)
        self.player_2_memory = ReplayBuffer(max_size=max_buffer_size, input_shape=player_2_obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)

        self.model = Model(action_dim=self.env.action_space.n, hidden_dim=hidden_layer, observation_shape=player_1_obs.shape, obs_stack=frame_stack).to(self.device)

        self.target_model = Model(action_dim=self.env.action_space.n, hidden_dim=hidden_layer, observation_shape=player_1_obs.shape, obs_stack=frame_stack).to(self.device)

        # Initialize target networks with model parameters
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer_1 = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate

        print(f"Initialized agents on device: {self.device}")


    def init_frame_stack(self, player_1_obs, player_2_obs):
        """Call once after env.reset().  Pre-fill both deques."""
        self.frames_player_1.clear()
        self.frames_player_2.clear()
        for _ in range(self.frame_stack):
            self.frames_player_1.append(player_1_obs)           # original
            self.frames_player_2.append(player_2_obs)


    def get_action(self, obs, model=None):
        if model == None:
            model = self.model

        q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
        action = torch.argmax(q_values, dim=-1).item()

        return action


    def process_observation(self, obs, clear_stack=False):
        # obs = torch.tensor(obs, dtype=torch.float32).permute(2,0,1)  
        player_1_obs = torch.tensor(obs, dtype=torch.float32)  
        player_2_obs = torch.flip(player_1_obs, dims=[2])

        if(len(self.frames_player_1) < self.frame_stack):
            self.init_frame_stack(player_1_obs, player_2_obs)

            print(f"Clearing stack due to player 1 frames({self.frames_player_1}) under self.framestack {self.frame_stack}")
            
        if(clear_stack):
            self.init_frame_stack(player_1_obs, player_2_obs)

        self.frames_player_1.append(player_1_obs)
        self.frames_player_2.append(player_2_obs)

        player_1_obs_stacked = torch.cat(tuple(self.frames_player_1), dim=0)     # (k,84,84)
        player_2_obs_stacked = torch.cat(tuple(self.frames_player_2), dim=0)

        return player_1_obs_stacked, player_2_obs_stacked


    def eval(self, bot_difficulty="easy"):
        # Player 0 is mapping to player 1 for 0-index purposes.

        for eval_env in self.eval_envs:
            eval_env.bot_difficulty = bot_difficulty
        
        episode_reward = [0, 0]

        for player in range(2):

            obs, info = self.eval_envs[player].reset()

            player_1_obs, player_2_obs = self.process_observation(obs, clear_stack=True)

            done = False

            episode_reward[player] = 0

            while not done:

                reward = 0

                if(player == 0):
                    action = self.get_action(player_1_obs) 
                    next_obs, reward, _, done, truncated, info = self.eval_envs[player].step(player_1_action=action)
                elif(player == 1):
                    action = self.get_action(player_2_obs) 
                    next_obs, _, reward, done, truncated, info = self.eval_envs[player].step(player_2_action=action)

                player_1_obs, player_2_obs = self.process_observation(next_obs)

                episode_reward[player] += reward

        return episode_reward[0], episode_reward[1]
                

    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        if not os.path.exists('models'):
            os.makedirs('models')

        total_steps = 0

        for episode in range(episodes):

            done = False
            player_1_episode_reward = 0
            player_2_episode_reward = 0
            obs, info = self.env.reset()

            player_1_obs, player_2_obs = self.process_observation(obs)

            episode_steps = 0

            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:

                if random.random() < epsilon:
                    player_1_action = self.env.action_space.sample()
                else:
                    player_1_action = self.get_action(player_1_obs)
                   
                if random.random() < epsilon:
                    player_2_action = self.env.action_space.sample()
                else:
                    player_2_action = self.get_action(player_2_obs) 

                player_1_reward = 0
                player_2_reward = 0

                next_obs, player_1_reward, player_2_reward, done, truncated, info = self.env.step(player_1_action=player_1_action, player_2_action=player_2_action)

                player_1_next_obs, player_2_next_obs = self.process_observation(next_obs)

                self.player_1_memory.store_transition(player_1_obs, player_1_action, player_1_reward, player_1_next_obs, done)
                self.player_2_memory.store_transition(player_2_obs, player_2_action, player_2_reward, player_2_next_obs, done)

                player_1_obs = player_1_next_obs
                player_2_obs = player_2_next_obs

                player_1_episode_reward += player_1_reward
                player_2_episode_reward += player_2_reward
                episode_steps += 1
                total_steps += 1

                if self.player_1_memory.can_sample(batch_size // 2) and self.player_2_memory.can_sample(batch_size // 2):
                    # Sample from both buffers equally
                    obs1, act1, rew1, next_obs1, done1 = self.player_1_memory.sample_buffer(batch_size // 2)
                    obs2, act2, rew2, next_obs2, done2 = self.player_2_memory.sample_buffer(batch_size // 2)
                    
                    # Concatenate samples from both players
                    observations = torch.cat([obs1, obs2], dim=0)
                    actions = torch.cat([act1, act2], dim=0)
                    rewards = torch.cat([rew1, rew2], dim=0)
                    next_observations = torch.cat([next_obs1, next_obs2], dim=0)
                    dones = torch.cat([done1, done2], dim=0)

                    dones = dones.unsqueeze(1).float()

                    # Current Q-values from both models
                    q_values = self.model(observations)
                    actions = actions.unsqueeze(1).long()
                    qsa_batch = q_values.gather(1, actions)

                    # Action selection using the main models
                    next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)

                    # Q-value evaluation using the target models
                    next_q_values = self.target_model(next_observations).gather(1, next_actions)

                    # Compute the target using Double DQN with minimization
                    target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

                    # Calculate the loss for both models
                    loss = F.mse_loss(qsa_batch, target_b.detach())

                    writer.add_scalar("Loss/model", loss.item(), total_steps)

                    # Backpropagation and optimization step for both models
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer_1.step()

                    # Update the target models periodically
                    if total_steps % self.target_update_interval == 0:
                        self.target_model.load_state_dict(self.model.state_dict())

            writer.add_scalar('Score/Player 1 Training', player_1_episode_reward, episode)
            writer.add_scalar('Score/Player 2 Training', player_2_episode_reward, episode)


            if episode > 0 and (episode % 100 == 0):

                print("\nEval Run Started")

                for difficulty in ['easy', 'hard']:
                    player_1_score_v_bot, player_2_score_v_bot = self.eval(bot_difficulty=difficulty)
                    writer.add_scalar(f'Score/Player 1 v. {difficulty} Bot', player_1_score_v_bot, episode)
                    writer.add_scalar(f'Score/Player 2 v. {difficulty} Bot', player_2_score_v_bot, episode)

                    print(f"Player 1 v. {difficulty} Bot: {player_1_score_v_bot}")
                    print(f"Player 2 v. {difficulty} Bot: {player_2_score_v_bot}")
                
                print("Eval Run Finished. Saving the model...\n")
                self.model.save_the_model()
                print("Model Saved")


            writer.add_scalar('Epsilon', epsilon, episode)

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

            episode_time = time.time() - episode_start_time

            print(f"Completed episode {episode} with Player 1 score {player_1_episode_reward}")
            print(f"Completed episode {episode} with Player 2 score {player_2_episode_reward}")
            print(f"Episode Time: {episode_time:1f} seconds")
            print(f"Episode Steps: {episode_steps}")
