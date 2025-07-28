from pygame.event import clear
import matplotlib.pyplot as plt
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
        self.frames = deque(maxlen=frame_stack)
        self.target_update_interval = target_update_interval
        
        self.debug_fig = None  # <- add this to your Agent init
        self.debug_axes = []
        
        if eval:
            self.model = Model(action_dim=3, hidden_dim=hidden_layer, observation_shape=(frame_stack,84,84), obs_stack=frame_stack).to(self.device)
            self.model.load_the_model()
            return

        self.env = Pong(player1="ai", player2="bot", render_mode="rgbarray")
        self.eval_envs = [Pong(player1="ai", player2="bot", render_mode="rgbarray"),
                          Pong(player1="bot", player2="ai", render_mode="rgbarray")]

        self.gamma = gamma

        obs, info = self.env.reset()

        obs = self.process_observation(obs, clear_stack=True)

        print(f"Player1 1 Obs Shape: {obs.shape}")

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)

        self.model = Model(action_dim=self.env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape, obs_stack=frame_stack).to(self.device)

        self.target_model = Model(action_dim=self.env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape, obs_stack=frame_stack).to(self.device)

        # Initialize target networks with model parameters
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer_1 = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        print(f"Initialized agents on device: {self.device}")


    def init_frame_stack(self, obs):
        """Call once after env.reset().  Pre-fill both deques."""
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)

    def show_stacked_frames(self, obs, flipped_obs, title="Stacked Frames Comparison"):
        """
        Show both regular and flipped observation stacks in one non-blocking persistent window.
        - obs and flipped_obs should be shape (C, 84, 84)
        """
        obs_np = obs.cpu().numpy()
        flipped_np = flipped_obs.cpu().numpy()
        num_frames = obs_np.shape[0]

        if self.debug_fig is None:
            plt.ion()
            self.debug_fig, self.debug_axes = plt.subplots(2, num_frames, figsize=(num_frames * 2.5, 5))
            self.debug_fig.suptitle(title)
        else:
            for row in self.debug_axes:
                for ax in row:
                    ax.clear()

        for i in range(num_frames):
            self.debug_axes[0][i].imshow(obs_np[i], cmap='gray', vmin=0, vmax=255)
            self.debug_axes[0][i].set_title(f"Frame {i}")
            self.debug_axes[0][i].axis('off')

            self.debug_axes[1][i].imshow(flipped_np[i], cmap='gray', vmin=0, vmax=255)
            self.debug_axes[1][i].set_title(f"Flipped {i}")
            self.debug_axes[1][i].axis('off')

        self.debug_fig.tight_layout()
        self.debug_fig.canvas.draw()
        self.debug_fig.canvas.flush_events()

    def save_debug_frame(self, frame_tensor, p1_score, p2_score, episode, step):
        """
        Saves the most recent frame from a stacked observation tensor (C,84,84).
        
        Parameters:
        - frame_tensor: torch.Tensor of shape (C, 84, 84) where C >= 1
        - p1_score, p2_score: ints
        - episode, step: ints for naming
        """
        os.makedirs("debug", exist_ok=True)

        # Use the most recent frame (last in stack)
        if frame_tensor.ndim != 3 or frame_tensor.shape[1:] != (84, 84):
            raise ValueError(f"Expected (C,84,84), got {frame_tensor.shape}")
        
        latest_frame = frame_tensor[-1].cpu().numpy()  # grab last channel (84,84)
        latest_frame = np.clip(latest_frame, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(latest_frame, cv2.COLOR_GRAY2BGR)

        # Add score text
        label = f"P1: {p1_score} | P2: {p2_score}"
        cv2.putText(frame_bgr, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Save image
        filename = f"debug/frame_ep{episode}_step{step}_p1_{p1_score}_p2_{p2_score}.png"
        cv2.imwrite(filename, frame_bgr)


    def get_action(self, obs, player=2):

        if(player == 2):
            obs = self.flip_obs(obs) 
        #
        q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
        action = torch.argmax(q_values, dim=-1).item()

        return action


    def flip_obs(self, obs):
        return torch.flip(obs, dims=[2])

    def process_observation(self, obs, clear_stack=False):
        # obs = torch.tensor(obs, dtype=torch.float32).permute(2,0,1)  
        obs = torch.tensor(obs, dtype=torch.float32)  

        if(len(self.frames) < self.frame_stack):
            self.init_frame_stack(obs)
            
        if(clear_stack):
            self.init_frame_stack(obs)

        self.frames.append(obs)

        obs_stacked = torch.cat(tuple(self.frames), dim=0)

        return obs_stacked


    def eval(self, bot_difficulty="easy"):
        # Player 0 is mapping to player 1 for 0-index purposes.

        for eval_env in self.eval_envs:
            eval_env.bot_difficulty = bot_difficulty
        
        episode_reward = [0, 0]

        for player in range(2):

            obs, info = self.eval_envs[player].reset()

            obs = self.process_observation(obs, clear_stack=True)

            done = False

            episode_reward[player] = 0

            while not done:

                reward = 0

                if(player == 0):
                    action = self.get_action(obs, player=1) 
                    next_obs, reward, _, done, truncated, info = self.eval_envs[player].step(player_1_action=action)
                elif(player == 1):
                    action = self.get_action(obs, player=2) # TODO - Fix this funky naming logic later 
                    next_obs, _, reward, done, truncated, info = self.eval_envs[player].step(player_2_action=action)

                obs = self.process_observation(next_obs)

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

            obs = self.process_observation(obs)

            episode_steps = 0

            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:

                if random.random() < epsilon:
                    player_1_action = self.env.action_space.sample()
                else:
                    player_1_action = self.get_action(obs, player=1)
                   
                if random.random() < epsilon:
                    player_2_action = self.env.action_space.sample()
                else:
                    player_2_action = self.get_action(obs, player=2) 

                player_1_reward = 0
                player_2_reward = 0

                next_obs, player_1_reward, player_2_reward, done, truncated, info = self.env.step(player_1_action=player_1_action)
                # next_obs, player_1_reward, player_2_reward, done, truncated, info = self.env.step(player_2_action=player_2_action)
    
                #if(player_1_reward != 0):
                #    self.save_debug_frame(obs, player_1_reward, player_2_reward, episode, episode_steps)
                
                next_obs = self.process_observation(next_obs)

                self.memory.store_transition(obs, player_1_action, player_1_reward, next_obs, done)
                # self.memory.store_transition(obs, player_2_action, player_2_reward, next_obs, done)
                # self.memory.store_transition(self.flip_obs(obs), player_2_action, player_2_reward, self.flip_obs(next_obs), done)

                obs = next_obs                

                player_1_episode_reward += player_1_reward
                player_2_episode_reward += player_2_reward
                episode_steps += 1
                total_steps += 1


                if self.memory.can_sample(batch_size):
                    # 1 — sample & reshape
                    observations, actions, rewards, next_observations, dones = \
                        self.memory.sample_buffer(batch_size)

                    actions  = actions.unsqueeze(1).long()
                    rewards  = rewards.unsqueeze(1)
                    dones    = dones.unsqueeze(1).float()

                    # 2 — Q(s,a) with the online network
                    q_values      = self.model(observations)
                    q_sa          = q_values.gather(1, actions)

                    # 3 — Double-DQN target  ─────── ★ only changes below ★ ───────
                    with torch.no_grad():                                    # ★ no grads
                        next_actions = torch.argmax(
                            self.model(next_observations), dim=1, keepdim=True
                        )                                                    # ★ a* from online net
                        next_q = self.target_model(next_observations).gather(1, next_actions)        # ★ Q_target(s',a*)

                        targets = rewards + (1 - dones) * self.gamma * next_q

                    # 4 — loss & optimise
                    loss = F.mse_loss(q_sa, targets)
                    writer.add_scalar("Stats/model_loss", loss.item(), total_steps)

                    self.optimizer_1.zero_grad()
                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    writer.add_scalar("Stats/grad_norm_before_clip", total_norm, total_steps)
                    self.optimizer_1.step()

                    # 5 — target-net sync
                    if total_steps % self.target_update_interval == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
            
                    if total_steps % 1000 == 0:
                        _, _, rewards, _, _ = self.memory.sample_buffer(16)
                        print("Sampled Player 1 rewards:", rewards.squeeze().tolist())


            writer.add_scalar('Score/Player 1 Training', player_1_episode_reward, episode)
            writer.add_scalar('Score/Player 2 Training', player_2_episode_reward, episode)

            if episode > 0 and (episode % 20 == 0):

                print("\nEval Run Started")

                for difficulty in ['easy']:
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
