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

class Agent():

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma) -> None:

        self.env = env

        self.step_repeat = step_repeat

        self.gamma = gamma

        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.memory = ReplayBuffer(max_size=500000, input_shape=obs.shape, n_actions=env.action_space.n, device=self.device)

        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)

        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)

        # Initialize target networks with model parameters
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer_1 = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate

        print(f"Initialized agents on device: {self.device}")


    def process_observation(self, obs):
        # obs = torch.tensor(obs, dtype=torch.float32).permute(2,0,1)  
        obs = torch.tensor(obs, dtype=torch.float32)  
        return obs


    def test(self):

        self.model.load_the_model()

        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        done = False

        episode_reward = 0

        while not done:

            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                action = torch.argmax(q_values, dim=-1).item()

            reward = 0

            for i in range(self.step_repeat):
                reward_temp = 0
                next_obs, reward_temp, done, truncated, info = self.env.step(player_1_action=action)

                reward += reward_temp

                # Try rendering as RGB array
                frame = self.env.env.env.render()

                # Resize the frame to fit the desired window size
                resized_frame = cv2.resize(frame, (500, 400))

                # Convert to BGR (OpenCV expects BGR, Gym outputs RGB)
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

                # Show frame with OpenCV
                cv2.imshow("Pong AI", resized_frame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.05)
                
                if done:
                    break

            obs = self.process_observation(next_obs)

            episode_reward += reward


    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        if not os.path.exists('models'):
            os.makedirs('models')

        total_steps = 0

        for episode in range(episodes):

            done = False
            episode_reward = 0
            obs, info = self.env.reset()
            obs = self.process_observation(obs)


            episode_steps = 0

            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:

                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                    action = torch.argmax(q_values, dim=-1).item()

                reward = 0

                for i in range(self.step_repeat):
                    reward_temp = 0
                    next_obs, reward_temp, done, truncated, info = self.env.step(player_1_action=action)

                    reward += reward_temp

                    if(done):
                        break

                next_obs = self.process_observation(next_obs)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs        

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if self.memory.can_sample(batch_size):
                    observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

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
                    if episode_steps % 4 == 0:
                        soft_update(self.target_model, self.model)

            self.model.save_the_model()

            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

            episode_time = time.time() - episode_start_time

            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:1f} seconds")
            print(f"Episode Steps: {episode_steps}")
