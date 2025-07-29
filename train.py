from agent import Agent
import time
from game import Pong

episodes = 10000
max_episode_steps = 10000
total_steps = 0

batch_size = 64 
learning_rate = 0.0001
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.99
gamma = 0.99
max_buffer_size = 400000
target_update_interval = 100

hidden_layer = 512 

# print(observation.shape)

# Constants
start_time = time.perf_counter()

summary_writer_suffix = f'dqn_lr={learning_rate}_bs={batch_size}_buffer={max_buffer_size}_tui={target_update_interval}'

agent = Agent(hidden_layer=hidden_layer,
              learning_rate=learning_rate,
              gamma=gamma,
              max_buffer_size=max_buffer_size,
              target_update_interval=target_update_interval)

# Training Phase 1

agent.train(episodes=episodes, max_episode_steps=max_episode_steps, summary_writer_suffix=summary_writer_suffix + "-phase-1",
            batch_size=batch_size, epsilon=epsilon, epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon)

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f"Elapsed time was: {elapsed_time}")
    

