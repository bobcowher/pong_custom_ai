from agent import Agent
import time
from game import Pong

episodes = 6000
max_episode_steps = 2000
total_steps = 0

batch_size = 32 
learning_rate = 0.0002
epsilon = 1
min_epsilon = 0.05
epsilon_decay = 0.995
gamma = 0.99
max_buffer_size = 200000
target_update_interval = 10000
checkpoint_pool = 5 

hidden_layer = 512 

# print(observation.shape)

# Constants
start_time = time.perf_counter()

summary_writer_suffix = f'dqn_lr={learning_rate}_bs={batch_size}_buffer={max_buffer_size}_tui={target_update_interval}_hl={hidden_layer}'

agent = Agent(hidden_layer=hidden_layer,
              learning_rate=learning_rate,
              gamma=gamma,
              max_buffer_size=max_buffer_size,
              target_update_interval=target_update_interval,
              max_episode_steps=max_episode_steps,
              epsilon=epsilon,
              min_epsilon=min_epsilon,
              epsilon_decay=epsilon_decay,
              checkpoint_pool=checkpoint_pool)

# Training Phase 1

agent.train(episodes=episodes, summary_writer_suffix=summary_writer_suffix + "-phase-1",
            batch_size=batch_size)

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f"Elapsed time was: {elapsed_time}")
    

