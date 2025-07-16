from agent import Agent
import time
from game import Pong

episodes = 3000
max_episode_steps = 10000
total_steps = 0
step_repeat = 4
max_episode_steps = max_episode_steps / step_repeat

batch_size = 64
learning_rate = 0.0001
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.995
gamma = 0.99

hidden_layer = 128

# print(observation.shape)

# Constants
start_time = time.perf_counter()

env = Pong(player1="ai", player2="bot", render_mode="rgbarray")


summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_mse_loss_bs={batch_size}_double_dqn'

agent = Agent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate, step_repeat=step_repeat,
              gamma=gamma)


# Training Phase 1

agent.train(episodes=episodes, max_episode_steps=max_episode_steps, summary_writer_suffix=summary_writer_suffix + "-phase-1",
            batch_size=batch_size, epsilon=epsilon, epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon)

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f"Elapsed time was: {elapsed_time}")
    

