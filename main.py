from game import Pong
from agent import Agent

agent = Agent(eval=True, hidden_layer=512)

# env = Pong(render_mode="human", player1="human", player2="bot", bot_difficulty="easy")
# env = Pong(render_mode="human", player1="human", player2="bot", bot_difficulty="hard")
# env = Pong(render_mode="human", player1="bot", player2="human", bot_difficulty="easy")
# env = Pong(render_mode="human", player1="bot", player2="human", bot_difficulty="hard")
env = Pong(render_mode="human", player1="ai", player2="human", bot_difficulty="hard", ai_agent=agent)

env.game_loop()


