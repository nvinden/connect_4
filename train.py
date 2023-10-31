from pytorch_lightning import Trainer

from game import Connect4GameEnv
from model import DQNAgent, BATCH_SIZE

from copy import deepcopy

import torch
print(torch.cuda.is_available())


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")


TARGET_UPDATE = 100
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01

N_GAMES = 100
N_EPOCHS = 100

env = Connect4GameEnv()
agent = DQNAgent()

for epoch in range(N_EPOCHS):

    for game in range(N_GAMES):
        state = env.reset()
        done = False
        turn_count = 0
        while not done:
            player = env.current_player
            action = agent.select_action(state, player)

            current_state = state
            env_step = env.step(action)
            next_state, reward, done, info = env_step

            if done: # Change the reward for the losing player's last action
                agent.replay_buffer[-1][2] = env.negative_reward

            entry = [deepcopy(current_state), action, reward, deepcopy(next_state), done, player]
            agent.populate_replay_buffer(entry)

            turn_count += 1

    # Update target network occasionally
    if epoch % TARGET_UPDATE == 0:
        agent.update_target()

    trainer = Trainer(max_epochs=int(len(agent.replay_buffer) * 1.5 / BATCH_SIZE), gpus = 1)
    trainer.fit(agent)
    
    # Adjust exploration
    agent.epsilon = max(agent.epsilon * EPSILON_DECAY, EPSILON_MIN)
