import gym
from gym import spaces

import numpy as np
import random


class Connect4GameEnv(gym.Env):
    def __init__(self):
        super(Connect4GameEnv, self).__init__()
        
        self.action_space = spaces.Discrete(7)  # In Connect 4, typically 7 columns to choose from
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=int)  # 6 rows x 7 columns board, 0: empty, 1: player1, 2: player2

        # Reward vals
        self.positive_reward = 1.0
        self.negative_reward = -1.0
        self.neutral_reward = -0.1

        # Initialize game state
        self.state = None
        self.current_player = None
        self.reset()

        np.set_printoptions(precision=2, suppress=True, linewidth=90)

    def reset(self):
        # Reset game state and return initial observation
        self.state = np.zeros((6, 7), dtype=int)
        self.current_player = random.choice([1, 2])
        return self.state

    def step(self, action):
        # Check if move is legal
        if self.state[0, action] != 0:
            return self.state, self.negative_reward, False, {"reason": "invalid action"}
        
        self.state = self._place_piece(self.state, action, self.current_player)

        winner = self._check_winner(self.state, self.current_player)
        if winner is not None:
            if winner == self.current_player:
                reward = self.positive_reward
            elif winner == 0:
                reward = self.neutral_reward
            else:
                reward = self.negative_reward

            return self.state, reward, True, {"reason": f"reward: {reward}"}
        
        # Switch players
        self.current_player = 1 if self.current_player == 2 else 2
        
        return self.state, 0, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(self.state)

        elif mode == 'rgb_array':
            # Convert game state to an image if required
            pass

    def close(self):
        pass

    def _place_piece(self, state, action, player):
        col_index = np.where(state[:, action] == 0)[0][-1]
        state[col_index, action] = player

        return state
    
    def _check_winner(self, state, player):
        #TODO: add the stack rule thing

        # Check if it is a draw
        if np.all(state != 0):
            return 0

        # Check horizontal locations for win
        for row in range(6):
            for col in range(4):
                if all(state[row, col:col+4] == player):
                    return player

        # Check vertical locations for win
        for col in range(7):
            for row in range(3):
                if all(state[row:row+4, col] == player):
                    return player

        # Check positively sloped diagonals
        for row in range(3):
            for col in range(4):
                if all([state[row+i, col+i] == player for i in range(4)]):
                    return player

        # Check negatively sloped diagonals
        for row in range(3, 6):
            for col in range(4):
                if all([state[row-i, col+i] == player for i in range(4)]):
                    return player

        # No win, return None (or you could check for draw and return 0 if needed)
        return None
