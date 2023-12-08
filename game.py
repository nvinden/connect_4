import gym
from gym import spaces

import numpy as np
import random
import copy

class Connect4GameEnv(gym.Env):
    def __init__(self, state = None, current_player = None):
        super(Connect4GameEnv, self).__init__()
        
        self.action_space = spaces.Discrete(7)  # In Connect 4, typically 7 columns to choose from
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=int)  # 6 rows x 7 columns board, 0: empty, 1: player1, 2: player2

        # Reward vals
        self.positive_reward = 1.0
        self.negative_reward = -1.0
        self.neutral_reward = -0.1

        # Initialize game state
        self.state = state
        self.current_player = current_player
        if state is None or current_player is None:
            self.reset()

        np.set_printoptions(precision=2, suppress=True, linewidth=90)

    def reset(self):
        # Reset game state and return initial observation
        self.state = np.zeros((6, 7), dtype=int)
        self.current_player = random.choice([1, 2])
        return self.state

    def step(self, action):
        # Check if move is legal
        state = self.state.copy()

        if self.state[0, action] != 0:
            return state, self.negative_reward, False, {"reason": "invalid action"}
        
        state = self._place_piece(state, action, self.current_player)

        next_player = 1 if self.current_player == 2 else 2

        winner = self.check_winner(state, self.current_player)
        if winner is not None:
            if winner == self.current_player:
                reward = self.positive_reward
            elif winner == 0:
                reward = self.neutral_reward
            else:
                reward = self.negative_reward

            game = Connect4GameEnv(state, next_player)
            return game, reward, True, {"reason": f"reward: {reward}"}
        
        game = Connect4GameEnv(state, next_player)
        return game, 0, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            for row in range(6):
                print("|", end="")
                for col in range(7):
                    if self.state[row, col] == 0:
                        print(" ", end="|")
                    elif self.state[row, col] == 1:
                        print("X", end="|")
                    else:
                        print("O", end="|")
                print()

            for col in range(7):
                print(f" {col}", end="")

            print()

        elif mode == 'rgb_array':
            # Convert game state to an image if required
            pass

    def close(self):
        pass

    def _place_piece(self, state, action, player):
        col_index = np.where(state[:, action] == 0)[0][-1]
        state[col_index, action] = player

        return state
    
    def game_ended(self):
        game_winner = self.check_winner(self.state, self.current_player)
        return game_winner is not None
    
    def get_legal_actions(self):
        if not isinstance(self.state, np.ndarray):
            self.state = self.state.state

        legal_moves = []
        for i in range(7):
            if self.state[0, i] == 0:
                legal_moves.append(i)
        return legal_moves
    
    def check_winner(self, state, player):
        #TODO: add the stack rule thing
        if isinstance(state, Connect4GameEnv):
            state = state.state

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
    
    def get_reward(self):
        winner = self.check_winner(self.state, self.current_player)
        if winner is not None:
            if winner == self.current_player:
                return self.positive_reward
            elif winner == 0:
                return self.neutral_reward
            else:
                return self.negative_reward
        else:
            return 0
        
    def backpropagate_game_results(self, game_history):
        game_history = copy.deepcopy(game_history)

        last_play = game_history[-1]

        winner = self.check_winner(last_play[4], last_play[6])

        if winner is None:
            raise ValueError("Game has not ended yet")
        
        if winner == 0:
            return game_history
        
        if winner == 1:
            p1_outcome = 1.0
            p2_outcome = -1.0

        if winner == 2:
            p1_outcome = -1.0
            p2_outcome = 1.0

        if winner == 0:
            p1_outcome = -0.1
            p2_outcome = -0.1

        for i, entry in enumerate(game_history):
            if entry[6] == 1:
                entry[3] = p1_outcome
            else:
                entry[3] = p2_outcome

        return game_history

