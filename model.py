import pytorch_lightning as pl
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import numpy as np
import random

from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 32
GAMMA = 0.99


class DummyDataset(Dataset):
    def __len__(self):
        return 1000  # or whatever you feel like

    def __getitem__(self, idx):
        return {}  # dummy data


class DQN(nn.Module):
    def __init__(self, input_dim=85, output_dim=7, hidden_dim=128, dropout_prob=0.3):
        super(DQN, self).__init__()

        # Define the network architecture
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            # Second hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            # Third hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            # Fourth hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            # Fifth hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            # Output layer
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, player):
        if isinstance(player, int):
            player = torch.FloatTensor([player])
        
        if len(player.shape) == 1:
            player = player.unsqueeze(1)

        p1_board = (x == 1).float()
        p2_board = (x == 2).float()

        p1_board_state = torch.FloatTensor(p1_board).view(x.shape[0], -1)
        p2_board_state = torch.FloatTensor(p2_board).view(x.shape[0], -1)

        player_state = torch.FloatTensor(player) * 2 - 3

        x = torch.cat((p1_board_state, p2_board_state, player_state), dim=1)
        q_vals = self.layers(x)

        return q_vals


class DQNAgent(pl.LightningModule):
    def __init__(self, replay_buffer_size=10000):
        super(DQNAgent, self).__init__()
        self.q_network = DQN()
        self.target_network = DQN()
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.epsilon = 1.0  # For epsilon-greedy policy

    def forward(self, x, player):
        # Concatenates the board state and the current player
        return self.q_network(x, player)

    def select_action(self, state, player):
        # REMOVE
        self.epsilon = 0.01

        illegal_actions = list(np.where(state[0] != 0)[0])

        if np.random.rand() < self.epsilon:
            return random.sample([i for i in range(7) if i not in illegal_actions], 1)[0]
        
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            self.q_network.eval()
            q_action = self(state, player)

            # Remove illegal actions
            q_action[:, illegal_actions] = -1000

            action_chosen = q_action.argmax(dim=1).item()
            self.q_network.train()
        
        return action_chosen

    def training_step(self, batch, batch_idx):

        # Getting data from experience replay buffer
        experience_batch = random.sample(self.replay_buffer, min(BATCH_SIZE, len(self.replay_buffer)))
        states, actions, rewards, next_states, dones, players = zip(*experience_batch)

        # Convert to tensors
        states = torch.stack([torch.tensor(state) for state in states]).float()  # Convert list of tensors to a single tensor
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack([torch.tensor(state) for state in next_states]).float()  # Convert list of tensors to a single tensor
        dones = torch.tensor(dones, dtype=torch.float32)
        players = torch.tensor(players, dtype=torch.float32)


        # Get current Q-values from the Q-network
        curr_Q = self.q_network(states, players).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Get next Q-values from the target Q-network
        next_player = players * -1 + 3
        next_Q = self.target_network(next_states, next_player).max(1)[0]

        # Compute the target Q-values
        target_Q = rewards + (1 - dones) * GAMMA * next_Q

        # Compute loss
        loss = F.mse_loss(curr_Q, target_Q)

        # Logging (optional, for using with TensorBoard or other logging tools)
        self.log('train_loss', loss, prog_bar=True)

        # In Lightning, you can directly return the loss as it will be used for backpropagation
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.q_network.parameters())
    
    def populate_replay_buffer(self, experience):
        # Add new experience to the buffer; oldest experience is automatically removed if buffer exceeds its capacity
        self.replay_buffer.append(experience)

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Inside your LightningModule (Agent class)
    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=BATCH_SIZE)