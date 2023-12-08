import pytorch_lightning as pl
import torch.optim as optim

from game import Connect4GameEnv
from config import CONFIG
from play import battle

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import numpy as np
import random
import math
import copy

from torch.utils.data import DataLoader, Dataset
import wandb
import dask


class ReplayBufferDataset(Dataset):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        item = self.replay_buffer[idx]
        return item

class DQN(nn.Module):
    def __init__(self, input_dim=85, output_dim=8, hidden_dim=256, dropout_prob=0.3):
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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, player):
        if isinstance(player, int):
            player = torch.tensor([player], dtype=torch.float32, device=CONFIG['DEVICE'])
        elif isinstance(player, torch.Tensor):
            player = player.float().to(CONFIG['DEVICE'])
        elif isinstance(player, np.ndarray) or isinstance(player, list):
            player = torch.tensor(player, dtype=torch.float32, device=CONFIG['DEVICE'])

        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32, device=CONFIG['DEVICE'])
        elif isinstance(x, torch.Tensor):
            x = x.to(CONFIG['DEVICE'])
            
        if len(player.shape) == 1:
            player = player.unsqueeze(1)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        p1_board = (x == 1).float()
        p2_board = (x == 2).float()

        # No need to re-create the tensor, just reshape and ensure it's on the right device
        p1_board_state = p1_board.view(x.shape[0], -1).to(CONFIG['DEVICE'])
        p2_board_state = p2_board.view(x.shape[0], -1).to(CONFIG['DEVICE'])

        # You already ensured 'player' tensor is in float and on the right device
        player_state = player * 2 - 3

        x = torch.cat((p1_board_state, p2_board_state, player_state), dim=1)
        out = self.layers(x)

        q_vals = out[:, :7]
        v = F.tanh(out[:, 7])

        return q_vals, v


class DQNPyCNN(nn.Module):
    def __init__(self, alpha, num_blocks, num_classes=7):
        super(DQNPyCNN, self).__init__()
        self.in_channels = 16

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Pyramid Blocks
        self.layers = self._make_layers(alpha, num_blocks)

        # Fully connected layer for classification
        self.fc = nn.Linear(self.in_channels, num_classes + 1)

    def _make_layers(self, alpha, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(self.in_channels, self.in_channels + alpha, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(self.in_channels + alpha))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels += alpha
        return nn.Sequential(*layers)
    
    def create_input(self, x, player):
        if isinstance(player, int):
            player = torch.tensor([player], dtype=torch.float32, device=CONFIG['DEVICE'])
        elif isinstance(player, torch.Tensor):
            player = player.float().to(CONFIG['DEVICE'])
        elif isinstance(player, np.ndarray) or isinstance(player, list):
            player = torch.tensor(player, dtype=torch.float32, device=CONFIG['DEVICE'])

        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32, device=CONFIG['DEVICE'])
        elif isinstance(x, torch.Tensor):
            x = x.to(CONFIG['DEVICE'])
            
        if len(player.shape) == 1:
            player = player.unsqueeze(1)

        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # Turning the board into a 3-channel tensor
        x_input = torch.zeros((x.shape[0], 3, 6, 7), dtype=torch.float32, device=CONFIG['DEVICE'])
        for batch_number in range(x.shape[0]):
            x_input[batch_number] = self.transform_board(x[batch_number], player[batch_number])

        return x_input
    
    def transform_board(self, x, player):
        # Ensure player tensor has the same dtype and device as x
        player = int(player.item())
        other_player = 1 if player == 2 else 2

        p1_board = (x == player).float()
        p2_board = (x == other_player).float()
        empty_spaces = (x == 0).float()

        # Stack them together to get the new representation
        new_representation = torch.stack((p1_board, p2_board, empty_spaces), dim=0)

        return new_representation


    def forward(self, state, player):
        x = self.create_input(state, player)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layers(x)
        x = x.mean([2, 3])  # Global Average Pooling (GAP)
        out = self.fc(x)

        q_vals = out[:, :7]
        v = F.tanh(out[:, 7])

        return q_vals, v

class MCTSNode:
    def __init__(self, state : Connect4GameEnv, parent=None, prior=1.0):
        self.state : Connect4GameEnv = state  # the game state this node represents
        self.parent = parent  # parent Node
        self.children = {}  # dict of child Nodes
        self.visits = 0  # number of times this node has been visited
        self.total_value = 0  # cumulative value of this node
        self.prior = prior  # prior probability computed by the policy network
        
    def add_child(self, child, action):
        child.parent = self
        self.children[action] = child
        
    def update(self, reward):
        self.visits += 1
        self.value += reward
        
    def fully_expanded(self):
        n_legal_actions = len(self.state.get_legal_actions())
        return len(self.children) == n_legal_actions
    
    def is_terminal(self):
        # Is leaf if there is a winner
        game_ended = self.state.game_ended()
        return game_ended
        
    def best_child(self, c_param):
        choices_weights = [
            (child.value / (child.visits + 1e-7) +
             c_param * (2 * np.log(self.visits + 1) / (child.visits + 1e-7)) ** 0.5)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class DQNAgent(pl.LightningModule):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.q_network = DQNPyCNN(alpha=8, num_blocks=4)
        self.q_network = self.q_network.to(CONFIG['DEVICE'])

        self.replay_buffer = deque(maxlen=CONFIG['ER_SIZE'])
        self.train_c = math.sqrt(2)  # For epsilon-greedy policy
        self.eval_c = 0.05

        self.train_losses = []
        self.v_losses = []
        self.pi_losses = []

        self.MCTS_iterations = CONFIG['MCTS_ITERATIONS']

        self.cycle_count = 0

    def forward(self, x, player):
        # Concatenates the board state and the current player
        self.q_network.to(CONFIG['DEVICE'])
        return self.q_network(x, player)

    # Uses MCTS to select an action
    def select_action(self, state, player, mode = "train", return_probabilities = False):
        state = copy.deepcopy(state)

        if mode == "train":
            c = self.train_c
        elif mode == "eval":
            c = self.eval_c
        else:
            raise ValueError("Invalid mode")

        action_chosen = self.MCTS_get_action(state, player, c, return_probabilites=return_probabilities)
        return state, action_chosen

    def training_step(self, batch, batch_idx):
        # Getting data from experience replay buffer
        states, actions, action_probabilities, outcomes, next_states, dones, players = batch

        # Convert to tensors
        states = torch.stack([state.clone().detach() for state in states]).float()
        actions = actions.clone().detach()
        action_probabilities = action_probabilities.clone().detach()
        outcomes = outcomes.clone().detach().float()
        next_states = torch.stack([state.clone().detach() for state in next_states]).float()
        dones = dones.clone().detach().int()
        players = players.clone().detach()

        # Get current Q-values from the Q-network
        self.q_network.to(CONFIG['DEVICE'])
        self.q_network.train()

        pred_Q, pred_v = self.q_network(states, players)

        pred_v = pred_v.unsqueeze(-1)

        """
        policy_logits: Output from the neural network representing the move probabilities (logits). Shape: [batch_size, num_actions]
        value_output: Output from the neural network representing the board value prediction. Shape: [batch_size, 1]
        mcts_probs: Move probabilities from MCTS. Shape: [batch_size, num_actions]
        z: Actual outcome of the game. Shape: [batch_size, 1]
        """

        # Policy loss (cross entropy loss)
        policy_loss = -torch.sum(action_probabilities * F.log_softmax(pred_Q, dim=1)) / pred_Q.size(0)

        # Value loss (mean squared error loss)
        value_loss = F.mse_loss(pred_v.view(-1), outcomes.view(-1))

        # Total loss (You can optionally add coefficients to weight these losses)
        total_loss = policy_loss + value_loss

        # Logging (optional, for using with TensorBoard or other logging tools)
        self.train_losses.append(total_loss.item())
        self.v_losses.append(value_loss.item())
        self.pi_losses.append(policy_loss.item())

        # In Lightning, you can directly return the loss as it will be used for backpropagation
        return total_loss

    def configure_optimizers(self):
        return optim.Adam(self.q_network.parameters(), lr = CONFIG["LEARNING_RATE"])
    
    def populate_replay_buffer(self, experience):
        # Add new experience to the buffer; oldest experience is automatically removed if buffer exceeds its capacity
        if isinstance(experience, list) and len(experience) > 0 and isinstance(experience[0], list):
            for exp in experience:
                self.replay_buffer.append(exp)
        elif isinstance(experience, list):
            self.replay_buffer.append(experience)
        else:
            raise ValueError("Invalid experience type")

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.to(CONFIG['DEVICE'])
        self.q_network.to(CONFIG['DEVICE'])

    # Inside your LightningModule (Agent class)
    def train_dataloader(self):
        dataset = ReplayBufferDataset(self.replay_buffer)
        return DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, drop_last=True, num_workers=23)
    
    def on_train_epoch_end(self, *arg, **kwargs):
        # Compute average training loss
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        avg_v_loss = sum(self.v_losses) / len(self.v_losses)
        avg_pi_loss = sum(self.pi_losses) / len(self.pi_losses)


        # Compute the winning vs random agent
        if CONFIG["USE_MULTIPROCESSING"]:
            n_battles_per_worker = CONFIG["N_RANDOM_BATTLES"] // CONFIG["N_WORKERS"]
            delayed_battles = [dask.delayed(battle)(self, opponent="random", n_battles=n_battles_per_worker) for _ in range(CONFIG["N_WORKERS"])]
            battle_results = dask.compute(*delayed_battles)
            
            # Aggregating the results to compute the overall winning percentage
            random_win_percentage = sum(battle_results) / len(battle_results)
        else:
            random_win_percentage = battle(self, opponent="random", n_battles=100)

        # Compute the winning percentage MTCS vs random
        #MCTS_win_percentage = self.battle(opponenet = "MCTS", n_games = 100)

        # Log using wandb
        if wandb.run is not None:
            log_dict = {
                'cycle': self.cycle_count, 
                'avg_train_loss': avg_train_loss,
                'avg_v_loss': avg_v_loss,
                'avg_pi_loss': avg_pi_loss,
                'random_win_percentage': random_win_percentage
            }
            
            wandb.log(log_dict)

        # Clear the train_losses list for the next epoch
        self.train_losses = []
        self.v_losses = []
        self.pi_losses = []

        self.cycle_count += 1

    # MCTS Functions
    def MCTS_get_action(self, root_board, root_player, c, return_probabilites = False):
        state = Connect4GameEnv(root_board, root_player)
        root = MCTSNode(state=state)

        for _ in range(self.MCTS_iterations):
            leaf = self.transverse_to_leaf(root, c)
            _, value = self.expand_and_evaluate(leaf, c)
            self.backpropagate(leaf, value)

        best_action = self.get_best_action(root.children, return_probabilities = return_probabilites)

        return best_action

    def transverse_to_leaf(self, node : MCTSNode, c : float):
        while not (node.is_terminal() or not node.fully_expanded()):
            node = self.best_uct(node, c)

        return node

    def expand_and_evaluate(self, node : MCTSNode, c : float):
        if node.is_terminal():
            leaf_reward = - node.state.get_reward()
            return node, leaf_reward

        self.q_network.eval()
        self.q_network.to(CONFIG['DEVICE'])
        q_value, value_est = self.q_network(node.state.state, node.state.current_player)

        legal_actions = node.state.get_legal_actions()
        q_val_probs = F.softmax(q_value[0, legal_actions], dim = 0).detach().cpu().numpy()

        # Calculates all values and adds them as children
        for action, probability in zip(legal_actions, q_val_probs):
            new_game, _, _, _ = node.state.step(action)
            new_node = MCTSNode(new_game, prior = probability.item())
            node.add_child(new_node, action)

        return node, value_est.item()

    def best_uct(self, node : MCTSNode, c : float):
        # UCT formula: Q + c * P * sqrt(sum(N)) / (1 + N)
        total_visits = sum([child.visits for child in node.children.values()])
        best_score = -math.inf
        best_child = None

        for child in node.children.values():
            uct_value = child.total_value / (child.visits + 1e-7) + c * child.prior * math.sqrt(total_visits) / (1 + child.visits)
            if uct_value > best_score:
                best_score = uct_value
                best_child = child

        return best_child

    def get_best_action(self, children_dict : dict, return_probabilities = False):
        max_val = max([child.visits for child in children_dict.values()])

        # Get all keys with that value
        max_keys = [key for key, val in children_dict.items() if val.visits == max_val]

        if return_probabilities:
            # Get the probabilities of each action
            root_visits = sum([child.visits for child in children_dict.values()])
            probabilities = {action : child.visits / root_visits for action, child in children_dict.items()}
            probabilities_list = np.array([probabilities[key] if key in children_dict.keys() else 0.0 for key in range(7)])
            return random.choice(max_keys), probabilities_list

        return random.choice(max_keys)


    def backpropagate(self, node : MCTSNode, value):
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

        
