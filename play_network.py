
TEST_MODEL_PATH = "/home/nvinden/Work/projects/connect_4/model_weights_cycle_12.pth"

from model import DQNAgent
from game import Connect4GameEnv

import torch

def ask_for_user_action(game : Connect4GameEnv):
    legal_actions = game.get_legal_actions()

    while True:
        action = input("Enter action: ")
        try:
            action = int(action)
        except ValueError:
            print("Invalid action. Please enter an integer.")
            continue

        if action not in legal_actions:
            print("Invalid action. Please enter a legal action.")
            continue

        return action

# Clears the screen and displays the game
def display_game(game : Connect4GameEnv):
    print("\033[H\033[J")
    game.render()

def main():
    #opponent = DQNAgent.load_from_checkpoint(TEST_MODEL_PATH)
    opponent = DQNAgent()
    opponent.q_network = torch.load(TEST_MODEL_PATH)
    opponent.eval()
    opponent.MCTS_iterations = 1000

    game = Connect4GameEnv()
    game.reset()

    display_game(game)


    done = False    
    while not done:
        player = game.current_player
        state = game.state
        
        if player == 2:
            # Player 2 (Neural Network)
            current_state, (action, action_probs) = opponent.select_action(state, player, mode="eval", return_probabilities=True)
        else:
            # Me
            action = ask_for_user_action(game)
        
        next_game, reward, done, info = game.step(action)
        next_state = next_game.state

        display_game(next_game)

        # Print action probabilities
        if player == 2:
            print(f"NN's action:")
            for i, prob in enumerate(action_probs):
                print(f"\t{i}: {prob:.3f}")
        
        game.state = next_state
        game.current_player = next_game.current_player

    

if __name__ == '__main__':
    main()
