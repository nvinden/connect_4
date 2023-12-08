from game import Connect4GameEnv

import copy
import random
import dask

# Playing a game
def play_game_return_ER(agent):
    game_experience = []

    env = Connect4GameEnv()  # Create a new environment for each process
    state = env.reset()
    done = False

    while not done:
        player = env.current_player
        state = env.state

        current_state, (action, MCTS_probabilities) = agent.select_action(state, player, mode="train", return_probabilities=True)

        next_game, reward, done, info = env.step(action)
        next_state = next_game.state

        entry = [copy.deepcopy(current_state), action, MCTS_probabilities, 0, copy.deepcopy(next_state), done, player]
        game_experience.append(entry)
            
        env.state = next_state
        env.current_player = next_game.current_player

    game_experience = env.backpropagate_game_results(game_experience)
    
    return game_experience

def play_games_return_ER(agent, n_games : int):
    game_experience = []

    for game in range(n_games):
        game_experience += play_game_return_ER(agent)
    
    return game_experience

def battle(agent, n_battles : int, opponent : str):
    assert opponent in ["random", "minimax"], "Invalid opponent"

    env = Connect4GameEnv()

    agent.eval()

    wins = 0
    draws = 0

    for game in range(n_battles):
        state = env.reset()
        done = False
        
        while not done:
            player = env.current_player
            state = env.state

            #print("\033[H\033[J")
            #env.render()
            
            if player == 1:
                # Player 1 (Neural Network)
                current_state, (action, _) = agent.select_action(state, player, mode="eval", return_probabilities=True)
            else:
                # Opponent (Random Actions)
                action = random.choice(env.get_legal_actions())
            
            next_game, reward, done, info = env.step(action)
            next_state = next_game.state
            
            env.state = next_state
            env.current_player = next_game.current_player
        
        #print("\033[H\033[J")
        #env.render()

        # Optional: Process game results
        winner = env.check_winner(env.state, player = 1)
        if winner == 1: # Player 1 wins
            wins += 1
        elif winner == 0: # If game is not a draw
            draws += 1
    

    # Optional: Return game results, statistics, etc.
    if n_battles == draws:
        return 0.5

    return wins / (n_battles - draws)

def evaluate_agents(agent_1, agent_2, n_battles: int):
    env = Connect4GameEnv()

    agent_1.eval()
    agent_2.eval()

    wins = 0
    draws = 0

    for game in range(n_battles):
        state = env.reset()
        done = False
        
        while not done:
            player = env.current_player
            state = env.state
            
            if player == 1:
                # Player 1 (Agent 1)
                current_state, (action, _) = agent_1.select_action(state, player, mode="eval", return_probabilities=True)
            else:
                # Player 2 (Agent 2)
                current_state, (action, _) = agent_2.select_action(state, player, mode="eval", return_probabilities=True)
            
            next_game, reward, done, info = env.step(action)
            next_state = next_game.state
            
            env.state = next_state
            env.current_player = next_game.current_player
        
        # Optional: Process game results
        winner = env.check_winner(env.state, 1)
        if winner == 1:  # Player 1 wins
            wins += 1
        elif winner == 0:  # Draw
            draws += 1
    
    # Optional: Return game results, statistics, etc.
    if n_battles == draws:
        return 0.5
    
    return wins / (n_battles - draws)