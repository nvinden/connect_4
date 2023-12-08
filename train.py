from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from game import Connect4GameEnv
from model import DQNAgent
from config import CONFIG
from play import play_games_return_ER, play_game_return_ER, evaluate_agents

from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import time
import os

import dask
from dask.distributed import Client
from distributed.protocol.torch import serialize, dask_serialize
import logging
import gc


@dask_serialize.register(torch.Tensor)
def serialize_torch_Tensor(t):
    """Need to fix this implementation when gpu is on device

    This is a bug in dask

    copied from here: https://github.com/dask/distributed/blob/172f23d78ac1f8c6117b9edfd0019ec94cd7d39d/distributed/protocol/torch.py#L15
    """  # noqa
    requires_grad_ = t.requires_grad

    if requires_grad_:
        sub_header, frames = serialize(t.detach().cpu().numpy())
    else:
        sub_header, frames = serialize(t.cpu().numpy())

    header = {"sub-header": sub_header}
    if t.grad is not None:
        grad_header, grad_frames = serialize(t.grad.numpy())
        header["grad"] = {"header": grad_header, "start": len(frames)}
        frames += grad_frames
    header["requires_grad"] = requires_grad_
    header["device"] = t.device.type
    return header, frames

def main():
    print(torch.cuda.is_available())
    torch.set_float32_matmul_precision('medium')

    agent = DQNAgent()
    #agent.q_network = torch.load("model_weights_cycle_8.pth")
    past_best_agent = deepcopy(agent)

    logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)

    if CONFIG["USE_MULTIPROCESSING"]:
        # Use separate processes for each Dask worker.
        #   # Adjust this value based on your needs
        client = Client(processes=True, n_workers = os.cpu_count(), threads_per_worker=CONFIG["THREADS_PER_PROCESS"])

    if CONFIG['USE_WANDB']:
        wandb.finish()
        wandb.init(project="connect_4_dqn")
        wandb.config.update(CONFIG)

    # Define a Trainer with 1 epoch. This will be run multiple times in a loop.
    for cycle in range(CONFIG['N_CYCLES']):
        game_time = time.time()

        # Using dask.delayed to play N_GAMES in parallel
        if CONFIG["USE_MULTIPROCESSING"]:
            total_num_threads = os.cpu_count() * CONFIG["THREADS_PER_PROCESS"]
            past_best_agent_future = client.scatter(past_best_agent)
            n_games = CONFIG['N_GAMES'] // total_num_threads

            delayed_games = [dask.delayed(play_games_return_ER)(past_best_agent_future, n_games) for _ in range(total_num_threads)]
            all_experiences = client.compute(delayed_games)
            all_experiences = client.gather(all_experiences)
            client.restart()
        else:
            all_experiences = [play_game_return_ER(past_best_agent) for _ in range(CONFIG['N_GAMES'])]

        # Now gather experiences and add them to replay buffer
        for game_experience in all_experiences:
            agent.populate_replay_buffer(game_experience)

        # Print time taken to play games
        game_time = time.time() - game_time
        print(f"Time taken to play {CONFIG['N_GAMES']} games: {game_time}")

        # 2. Train the agent for TRAIN_EPOCHS epochs
        train_time = time.time()
        accelerator = "gpu" if CONFIG["DEVICE"] == "cuda" else "cpu"
        cycle_trainer = Trainer(max_epochs=CONFIG['N_EPOCHS'], accelerator=accelerator, enable_model_summary=False)
        cycle_trainer.fit(agent)

        # Print time taken to train
        train_time = time.time() - train_time
        print(f"Time taken to train for {CONFIG['N_EPOCHS']} epochs: {train_time}")

        # 4. Test if the new agent is better than the old one
        if CONFIG["USE_MULTIPROCESSING"]:
            n_battles_per_worker = CONFIG["N_RANDOM_BATTLES"] // CONFIG["N_WORKERS"]
            past_best_agent_future = client.scatter(past_best_agent)
            agent_future = client.scatter(agent)
            delayed_battles = [dask.delayed(evaluate_agents)(past_best_agent_future, agent_future, n_battles_per_worker) for _ in range(CONFIG["N_WORKERS"])]
            battle_results = dask.compute(*delayed_battles)
            
            # Aggregating the results to compute the overall winning percentage
            old_opponent_win_rate = sum(battle_results) / len(battle_results)
        else:
            old_opponent_win_rate = old_opponent_win_rate(agent, past_best_agent, 100)

        if old_opponent_win_rate < CONFIG["WIN_RATE_THRESHOLD"]:
            past_best_agent.q_network = deepcopy(agent.q_network)

        # 5. Log some information
        wandb_log = {
            "cycle": cycle,
            "replay_buffer_size": len(agent.replay_buffer),
            "game_time": game_time,
            "train_time": train_time,
            "total_time": game_time + train_time,
            "old_opponent_win_rate": old_opponent_win_rate
        }

        save_model_filepath = f'model_weights_cycle_{cycle}.pth'
        torch.save(agent.q_network, save_model_filepath)

        if CONFIG['USE_WANDB']:
            wandb.save(save_model_filepath)
            wandb.log(wandb_log)

if __name__ == '__main__':
    main()
