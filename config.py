

CONFIG = {
    'LEARNING_RATE': 0.0003,
    'N_CYCLES': 300,
    'N_GAMES': 150,
    'N_EPOCHS': 1,
    'BATCH_SIZE': 32,
    'USE_WANDB': True,
    'DEVICE': "cpu",
    'MCTS_ITERATIONS': 200,
    'ER_SIZE': 50000,

    # Multiprocessing
    'USE_MULTIPROCESSING': True,
    'N_WORKERS': 8,
    'THREADS_PER_PROCESS': 3,

    # Evaluation
    'N_RANDOM_BATTLES': 48,
    'WIN_RATE_THRESHOLD': 0.55,
}

CONFIGGDF = {
    'LEARNING_RATE': 0.0003,
    'N_CYCLES': 3,
    'N_GAMES': 2,
    'N_EPOCHS': 1,
    'BATCH_SIZE': 2,
    'USE_WANDB': False,
    'DEVICE': "cpu",
    'MCTS_ITERATIONS': 30,
    'ER_SIZE': 20000,

    # Multiprocessing
    'USE_MULTIPROCESSING': False,
    'N_WORKERS': 24,
    'THREADS_PER_PROCESS': 2,

    # Evaluation
    'N_RANDOM_BATTLES': 3,
    'WIN_RATE_THRESHOLD': 0.55,
}
