"""
Main file for FL model training
"""
import random
import torch
import numpy as np
from fl_strategies import training
from configs import train_config
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main() -> None:
    args = train_config.arguments
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    start_time = time.time()

    fl = training.fl_training(arguments=args)
    fl.train()

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Time elapsed: {round(time_elapsed, 4)} s")


if __name__ == "__main__":
    main()