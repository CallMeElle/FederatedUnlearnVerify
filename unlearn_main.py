"""
Main file for feature unlearning
"""
import random
import torch
import numpy as np
from unlearn_strategies import strategies
from configs import unlearn_config
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main() -> None:
    args = unlearn_config.arguments
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    unlearn_feature = strategies.FeatureUnlearning(args=args)
    unlearn_feature.unlearn()


if __name__ == "__main__":
    main()