import os
import random

import torch
import numpy as np
import transformers

from trustpilot import get_dataset


def set_seed(val: int):
    """Sets the seed for all experiments

    Args:
        val (int): seed to set
    """
    transformers.set_seed(val)
    torch.manual_seed(val)
    random.seed(0)
    np.random.seed(0)


def main():
    data_file = os.path.join(
        'data', 'united_states.auto-adjusted_gender.jsonl')
    model_type = 'bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train, val, test = get_dataset(
        data_file, model_type, 12, device)

    train.describe('Train')
    val.describe('Val')
    test.describe('Test')


if __name__ == '__main__':
    main()
