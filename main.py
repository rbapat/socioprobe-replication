import os
import random

import torch
import numpy as np
import transformers
import matplotlib.pyplot as plt

import trustpilot
import train_simple as simple


def set_seed(val: int):
    """Sets the seed for all experiments

    Args:
        val (int): seed to set
    """
    transformers.set_seed(val)
    torch.manual_seed(val)
    random.seed(val)
    np.random.seed(val)


def plot_data(test_f1_scores: np.ndarray):
    """Plots the f1 scores given an array of shape `(num_reps, 12)`

    Args:
        test_f1_scores (np.ndarray): f1 scores from experiments
    """
    mean_data = np.mean(test_f1_scores, axis=0)
    max_data = np.amax(test_f1_scores, axis=0)
    min_data = np.amin(test_f1_scores, axis=0)
    x_coords = list(range(len(mean_data)))

    plt.title("bert-base-uncased probing age on TrustPilot")
    plt.xlabel("Layer number")
    plt.ylabel("Test F1 Score")

    plt.plot(mean_data[:, 0], c='b')
    plt.fill_between(
        x_coords, mean_data[:, 0], max_data[:, 0], color='b', alpha=0.2)
    plt.fill_between(
        x_coords, mean_data[:, 0], min_data[:, 0], color='b', alpha=0.2)
    plt.savefig(f"f1.png")


def main():
    set_seed(0)

    data_file = os.path.join(
        'data', 'united_states.auto-adjusted_gender.jsonl')
    model_type = 'bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_f1_scores = np.zeros((5, 12))
    for rep in range(5):
        for emb_layer in range(1, 13):
            train, val, test = trustpilot.get_dataset(
                data_file, model_type, emb_layer, device)

            f1 = simple.train_probe(
                train, val, test, device, True, verbose=False)
            test_f1_scores[rep, emb_layer - 1] = f1

    plot_data(test_f1_scores)


if __name__ == '__main__':
    main()
