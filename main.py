import os
import random
from datetime import datetime

import torch
import numpy as np
import transformers
import matplotlib.pyplot as plt

import trustpilot
import train_simple as simple
import train_mdl as mdl


def set_seed(val: int):
    """Sets the seed for all experiments

    Args:
        val (int): seed to set
    """
    transformers.set_seed(val)
    torch.manual_seed(val)
    random.seed(val)
    np.random.seed(val)


def plot_data(test_scores: np.ndarray, name: str):
    """Plots the f1 or MDL scores given an array of shape `(num_reps, 12)`

    Args:
        test_scores (np.ndarray): f1 or MDL scores from experiments
    """
    mean_data = np.mean(test_scores, axis=0)
    max_data = np.amax(test_scores, axis=0)
    min_data = np.amin(test_scores, axis=0)
    x_coords = list(range(len(mean_data)))

    plt.title("bert-base-uncased probing age on TrustPilot")
    plt.xlabel("Layer number")
    plt.ylabel(f"Test {name} Score")

    plt.plot(mean_data, c="b")
    plt.fill_between(x_coords, mean_data, max_data, color="b", alpha=0.2)
    plt.fill_between(x_coords, mean_data, min_data, color="b", alpha=0.2)


def main():
    set_seed(0)

    data_file = os.path.join("data", "united_states.auto-adjusted_gender.jsonl")
    model_type = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating F1 and MDL plots for {model_type} on {data_file}...")
    test_scores = np.zeros((2, 5, 12))
    for rep in range(5):
        for emb_layer in range(1, 13):
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Repetition {rep+1}/5, Layer {emb_layer}/12"
            )

            dataset = trustpilot.get_dataset(
                data_file,
                model_type,
                emb_layer,
                device,
                use_age=True,
            )

            train, val, test = simple.get_data_loaders_simple(dataset)

            test_loss, test_f1 = simple.train_probe(
                train,
                val,
                test,
                dataset.get_hidden_dim(),
                device,
                verbose=False,
            )
            mdl_score = mdl.train_probe(dataset, device, verbose=False)

            test_scores[0, rep, emb_layer - 1] = test_f1
            test_scores[1, rep, emb_layer - 1] = mdl_score

    os.makedirs("figures", exist_ok=True)
    for idx, score_type in enumerate(("F1", "MDL")):
        path = os.path.join("figures", f"{model_type}_{score_type}.png")
        plot_data(test_scores[idx], "MDL")
        plt.savefig(path)
        plt.figure()
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
