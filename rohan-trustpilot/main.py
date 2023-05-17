import os
import random
from pprint import pprint
from datetime import datetime

import torch
import numpy as np
import pandas as pd
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


def plot_data(test_scores: np.ndarray, model_type: str, name: str):
    """Plots the f1 or MDL scores given an array of shape `(num_reps, 12)`

    Args:
        test_scores (np.ndarray): f1 or MDL scores from experiments
    """
    mean_data = np.mean(test_scores, axis=0)
    max_data = np.amax(test_scores, axis=0)
    min_data = np.amin(test_scores, axis=0)
    x_coords = list(range(len(mean_data)))

    plt.title(f"{model_type} probing age on TrustPilot")
    plt.xlabel("Layer number")
    plt.ylabel(f"Test {name} Score")

    plt.plot(mean_data, c="b")
    plt.fill_between(x_coords, mean_data, max_data, color="b", alpha=0.2)
    plt.fill_between(x_coords, mean_data, min_data, color="b", alpha=0.2)


def format_scores(
    input_data: np.ndarray, data_name: str, model: str, layer: int = None
) -> pd.DataFrame:
    """Given a numpy ndarray, converts it to a pandas dataframe
    Checks the dimensions of the ndarray and alters it to fit into a dataframe
    If it's 2D (2, 5), outer loop is F1 score or MDL, and inner is the 5 iterations of the model
    If it's 3D (2, 5, 12), outer loop is F1 score or MDL, second loop is 5 iterations of the model,
        and 12 is the representation for each layer

    Parameters
    ----------
    input_data : np.ndarray
        numpy ndarray holding the F1 scores for MDL scores for a particular model
        Assumes that first layer represents F1, second layer is MDL
    data_name : str
        name of dataset used for columns
    model : str
        name of LLM model used
    layer : int, by default None
        layer used for 2D array. Not used if 3D array is given

    Returns
    ----------
    pd.DataFrame
        dataframe holding contents of scores, with columns
        `data_name`, `model`, `layer`, `score_type`, `run1`, `run2`, `run3`, `run4`, `run5`
    """
    # First, ensure that dimensions are correct
    input_dim = input_data.shape

    data = pd.DataFrame()

    if len(input_dim) == 2:
        assert input_dim == (
            2,
            5,
        ), "Input dimension should be (2, 5), where 2 represents F1 or MDL scores, and 5 is the number of runs"
        assert layer is not None, "Layer must be specified data creation"

        df = pd.DataFrame(input_data)
        df.columns = ["run1", "run2", "run3", "run4", "run5"]
        df["data_name"] = data_name
        df["model"] = model
        df["layer"] = layer
        df["score_type"] = ["F1", "MDL"]

        data = pd.concat([data, df], ignore_index=True)

    elif len(input_dim) == 3:
        assert input_dim == (
            2,
            5,
            12,
        ), "Input dimension should be (2, 5, 12), where 2 represents F1 or MDL scores, 5 is the number of runs, and 12 is the number of layers"

        for idx, score_type in enumerate(("F1", "MDL")):
            df = pd.DataFrame(input_data[idx]).transpose()
            df.columns = ["run1", "run2", "run3", "run4", "run5"]
            df["data_name"] = data_name
            df["model"] = model
            df["layer"] = range(1, input_dim[2] + 1)
            df["score_type"] = score_type

            data = pd.concat([data, df], ignore_index=True)
    else:
        Exception(
            f"Unknown input dimension provided. Data provided has shape {input_dim}"
        )

    return data


def run_experiments(params: dict):
    print("Running experiment:")
    pprint(params)
    print()

    num_reps = params["num_reps"]
    num_hidden = params["num_hidden"]
    model_type = params["model_type"]

    test_scores = np.zeros((2, num_reps, num_hidden))
    for rep in range(num_reps):
        for emb_layer in range(1, num_hidden + 1):
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Repetition {rep+1}/{num_reps}, Layer {emb_layer}/{num_hidden}"
            )

            dataset = trustpilot.get_dataset(
                params["data_file"],
                model_type,
                emb_layer,
                params["device"],
                use_age=params["use_age"],
                use_gender=params["use_gender"],
            )

            train, val, test = simple.get_data_loaders_simple(dataset)

            test_loss, test_f1, _ = simple.train_probe(
                train,
                val,
                test,
                dataset.get_hidden_dim(),
                params["device"],
                verbose=params["verbose"],
            )
            mdl_score = mdl.train_probe(
                dataset, params["device"], verbose=params["verbose"]
            )

            test_scores[0, rep, emb_layer - 1] = test_f1
            test_scores[1, rep, emb_layer - 1] = mdl_score / 1024

    model_type = model_type.replace("/", "-")
    os.makedirs("figures", exist_ok=True)

    gt = "age" if params["use_age"] else "gender"
    for idx, score_type in enumerate(("F1", "MDL")):
        figure_path = os.path.join("figures", f"{model_type}_{score_type}_{gt}.png")
        plot_data(test_scores[idx], model_type, score_type)
        plt.savefig(figure_path)
        plt.figure()

        print(f"Saved {figure_path}")

    scores_path = os.path.join("figures", f"{model_type}_{gt}")
    torch.save(test_scores, scores_path + ".pt")

    df = format_scores(torch.from_numpy(test_scores), "TrustPilot", model_type)
    df.to_csv(scores_path + ".csv")


def main():
    set_seed(0)

    params = {
        "model_type": "microsoft/deberta-base",
        "data_file": os.path.join("data", "united_states.auto-adjusted_gender.jsonl"),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "num_hidden": 12,
        "num_reps": 5,
        "use_age": True,
        "use_gender": False,
        "verbose": False,
    }

    for model_type in ["microsoft/deberta-base", "google/electra-small-discriminator"]:
        experiment_params = params.copy()
        experiment_params["model_type"] = model_type

        run_experiments(experiment_params)


if __name__ == "__main__":
    main()
