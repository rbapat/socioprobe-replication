import os
import random
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import transformers
import matplotlib.pyplot as plt

import class_general as cg
import train_simple_general as simple
import train_mdl_general as mdl


def set_seed(val: int):
    """Sets the seed for all experiments

    Args:
        val (int): seed to set
    """
    transformers.set_seed(val)
    torch.manual_seed(val)
    random.seed(val)
    np.random.seed(val)


def plot_data(test_scores: np.ndarray, model_type: str, name: str, variable: str, dataset: str):
    """Plots the f1 or MDL scores given an array of shape `(num_reps, 12)`

    Args:
        test_scores (np.ndarray): f1 or MDL scores from experiments
        model_type (str): name of LLM that was probed
        name (str): name of metric used (F1 or MDL)
        variable (str): variable type for label, such as age or gender
        dataset (str): name of dataset being used
    """
    mean_data = np.mean(test_scores, axis=0)
    max_data = np.amax(test_scores, axis=0)
    min_data = np.amin(test_scores, axis=0)
    x_coords = list(range(len(mean_data)))

    plt.title(f"{model_type} probing {variable} on {dataset}")
    plt.xlabel("Layer number")
    plt.ylabel(f"Test {name} Score")

    plt.plot(mean_data, c="b")
    plt.fill_between(x_coords, mean_data, max_data, color="b", alpha=0.2)
    plt.fill_between(x_coords, mean_data, min_data, color="b", alpha=0.2)


def model_pipeline(model: str, filename: str, layer: int = None, verbose: bool = False):
    set_seed(0)

    data_file = os.path.join("data", filename)
    model_type = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating F1 and MDL plots for {model_type} on {data_file}...")
    test_scores = np.zeros((2, 5, 12)) if layer is None else np.zeros((2, 5))
    for rep in range(5):
        if layer is None:
            for emb_layer in range(1, 13):
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Repetition {rep+1}/5, Layer {emb_layer}/12"
                )

                dataset = cg.get_dataset(
                    data_file,
                    model_type,
                    emb_layer,
                    device,
                )

                train, val, test = simple.get_data_loaders_simple(dataset)

                test_loss, test_f1 = simple.train_probe(
                    train,
                    val,
                    test,
                    dataset.get_hidden_dim(),
                    device,
                    verbose=False,
                    out_dim=dataset.get_output_dim(),
                )
                mdl_score = mdl.train_probe(dataset, device, verbose=verbose)

                test_scores[0, rep, emb_layer - 1] = test_f1
                test_scores[1, rep, emb_layer - 1] = mdl_score
        else:
            dataset = cg.get_dataset(
                data_file,
                model_type,
                layer,
                device,
            )

            train, val, test = simple.get_data_loaders_simple(dataset)

            test_loss, test_f1 = simple.train_probe(
                train,
                val,
                test,
                dataset.get_hidden_dim(),
                device,
                verbose=False,
                out_dim=dataset.get_output_dim(),
            )
            mdl_score = mdl.train_probe(dataset, device, verbose=verbose)

            test_scores[0, rep] = test_f1
            test_scores[1, rep] = mdl_score

    return test_scores


def format_scores(input_data: np.ndarray, data_name: str, model: str, layer: int = None) -> pd.DataFrame:
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
        assert input_dim == (2, 5), "Input dimension should be (2, 5), where 2 represents F1 or MDL scores, and 5 is the number of runs"
        assert layer is not None, "Layer must be specified data creation"

        df = pd.DataFrame(input_data)
        df.columns = ["run1", "run2", "run3", "run4", "run5"]
        df["data_name"] = data_name
        df["model"] = model
        df["layer"] = layer
        df["score_type"] = ["F1", "MDL"]

        data = pd.concat([data, df], ignore_index=True)

    elif len(input_dim) == 3:
        assert input_dim == (2, 5, 12), "Input dimension should be (2, 5, 12), where 2 represents F1 or MDL scores, 5 is the number of runs, and 12 is the number of layers"

        for idx, score_type in enumerate(("F1", "MDL")):
            df = pd.DataFrame(input_data[idx]).transpose()
            df.columns = ["run1", "run2", "run3", "run4", "run5"]
            df["data_name"] = data_name
            df["model"] = model
            df["layer"] = range(1, input_dim[2]+1)
            df["score_type"] = score_type

            data = pd.concat([data, df], ignore_index=True)
    else:
        Exception(f"Unknown input dimension provided. Data provided has shape {input_dim}")

    return data


if __name__ == "__main__":
    # VARIABLES
    verbose = True
    data_location = "nps_chat_corpus_clean.csv"
    data_name = "NPS"

    # ELECTRA
    # All ELECTRA layers
    model_type = "google/electra-base-discriminator"
    nps_electra_f1_scores_layerall = model_pipeline(model_type, data_location, verbose=verbose)

    model_type = model_type.replace("/", "-")
    os.makedirs("figures", exist_ok=True)
    for idx, score_type in enumerate(("F1", "MDL")):
        path = os.path.join("figures", f"{model_type}_{score_type}.png")
        plot_data(nps_electra_f1_scores_layerall[idx], "ELECTRA", score_type, "Age", data_name)
        plt.savefig(path)
        plt.figure()
        print(f"Saved {path}")

    # Output layer (layer 0)
    model_type = "google/electra-base-discriminator"
    nps_electra_f1_scores_layer0 = model_pipeline(model_type, data_location, verbose=verbose,
                                                  layer=0)


    # RoBERTa-base
    model_type = "roberta-base"
    nps_robertabase_f1_scores_layerall = model_pipeline(model_type, data_location, verbose=verbose)

    model_type = model_type.replace("/", "-")
    os.makedirs("figures", exist_ok=True)
    for idx, score_type in enumerate(("F1", "MDL")):
        path = os.path.join("figures", f"{model_type}_{score_type}.png")
        plot_data(nps_robertabase_f1_scores_layerall[idx], "RoBERTa-base", score_type, "Age", data_name)
        plt.savefig(path)
        plt.figure()
        print(f"Saved {path}")

    # Output layer (layer 0)
    model_type = "roberta-base"
    nps_robertabase_f1_scores_layer0 = model_pipeline(model_type, data_location, verbose=verbose, layer=12)


    # DeBERTa-base
    model_type = "microsoft/deberta-base"
    nps_debertabase_f1_scores_layerall = model_pipeline(model_type, data_location, verbose=verbose)

    model_type = model_type.replace("/", "-")
    os.makedirs("figures", exist_ok=True)
    for idx, score_type in enumerate(("F1", "MDL")):
        path = os.path.join("figures", f"{model_type}_{score_type}.png")
        plot_data(nps_robertabase_f1_scores_layerall[idx], "DeBERTa-base", score_type, "Age", data_name)
        plt.savefig(path)
        plt.figure()
        print(f"Saved {path}")

    # Output layer (layer 0)
    model_type = "microsoft/deberta-base"
    nps_debertabase_f1_scores_layer0 = model_pipeline(model_type, data_location, verbose=verbose, layer=12)


    # Combine all data together
    final_df = pd.concat([format_scores(nps_electra_f1_scores_layerall, data_name, "ELECTRA"),
                          format_scores(nps_electra_f1_scores_layer0, data_name, "ELECTRA", layer=0),
                          format_scores(nps_robertabase_f1_scores_layerall, data_name, "RoBERTa-base"),
                          format_scores(nps_robertabase_f1_scores_layer0, data_name, "RoBERTa-base", layer=0),
                          format_scores(nps_debertabase_f1_scores_layerall, data_name, "DeBERTa-base"),
                          format_scores(nps_debertabase_f1_scores_layer0, data_name, "DeBERTa-base", layer=0)],
                         ignore_index=True)

    # Save scores
    os.makedirs("scores", exist_ok=True)
    final_df.to_csv(f"scores/{data_name}_scores.csv")
