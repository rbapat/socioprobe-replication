import argparse
import os
import random
from datetime import datetime

import torch
import numpy as np
import transformers
import matplotlib.pyplot as plt

import class_general as cg
import train_simple_general as simple
import train_mdl_general as mdl

import pandas as pd


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
    
    results = dict()
    results['data'] = []
    results['model'] = []
    results['layer'] = []
    results['seed'] = []
    results['test_f1'] = []
    results['test_mdl'] = []

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
                test_scores[0, rep, emb_layer - 1] = test_f1
                mdl_score = mdl.train_probe(dataset, device, verbose=verbose)
                test_scores[1, rep, emb_layer - 1] = mdl_score

                results['data'].append(data_file)
                results['model'].append(model_type)
                results['seed'].append(rep)
                results['layer'].append(emb_layer)
                results['test_f1'].append(test_f1)
                results['test_mdl'].append(mdl_score)
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
            test_scores[0, rep] = test_f1
            mdl_score = mdl.train_probe(dataset, device, verbose=verbose)
            test_scores[1, rep] = mdl_score

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{filename.replace(".csv", "")}_{model_type.replace("/", "_")}_results.csv')

    return test_scores

def get_args():
    """Parses arguments to the script; Specifically expects a model arg"""
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse Arugments
    args = get_args()

    # VARIABLES
    verbose = True
    bases = ['reddit', 'facebook_wiki', 'facebook_congress', 'fitocracy']
    model_type = args.model

    for b in bases:
        print(f'Probing: {b}')
        data_location = f"{b}_posts_clean.csv"
        f1_scores_layerall = model_pipeline(model_type, data_location, verbose=verbose)
        model_type = model_type
        os.makedirs("figures", exist_ok=True)
        for idx, score_type in enumerate(("F1", "MDL")):
            path = os.path.join("figures", f'{b}_{model_type.replace("/", "_")}_{score_type}.png')
            try:
                plot_data(f1_scores_layerall[idx], model_type, score_type, 'Gender', b)
                plt.savefig(path)
                plt.figure()
                print(f"Saved {path}")
            except:
                np.savetxt(f'{b}_{model_type.replace("/", "_")}_{idx}_{score_type}.txt', f1_scores_layerall[idx], delimiter=',')


    # Output layer (layer 0)
    f1_scores_layer12 = model_pipeline(model_type, data_location, verbose=verbose, layer=12)
