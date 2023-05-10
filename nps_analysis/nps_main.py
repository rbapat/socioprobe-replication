import os
import random

import torch
import numpy as np
import transformers
import matplotlib.pyplot as plt

import nps_class
import nps_train_simple as simple


def set_seed(val: int):
    """Sets the seed for all experiments

    Args:
        val (int): seed to set
    """
    transformers.set_seed(val)
    torch.manual_seed(val)
    random.seed(val)
    np.random.seed(val)


def plot_data(test_f1_scores: np.ndarray, plot_title: str, filename="Blah.png"):
    """Plots the f1 scores given an array of shape `(num_reps, 12)`

    Args:
        test_f1_scores (np.ndarray): f1 scores from experiments
    """
    mean_data = np.mean(test_f1_scores, axis=0)
    upper_data = np.array(mean_data) + np.std(test_f1_scores, axis=0)
    lower_data = np.array(mean_data) - np.std(test_f1_scores, axis=0)
    x_coords = list(range(len(mean_data)))

    plt.title(plot_title)
    plt.xlabel("Layer number")
    plt.ylabel("Test F1 Score")

    plt.plot(mean_data[:], c='b')
    plt.fill_between(
        x_coords, mean_data[:], upper_data[:], color='b', alpha=0.2)
    plt.fill_between(
        x_coords, mean_data[:], lower_data[:], color='b', alpha=0.2)
    plt.savefig(filename)
    plt.show()

# # Rohan's Code!
# def main():
#     set_seed(0)

#     data_file = os.path.join(
#         'data', 'nps_chat_corpus_clean.csv')
#     model_type = 'google/electra-base-discriminator'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     test_f1_scores = np.zeros((5, 12))
#     for rep in range(5):
#         for emb_layer in range(1, 13):
#             train, val, test = nps_class.get_dataset(
#                 data_file, model_type, emb_layer, device)

#             f1 = simple.train_probe(
#                 train, val, test, device, True, verbose=False)
#             test_f1_scores[rep, emb_layer - 1] = f1

#     plot_data(test_f1_scores)


def model_pipeline(model: str, filename: str, layer: int = None, verbose: bool = False):
    set_seed(0)

    data_file = os.path.join('data', filename)
    model_type = model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_f1_scores = np.zeros((5, 12)) if layer is None else np.zeros(5)
    for rep in range(5):  # Loop used to initiate probe at different seeds
        if layer is None:
            for emb_layer in range(1, 13):
                train, val, test = nps_class.get_dataset(
                    data_file, model_type, emb_layer, device)

                f1 = simple.train_probe(
                    train, val, test, device, True, verbose=verbose)
                test_f1_scores[rep, emb_layer - 1] = f1
        else:
            train, val, test = nps_class.get_dataset(
                data_file, model_type, layer, device)

            f1 = simple.train_probe(
                train, val, test, device, True, verbose=verbose)
            test_f1_scores[rep] = f1
    return np.flip(test_f1_scores, axis=-1)  # Need to flip because final embedding is 1!


if __name__ == '__main__':
    verbose = False

    # ELECTRA
    nps_electra_f1_scores_layerall = model_pipeline("google/electra-base-discriminator", "nps_chat_corpus_clean.csv", verbose=verbose)
    plot_data(nps_electra_f1_scores_layerall,
              plot_title="ELECTRA Model Probing Age on NPS Chat Corpus",
              filename="ELECTRA_NPS_layerall.png")

    nps_electra_f1_scores_layer1 = model_pipeline("google/electra-base-discriminator", "nps_chat_corpus_clean.csv", layer=1, verbose=verbose)
    # Error expected for this plot
    # plot_data(nps_electra_f1_scores_layer1,
    #           plot_title="ELECTRA probing Age on NPS Chat Corpus, Layer 1",
    #           filename="ELECTRA_NPS_layer1.png")
    print("Electra has finished")

    # RoBERTa - roberta-base
    nps_roberta_f1_scores = model_pipeline("roberta-base", "nps_chat_corpus_clean.csv", verbose=verbose)
    plot_data(nps_roberta_f1_scores,
              plot_title="RoBERTa-Base Model Probing Age on NPS Chat Corpus, all layers",
              filename="BERT_NPS_layerall.png")
    print("RoBERTa has finished")
