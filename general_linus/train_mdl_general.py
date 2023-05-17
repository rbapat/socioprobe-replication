import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

import train_simple_general as simple
from class_general import ProbingDataset


def train_probe(
    dataset: ProbingDataset,
    device: torch.device,
    batch_size: int = 32,
    val_split: float = 0.1,
    **kwargs
):
    """Trains and evaluates a probe using Minimum description length probing

    Args:
        dataset (ProbingDataset): full dataset (with pooled embeddings) that was parsed
        device (torch.device): torch device that we're doing all computations on (cpu/cuda)
        batch_size (int, optional): batch size for training. Defaults to 32.
        val_split (float, optional): Percent of dataset to set aside as a validation set. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    stratify = dataset.ground_truth.cpu()
    indices = torch.LongTensor(range(len(dataset)))

    mdl_idxs, val_idxs = train_test_split(
        indices, test_size=val_split, stratify=stratify
    )

    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idxs)
    )

    partitions = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]
    partitions = [p / 100 for p in partitions]

    mdl_length = (partitions[0] * len(dataset)) * np.log(2)

    for i in range(len(partitions) - 1):
        train, test = partitions[i], partitions[i + 1] - partitions[i]
        splits = (train, test, 1 - (train + test))

        train_loader, test_loader, _ = simple.get_data_loaders_simple(
            dataset, splits=splits, indices=mdl_idxs
        )

        test_loss, test_f1 = simple.train_probe(
            train_loader,
            val_loader,
            test_loader,
            dataset.get_hidden_dim(),
            device,
            out_dim=dataset.get_output_dim(),
            **kwargs
        )

        mdl_length -= test_loss

    return mdl_length
