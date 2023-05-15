from typing import Tuple, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

from trustpilot import ProbingDataset
from probes import SimpleProbe


def run_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    dataloader: DataLoader,
    is_train: bool,
) -> Tuple[float, float]:
    """Runs a single epoch for a given model, and updates model parameters if is_train is true

    Args:
        model (torch.nn.Module): Model to train or evaluate
        optimizer (torch.optim.Optimizer): optimizer used to update model parameters
        loss_function (torch.nn.Module): loss function used to update model parameters
        dataloader (DataLoader): dataset to sample data from
        is_train (bool): Whether or not to update the model parameters and query it in training mode

    Returns:
        Tuple[float, float]: loss and f1 score over entire epoch
    """
    running_loss = 0
    running_preds, running_gt = [], []

    orig_dataset = dataloader.dataset

    model.train(is_train)
    for embeds, ground_truth, sampled_idxs in dataloader:
        optimizer.zero_grad()
        raw_output = model(embeds)

        loss = loss_function(raw_output, ground_truth)
        if is_train:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(F.softmax(raw_output, dim=1), dim=1)
        for pred, actual in zip(preds, ground_truth):
            running_preds.append(pred.item())
            running_gt.append(actual.item())

        # if not is_train:
        #     for batch_idx in torch.where(preds != ground_truth)[0]:
        #         text = orig_dataset.orig_strings[sampled_idxs[batch_idx]]
        #         print(
        #             f"Predicted {preds[batch_idx].item()} when it should be {ground_truth[batch_idx].item()} for the following text: {text}"
        #         )

        running_loss += loss.item() * embeds.shape[0]

    return running_loss / len(running_preds), metrics.f1_score(
        running_gt, running_preds
    )


def get_data_loaders_simple(
    dataset: ProbingDataset,
    batch_size: int = 32,
    splits: List[int] = [0.8, 0.1, 0.1],
    indices: torch.Tensor = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Converts the probing dataset into train/val/test dataloaders

    Args:
        dataset (ProbingDataset): dataset we are converting
        batch_size (int): Batch size for each dataloader. Defaults = 32.
        splits: (List[int], optional): three integers that determine how big the training, validation, and testing sets are. Must sum to 1. Defaults to [0.8, 0.1, 0.1]
        indices: (torch.Tensor): indices of original dataset that we are splitting. Defaults to None, and will use the entire dataset in this case.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, and test dataloaders
    """

    assert len(splits) == 3 and sum(splits) == 1
    assert all(s >= 0 and s <= 1 for s in splits)

    if indices is None:
        indices = torch.LongTensor(range(len(dataset)))

    stratify = dataset.ground_truth[indices].cpu()
    train_idxs, _test_idxs = train_test_split(
        indices, train_size=splits[0], stratify=stratify
    )
    stratify = dataset.ground_truth[_test_idxs].cpu()

    if splits[2] == 0:
        val_idxs = _test_idxs
        test_idxs = []
    else:
        val_pct = splits[1] / (splits[1] + splits[2])
        val_idxs, test_idxs = train_test_split(
            _test_idxs, train_size=val_pct, stratify=stratify
        )

    subsets = []
    for idxs in (train_idxs, val_idxs, test_idxs):
        subsets.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(idxs),
            )
        )
    return tuple(subsets)


def train_probe(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    hidden_dim: int,
    device: torch.device,
    learning_rate: float = 1e-3,
    lr_factor: float = 0.5,
    es_patience: int = 5,
    verbose=True,
) -> Tuple[float, float]:
    """Trains a simple probe to predict either age or gender from the dataset

    Args:
        train_loader (DataLoader): training dataset
        val_loader (DataLoader): validation dataset
        test_loader (DataLoader): testing dataset
        hidden_dim (int): hidden dim of embeddings
        device: (torch.device): the device where we will be doing computations (cpu/cuda)
        learning_rate (float, optional): learning rate for Adam optimizer. Defaults to 1e-3.
        batch_size (int, optional): batch size of dataset. Defaults to 32.
        lr_factor (float, optional): Factor to decrease learning rate by if loss doesn't improve. Defaults to 0.5.
        es_patience (int, optional): How many consecutive non-optimal losses can occur before training stops. Defaults to 5.
        verbose (bool, optional): If true, will print out training info. Defaults to true.

    Returns:
        Tuple[float, float]: the test loss and f1 score
    """

    probe = SimpleProbe(hidden_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor
    )
    loss_function = torch.nn.CrossEntropyLoss()

    no_improvement = 0
    best_loss = float("inf")

    epoch = 0
    while no_improvement < es_patience:
        train_loss, train_f1 = run_epoch(
            probe, optimizer, loss_function, train_loader, True
        )
        val_loss, val_f1 = run_epoch(probe, optimizer, loss_function, val_loader, False)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        scheduler.step(val_loss)
        epoch += 1

        if verbose:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}: train_loss[{train_loss:.4f}], train_f1[{train_f1:.4f}], val_loss[{val_loss:.4f}], val_f1[{val_f1:.4}]"
            )

    test_loss, test_f1 = run_epoch(probe, optimizer, loss_function, test_loader, False)
    if verbose:
        print(f"Training completed after {epoch} epochs")
        print(f"Test loss: {test_loss:.4f}, test f1: {test_f1:.4f}")

    return test_loss, test_f1
