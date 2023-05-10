from typing import Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from torch.utils.data import DataLoader

from nps_class import NPSChatCorpus


class SimpleProbe(torch.nn.Module):
    def __init__(self, in_dim):
        super(SimpleProbe, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Change based on number of classes
        )

    def forward(self, x):
        return self.model(x)


def run_epoch(model: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              loss_function: torch.nn.Module,
              dataloader: DataLoader,
              is_train: bool) -> Tuple[float, float]:
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

    model.train(is_train)
    for embeds, labels in dataloader:
        optimizer.zero_grad()
        raw_output = model(embeds)
        gt = labels

        loss = loss_function(raw_output, gt)
        if is_train:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(F.softmax(raw_output, dim=1), dim=1)
        for pred, actual in zip(preds, gt):
            running_preds.append(pred.item())
            running_gt.append(actual.item())

        running_loss += loss.item() * embeds.shape[0]

    # TODO: Need error analysis! So must return preds and actual labels
    if is_train is True:
        return running_loss / len(running_preds), metrics.f1_score(running_gt, running_preds, average="weighted")
    elif is_train is False:
        return running_loss / len(running_preds), metrics.f1_score(running_gt, running_preds, average="weighted"), running_preds, running_gt


def train_probe(train_set: NPSChatCorpus,
                val_set: NPSChatCorpus,
                test_set: NPSChatCorpus,
                device: torch.device,
                learning_rate: float = 1e-3,
                batch_size: int = 32,
                lr_factor: float = 0.5,
                es_patience: int = 5,
                verbose=True):
    """Trains a simple probe to predict labels from the dataset

    Args:
        train_set (NPSChatCorpus): training dataset
        val_set (NPSChatCorpus): validation dataset
        test_set (NPSChatCorpus): testing dataset
        device (torch.device): torch device where computations should occur
        probe_age (bool, optional): _description_. Defaults to False.
        probe_gender (bool, optional): _description_. Defaults to False.
        learning_rate (float, optional): learning rate for Adam optimizer. Defaults to 1e-3.
        batch_size (int, optional): batch size of dataset. Defaults to 32.
        lr_factor (float, optional): Factor to decrease learning rate by if loss doesn't improve. Defaults to 0.5.
        es_patience (int, optional): How many consecutive non-optimal losses can occur before training stops. Defaults to 5.
        verbost (bool, optional): If true, will print out training info. Defaults to true.
    """

    model = SimpleProbe(train_set.get_hidden_dim()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    no_improvement = 0
    best_loss = float('inf')

    epoch = 0
    while no_improvement < es_patience:
        train_loss, train_f1 = run_epoch(model, optimizer, loss_function,
                                         train_loader, True)
        val_loss, val_f1, val_preds, val_actual = run_epoch(model, optimizer, loss_function,
                                                            val_loader, False)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        scheduler.step(val_loss)
        epoch += 1

        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}: train_loss[{train_loss:.4f}], train_f1[{train_f1:.4f}], val_loss[{val_loss:.4f}], val_f1[{val_f1:.4}]")

    test_loss, test_f1, test_preds, test_actual = run_epoch(model, optimizer, loss_function, test_loader, False)

    # Error analysis
    mismatch_index = [idx for idx, elem in enumerate(test_preds) if elem != test_actual[idx]]
    # TODO: Return mismatching predictions

    if verbose:
        print(f'Training completed after {epoch} epochs')
        print(f'Test loss: {test_loss:.4f}, test f1: {test_f1:.4f}')

    return test_f1
