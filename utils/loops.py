"""Training and testing loops defined for convenient reuse"""

import torch
import torch.nn as nn
from tqdm import tqdm
from model import device


def train_loop(train_loader, model, loss_fn, optimizers, epoch, clip_gradients, augment):
    """
    Runs the training loop on a given model for a certain number of epochs

    Args:
        dataloader: A dataloader providing the batched training data
        model: The model to train
        loss_fn: A function which evaluates a single loss value given the output of `model`
        optimizers: A list of optimizers to use when training
        epoch: The current epoch (0 indexed)
        clip_gradients: Clip gradients on the interval [-clip_gradients, clip_gradients] during training,
            set to `None` to disable
        augment: A `nn.Module` for data augmentation 

    Returns:
        A list of losses for each batch in this epoch
    """
    model.train()
    losses = []  # Keep track of the losses during each epoch

    with tqdm(train_loader, unit="batch") as train_epoch:
        for X in train_epoch:
            train_epoch.set_description(f"Epoch {epoch+1}")
            
            # Make sure we have the training data on the right device
            X = X.to(device)
            X = augment(X)

            # Reset the optimizers
            for optimizer in optimizers:
                optimizer.zero_grad()

            # Forward step
            Y = model(X)

            # Compute loss and run the backwards step
            loss = loss_fn(X, Y)
            loss.backward()

            # Gradient clipping
            if clip_gradients:
                nn.utils.clip_grad_value_(model.parameters(), clip_gradients)

            # Let the optimizers do their magic using the information from the backwards step
            for optimizer in optimizers:
                optimizer.step()

            losses.append(float(loss.item()))
            train_epoch.set_postfix(loss=loss.item())

    return losses


def eval_loop(dataloader, model, loss_fn):
    """
    Runs an evaluation loop on either the validation or test data

    Args:
        dataloader: A data loader providing either validation or test data
        model: The model to evaluate
        loss_fn: The loss function used for evaluation

    Returns:
        The average loss over all data in the given dataloader
    """
    model.eval()
    loss = 0
    for X in dataloader:
        with torch.no_grad():
            X = X.to(device)
            Y = model(X)
            loss += loss_fn(X, Y)
    model.train()
    return loss.item()/len(dataloader)
