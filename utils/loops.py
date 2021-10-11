"""Training and testing loops defined for convenient reuse"""

import torch.nn as nn
from tqdm import tqdm
from model import device


def train_loop(dataloader, model, loss_fn, optimizers, epoch, clip_gradients):
    """
    Runs the training loop on a given model for a certain number of epochs

    Args:
        dataloader: A dataloader providing the batched training data
        model: The model to train
        loss_fn: A function which evaluates a single loss value given the output of `model`
        optimizers: A list of optimizers to use when training
        epoch: The current epoch
        clip_gradients: Clip gradients on the interval [-clip_gradients, clip_gradients] during training,
            set to `None` to disable
    """
    
    with tqdm(dataloader, unit="batch") as train_epoch:
        for X in train_epoch:
            train_epoch.set_description(f"Epoch {epoch}")
            
            # Make sure we have the training data on the right device
            X = X.to(device)

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

            train_epoch.set_postfix(loss=loss.item())
