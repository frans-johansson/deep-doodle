"""Training and testing loops defined for convenient reuse"""

import torch.nn as nn


def train_loop(dataloader, model, loss_fn, optimizers, num_epochs, print_every, clip_gradients):
    """
    Runs the training loop on a given model for a certain number of epochs

    Args:
        dataloader: A dataloader providing the batched training data
        model: The model to train
        loss_fn: A function which evaluates a single loss value given the output of `model`
        optimizers: A list of optimizers to use when training
        num_epochs: The number of epochs to run
        print_every: How often to log the loss value
        clip_gradients: Clip gradients on the interval [-clip_gradients, clip_gradients] during training 
    """
    size = len(dataloader.dataset)

    for e in range(num_epochs):
        print(f"\nEpoch {e+1}\n")

        for i, X in enumerate(dataloader):
            # Reset the optimizers
            for optimizer in optimizers:
                optimizer.zero_grad()

            # Forward step
            Y = model(X)

            # Compute loss and run the backwards step
            loss = loss_fn(X, Y)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), clip_gradients)

            # Let the optimizers do their magic using the information from the backwards step
            for optimizer in optimizers:
                optimizer.step()

            if i % print_every == 0:
                loss, current = loss.item(), i * len(X)
                print(f"[{current:>5d}/{size:>5d}] loss = {loss:>7f} ")
