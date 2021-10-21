import datetime as dt
import argparse
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import device
from model.basic import SketchRNN
from model.loss import sketch_rnn_loss
from utils.data import load_quickdraw_data
from utils.loops import train_loop, eval_loop


def handle_arguments():
    """Handle arguments from the command line."""

    argparser = argparse.ArgumentParser(
        prog="DeepDoodle training script",
        description="Trains a VAE to reproduce doodles of a given class using given hyperparameters",
    )

    argparser.add_argument(
        "-c",
        help="Classes to train on. Will expect sketchrnn_[class name].npz files located in the data directory for each class",
        dest="classes",
        nargs="+",
        type=str,
    )
    argparser.add_argument(
        "-e", help="Number of epochs to run", dest="num_epochs", type=int
    )
    argparser.add_argument(
        "-b",
        help="Batch size during training",
        dest="batch_size",
        type=int,
        default=100,
    )
    argparser.add_argument(
        "-lr",
        help="Learning rate for the ADAM optimizer",
        dest="learning_rate",
        type=float,
        default=0.0001,
    )
    argparser.add_argument(
        "-dr",
        help="Dropout keep probability",
        dest="dropout",
        type=float,
        default=0.9,
    )
    argparser.add_argument(
        "-dd",
        help="Data directory holding SketchRNN data in .npz format",
        dest="data_dir",
        type=str,
        default="data/quickdraw",
    )
    argparser.add_argument(
        "-o",
        help="Output directory to place the model files when done with training",
        dest="output_dir",
        type=str,
        default="data/models",
    )
    argparser.add_argument(
        "-Eh",
        help="Size of the hidden latent space of the encoder",
        dest="enc_hidden",
        type=int,
        default=256,
    )
    argparser.add_argument(
        "-Nz",
        help="Size of the hidden latent space of the sampled vectors z",
        type=int,
        default=128,
    )
    argparser.add_argument(
        "-Dh",
        help="Size of the hidden latent space of the decoder",
        dest="dec_hidden",
        type=int,
        default=512,
    )
    argparser.add_argument(
        "-m",
        help="Number of mixture components to learn",
        dest="num_mixtures",
        type=int,
        default=10,
    )

    return argparser.parse_args()


if __name__ == "__main__":
    # Parse and retrieve arguments
    args = handle_arguments()
    classes = args.classes
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    data_dir = args.data_dir
    output_dir = args.output_dir
    enc_hidden = args.enc_hidden
    Nz = args.Nz
    dec_hidden = args.dec_hidden
    num_mixtures = args.num_mixtures

    # TODO: Consider moving to the argparser at some point
    W_kl = 1.0
    clip_gradients = 1.0
    loss_dir = "data/loss"

    # Set up datasets and loaders for training, testing and validating
    train_data, test_data, valid_data = load_quickdraw_data(classes, data_dir)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=("cuda" in device)
    )
    test_loader = DataLoader(
        test_data, batch_size=len(test_data), pin_memory=("cuda" in device)
    )
    valid_loader = DataLoader(
        valid_data, batch_size=len(valid_data), pin_memory=("cuda" in device)
    )

    # Set up model and optimizers
    model = SketchRNN(
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        z_dims=Nz,
        num_mixtures=num_mixtures,
        dropout=dropout
    )
    model = model.to(device)
    enc_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
    dec_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate)
    loss_fn = sketch_rnn_loss(W_kl)

    # Keep track of losses for visualization
    valid_losses = []
    train_losses = []

    # Run the training and validation loops
    for epoch in range(num_epochs):
        print()

        epoch_losses = train_loop(
            train_loader,
            model,
            loss_fn,
            [enc_optimizer, dec_optimizer],
            epoch,
            clip_gradients
        )
        valid_loss = eval_loop(
            valid_loader,
            model,
            loss_fn
        )
        
        print(f"Validation loss: {valid_loss:.3f}")
        valid_losses += [valid_loss] * batch_size
        train_losses += epoch_losses
    
    # Final test score
    test_loss = eval_loop(
        test_loader,
        model,
        loss_fn
    )
    print(f"Test loss: {test_loss:.3f}")

    # Save the model, encoder and decoder separately
    timestamp = dt.datetime.now().strftime("%d%m%y_%H%M")
    torch.save(model, pathlib.Path(output_dir) / pathlib.Path(f"sketchrnn_{timestamp}.pth"))
    torch.save(model.encoder.state_dict(), pathlib.Path(output_dir) / pathlib.Path(f"encoder_{timestamp}.pth"))
    torch.save(model.decoder.state_dict(), pathlib.Path(output_dir) / pathlib.Path(f"decoder_{timestamp}.pth"))

    # Save the loss values for plotting
    np.save(pathlib.Path(loss_dir) / pathlib.Path(f"train_{timestamp}.npy"), train_losses)
    np.save(pathlib.Path(loss_dir) / pathlib.Path(f"valid_{timestamp}.npy"), valid_losses)
