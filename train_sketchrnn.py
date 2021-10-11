import argparse
import pathlib
import torch
from torch.utils.data import DataLoader
from model import device
from model.basic import SketchRNN
from model.loss import sketch_rnn_loss
from utils.data import load_quickdraw_data
from utils.loops import train_loop


def handle_arguments():
    """Handle arguments from the command line."""

    argparser = argparse.ArgumentParser(
        prog="DeepDoodle training script",
        description="Trains a VAE to reproduce doodles of a given class using given hyperparameters",
    )

    argparser.add_argument(
        "-c",
        help="Class to train on. Will expect a sketchrnn_[class name].npz file located in the data directory",
        dest="class_name",
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
    class_name = args.class_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    data_dir = args.data_dir
    enc_hidden = args.enc_hidden
    Nz = args.Nz
    dec_hidden = args.dec_hidden
    num_mixtures = args.num_mixtures

    # TODO: Consider moving to the argparser at some point
    print_every = 1
    W_kl = 1.0
    clip_gradients = 1.0

    # Set up dataset and loader for training
    data_path = pathlib.Path(data_dir) / pathlib.Path(f"sketchrnn_{class_name}.npz")
    train_data, _, _ = load_quickdraw_data(data_path)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=("cuda" in device)
    )  # Choo-chooo c:

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

    # Run the training loop
    train_loop(
        train_loader,
        model,
        loss_fn,
        [enc_optimizer, dec_optimizer],
        num_epochs,
        print_every,
        clip_gradients
    )
