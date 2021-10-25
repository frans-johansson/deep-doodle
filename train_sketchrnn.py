import datetime as dt
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import device
from model.sketchrnn import SketchRNN
from model.loss import sketch_rnn_loss
from utils.data import DataAugmentation, load_quickdraw_data, to_stroke_3
from utils.loops import train_loop, eval_loop
from utils.canvas import strokes_to_rgb


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
        "-Wkl",
        help="Kullback-Leibler coefficient for loss",
        dest="w_kl",
        type=float,
        default=1.0,
    )
    argparser.add_argument(
        "-dr",
        help="Dropout keep probability",
        dest="dropout",
        type=float,
        default=0.1,
    )
    argparser.add_argument(
        "-dd",
        help="Data directory holding SketchRNN data in .npz format",
        dest="data_dir",
        type=str,
        default="data/quickdraw",
    ),
    argparser.add_argument(
        "-ld",
        help="Log directory for TensorBoard",
        dest="log_dir",
        type=str,
        default="logs",
    ),
    argparser.add_argument(
        "-o",
        help="Output directory to place the model files when done with training",
        dest="output_dir",
        type=str,
        default="outputs",
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
    argparser.add_argument(
        "-de",
        help="Shows the progress of the model by reconstructing a test sample",
        dest="draw_every",
        type=int,
        default=0
    ),
    argparser.add_argument(
        "-se",
        help="Regularly save a checkpoint of the model after a certain number of epochs",
        dest="save_every",
        type=int,
        default=0
    ),
    argparser.add_argument(
        "--finetune",
        help="Whether or not to run in finetuning mode, must supply a model to finetune",
        type=str,
        default=None
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
    log_dir = args.log_dir
    output_dir = args.output_dir
    enc_hidden = args.enc_hidden
    Nz = args.Nz
    dec_hidden = args.dec_hidden
    num_mixtures = args.num_mixtures
    draw_every = args.draw_every
    save_every = args.save_every
    finetune = args.finetune
    W_kl = args.w_kl

    # TODO: Consider moving to the argparser at some point
    clip_gradients = 1.0
    kl_min = 0.2
    eta_min = 0.01
    R = 0.9999

    # Set up datasets and loaders for training, testing and validating
    train_data, test_data, valid_data = load_quickdraw_data(classes, data_dir)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False, pin_memory=("cuda" in device)
    )
    test_loader = DataLoader(
        test_data,  batch_size=batch_size, pin_memory=("cuda" in device)
    )
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, pin_memory=("cuda" in device)
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
    loss_fn = sketch_rnn_loss(W_kl, kl_min, eta_min, R)

    # Handle finetuning if utilized
    e = 0  # Number of epochs already trained before finetuning
    if finetune:
        finetune_dir = pathlib.Path(output_dir) / finetune
        model = torch.load(finetune_dir / "sketchrnn.pth")
        enc_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=learning_rate)
        dec_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=learning_rate)
        enc_optimizer.load_state_dict(torch.load(finetune_dir / "enc_opt.pth"))
        dec_optimizer.load_state_dict(torch.load(finetune_dir / "dec_opt.pth"))

        e = int(finetune.split('_')[0][:-3])

    # Identifier directory for model
    timestamp = dt.datetime.now().strftime("%d%m%y_%H%M")
    identifier = f"{e+num_epochs}eps_{'_'.join(classes)}_{timestamp}"

    # Set up directories
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    output_dir = pathlib.Path(output_dir) / identifier
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # TensorBoard setup
    writer = SummaryWriter(log_dir / identifier)
    writer.add_graph(model, next(iter(test_loader)).to(device))

    # Run the training and validation loops
    for epoch in range(num_epochs):
        print()

        epoch_losses = train_loop(
            train_loader,
            model,
            loss_fn,
            [enc_optimizer, dec_optimizer],
            epoch,
            clip_gradients,
            DataAugmentation()
        )
        valid_losses = eval_loop(
            valid_loader,
            model,
            loss_fn
        )
        
        epoch_df = pd.DataFrame.from_records(epoch_losses)
        for key, value in epoch_df.mean().iteritems():
            writer.add_scalar(f"train/{key}", value, global_step=epoch+1)

        valid_df = pd.DataFrame.from_records(valid_losses)
        for key, value in valid_df.mean().iteritems():
            writer.add_scalar(f"valid/{key}", value, global_step=epoch+1)

        # Draw a progress reconstruction
        if draw_every != 0 and ((epoch+1) % draw_every) == 0:
            test_example = next(iter(test_loader))[0]
            org_img = strokes_to_rgb(to_stroke_3(test_example.cpu()))
            rec_img = np.zeros_like(org_img)
            try:
                rec = model.conditional_sample(test_example.unsqueeze(0).to(device))
                rec_img = strokes_to_rgb(to_stroke_3(rec.cpu()))
            except:
                pass  # Sometimes the sampling fails early on
            
            writer.add_image("conditional sampling/original", org_img, global_step=epoch+1, dataformats='HWC')
            writer.add_image("conditional sampling/reconstruction", rec_img, global_step=epoch+1, dataformats='HWC')
        
        # Save a model checkpoint
        if save_every != 0 and ((epoch+1) % save_every) == 0:
            torch.save(model, checkpoint_dir / f"e{epoch+1}_sketchrnn.pth")
            torch.save(model.encoder.state_dict(), checkpoint_dir / f"e{epoch+1}_encoder.pth")
            torch.save(model.decoder.state_dict(), checkpoint_dir / f"e{epoch+1}_decoder.pth")
            torch.save(enc_optimizer.state_dict(), checkpoint_dir / f"e{epoch+1}_enc_opt.pth")
            torch.save(dec_optimizer.state_dict(), checkpoint_dir / f"e{epoch+1}_dec_opt.pth")
    
    # Final test score
    test_losses = eval_loop(
        test_loader,
        model,
        loss_fn
    )
    test_df = pd.DataFrame.from_records(test_losses)
    writer.add_hparams({
        "Eh": enc_hidden,
        "Nz": Nz,
        "Dh": dec_hidden,
        "lr": learning_rate,
        "M": num_mixtures
    }, {f"hparams/{k}": v for k, v in test_df.mean().iteritems()})

    # Save the model, encoder and decoder separately
    torch.save(model, output_dir / "sketchrnn.pth")
    torch.save(model.encoder.state_dict(), output_dir / "encoder.pth")
    torch.save(model.decoder.state_dict(), output_dir / "decoder.pth")
    torch.save(enc_optimizer.state_dict(), output_dir / "enc_opt.pth")
    torch.save(dec_optimizer.state_dict(), output_dir / "dec_opt.pth")

    writer.flush()
    writer.close()
