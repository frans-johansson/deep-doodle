"""Model components for the Sketch-RNN VAE"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling import normalized_gmm_params, sample_normal


class Encoder(nn.Module):
    """Encodes an input set of strkoes to a latent representation h"""

    def __init__(self, hidden_size, z_dims, num_layers, dropout):
        """
        Initialize the VAE encoder block. The input data is expected to have
        5 features (2 for the x, y offsets and 3 for the pen states).

        Note that the dimensions of h will likely be double that of the supplied
        number due to the bidirectionality of the LSTM module.

        Args:
            hidden_size: The number of dimensions for the latent space H
            z_dims: The number of dimensions for the vectors Z
            num_layers: Numbers of stacked LSTMs
            dropout: The keep probability for random dropout regularization during training
        """
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.h2mu = nn.Linear(in_features=hidden_size*2, out_features=z_dims)
        self.h2sigma = nn.Linear(in_features=hidden_size*2, out_features=z_dims)

    def forward(self, x):
        y, _ = self.lstm(x)
        y = y[:, -1, :]  # Only care about the last output
        mu = self.h2mu(y)
        sigma_hat = self.h2sigma(y)
        sigma = torch.exp(sigma_hat)
        z = sample_normal(mu, sigma)
        return z, mu, sigma_hat


class Decoder(nn.Module):
    """Decodes a given latent space seed z into a fixed-length sequence of mixture model parameters"""

    def __init__(self, z_dims, hidden_size, num_layers, dropout, num_mixtures):
        """
        Constructs the VAE decoder block

        Args:
            z_dims: The number of dimensions of the latent Z vector
            hidden_size: The size of the hidden vector of the decoder LSTM
            num_layers: Numbers of stacked LSTMs
            dropout: The keep probability for random dropout regularization during training
            num_mixtures: The number of Gaussian mixtures to inlcude in the sampling scheme
        """
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers

        self.z2h = nn.Linear(
            in_features=z_dims,
            out_features=hidden_size
        )
        self.lstm = nn.LSTM(
            input_size=5+z_dims,  # x = [S, z]
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.y2params = nn.Linear(
            in_features=hidden_size,
            out_features=num_mixtures*6 + 3 
            # Each y has the parameters for a number of Gaussian mixtures
            # as well as a categorical distribution with three categories
        )

    def forward(self, z, S):
        """
        Runs the forward propagation step for one batch through the decoder

        Args:
            z: A sampled latent vector of size (batch_size, z_dims)
            S: Ground truth padded stroke-5 data of size including the initial start of
                sequence vector. Should be (batch_size, seq_len+1, 5)
        """
        h = self.z2h(z)
        h = torch.tanh(h)
        h = torch.stack([h]*self.num_layers, dim=0)  # Needs to be (num_layers, batch_size, hidden_size)
        c = torch.zeros_like(h)
        z_stack = torch.stack([z] * S.shape[1], dim=1)

        x = torch.cat([S, z_stack], dim=2)
        y, _ = self.lstm(x, (h, c))
        params = self.y2params(y)

        # Separate into parameters for GMM and pen
        split_params = torch.split(params, 6, dim=2)
        gmm_params = torch.stack(split_params[:-1], dim=1)  # (batch, mixture, stroke, 6)
        pen_params = split_params[-1]  # (batch, stroke, 3)
        
        # Grab separate normalized GMM parameters
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = normalized_gmm_params(gmm_params[:, :, :-1, :])

        # Normalize the pen parameters
        q = F.softmax(pen_params[:, :-1, :], dim=1)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q
