"""Model components for the Sketch-RNN VAE"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling import sample_normal


class Encoder(nn.Module):
    """Encodes an input set of strkoes to a latent representation h"""

    def __init__(self, hidden_size, z_dims, dropout, num_layers=1):
        """
        Initialize the VAE encoder block. The input data is expected to have
        5 features (2 for the x, y offsets and 3 for the pen states).

        Note that the dimensions of h will likely be double that of the supplied
        number due to the bidirectionality of the LSTM module.

        Args:
            hidden_size: The number of dimensions for the latent space H
            z_dims: The number of dimensions for the vectors Z
            dropout: The keep probability for random dropout regularization during training
            num_layers: Numbers of stacked LSTMs, defaults to 1
        """
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(dropout)
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
        _, (h, _) = self.lstm(x)
        h_forward, h_backward = torch.split(h, 1)
        y = torch.cat([h_forward.squeeze(0), h_backward.squeeze(0)], 1)
        y = self.dropout(y)
        mu = self.h2mu(y)
        sigma_hat = self.h2sigma(y)
        sigma = torch.exp(sigma_hat / 2.0)
        z = sample_normal(mu, sigma)
        return z, mu, sigma_hat


class Decoder(nn.Module):
    """Decodes a given latent space seed z into a fixed-length sequence of mixture model parameters"""

    def __init__(self, z_dims, hidden_size, dropout, num_mixtures, num_layers=1):
        """
        Constructs the VAE decoder block

        Args:
            z_dims: The number of dimensions of the latent Z vector
            hidden_size: The size of the hidden vector of the decoder LSTM
            dropout: The keep probability for random dropout regularization during training
            num_mixtures: The number of Gaussian mixtures to inlcude in the sampling scheme
            num_layers: Numbers of stacked LSTMs, defaults to 1
        """
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures
        self.t = 1.0  # Temperature parameter should be 1.0 by default

        self.dropout = nn.Dropout(dropout)
        self.z2hc = nn.Linear(
            in_features=z_dims,
            out_features=2*hidden_size
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

    def forward(self, z, S, h_c=None):
        """
        Runs the forward propagation step for one batch through the decoder

        TODO: This should have different behaviour when self.training is False

        Args:
            z: A sampled latent vector of size (batch_size, z_dims)
            S: Ground truth padded stroke-5 data of size including the initial start of
                sequence vector. Should be (batch_size, seq_len+1, 5)
            h_c: Optional values for the hidden and cell states for the LSTM, should be given as a tuple.
                These can be used for sampling from a trained model
        """
        if h_c is None:  # Initialize from z
            hc = self.z2hc(z)
            hc = self.dropout(hc)
            hc = torch.tanh(hc)
            hc = torch.stack([hc]*self.num_layers, dim=0)  # Needs to be (num_layers, batch_size, hidden_size)
            h, c = torch.split(hc, self.hidden_size, dim=2)
            h_c = (h.contiguous(), c.contiguous())

        z_stack = torch.stack([z] * S.shape[1], dim=1)
        x = torch.cat([S, z_stack], dim=2)
        y, h_c = self.lstm(x, h_c)
        y = self.dropout(y)
        params = self.y2params(y)

        # Separate into parameters for GMM and pen
        pi_hat, mu_x, mu_y, sigma_x_hat, sigma_y_hat, rho_xy, q_hat = torch.split(params, self.num_mixtures, dim=2)
        
        # Normalized GMM and pen parameters
        pi = F.softmax(pi_hat / self.t, dim=2)
        sigma_x = torch.exp(sigma_x_hat) * np.sqrt(self.t)
        sigma_y = torch.exp(sigma_y_hat) * np.sqrt(self.t)
        rho_xy = torch.tanh(rho_xy)
        q = F.softmax(q_hat / self.t, dim=2)

        return (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), h_c
    