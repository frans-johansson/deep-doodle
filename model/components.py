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
        self.t = 1.0  # Temperature parameter should be 1.0 by default

        self.z2h = nn.Linear(
            in_features=z_dims,
            out_features=hidden_size
        )
        self.lstm =  nn.LSTM(
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
            h = self.z2h(z)
            h = torch.tanh(h)
            h = torch.stack([h]*self.num_layers, dim=0)  # Needs to be (num_layers, batch_size, hidden_size)
            c = torch.zeros_like(h)
        else:
            h, c = h_c

        z_stack = torch.stack([z] * S.shape[1], dim=1)
        x = torch.cat([S, z_stack], dim=2)
        y, h_c = self.lstm(x, (h, c))
        params = self.y2params(y)

        # Separate into parameters for GMM and pen
        split_params = torch.split(params, 6, dim=2)
        gmm_params = torch.stack(split_params[:-1], dim=1)  # (batch, mixture, stroke, 6)
        pen_params = split_params[-1]  # (batch, stroke, 3)
        
        # Grab separate normalized GMM parameters
        # pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = normalized_gmm_params(gmm_params[:, :, :-1, :])
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = self._normalized_gmm_params(gmm_params)

        # Normalize the pen parameters
        # q = F.softmax(pen_params[:, :-1, :], dim=2)
        q = self._normalized_pen_params(pen_params)

        return (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), h_c
    
    def _normalized_pen_params(self, params):
        """Returns a tensor of normalized pen state probabilities scaled by some temperature value"""
        return F.softmax(params / self.t, dim=2)

    def _normalized_gmm_params(self, params):
        """
        Returns a tuple of normalized GMM parameters given the unnormalized output of the SketchRNN model.

        Args:
            params: The unnormalized GMM parameters output by e.g. the SketchRNN model. Should have shape
                (batch, num_mixtures, seq_len, 6)

        Returns:
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy
        """
        pi = F.softmax(params[..., 0] / self.t , dim=1).transpose(1, 2)
        mu_x = params[..., 1].transpose(1, 2)
        mu_y = params[..., 2].transpose(1, 2)
        sigma_x = torch.exp(params[..., 3]).transpose(1, 2) * np.sqrt(self.t)
        sigma_y = torch.exp(params[..., 4]).transpose(1, 2) * np.sqrt(self.t)
        rho_xy = torch.tanh(params[..., 5]).transpose(1, 2)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy

    
