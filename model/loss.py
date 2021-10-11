"""Loss function definitions"""

import numpy as np
import torch
from utils.data import create_stroke_mask, separate_stroke_params
from utils.sampling import normalized_gmm_params
from model import device


def sketch_rnn_loss(W_kl):
    """
    Generates and reterns the main loss function for training the SketchRNN VAE.
    This loss function computes three individual losses: Lp, Ls, and Lk representing the
    pen state, stroke likelihood from the learned GMM as well as the Kullback-Leibler Divergence loss
    and returns a weighted sum as the final loss as follows Lp + Ls + W_kl * Lk.

    Args:
        W_kl: The weight to be given to the Lk term of the computed loss
    """

    def _loss_fn(X, Y):
        # Break out the learned probabilities and parameters for the latent space Z
        (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), mu, sigma_hat = Y
        gmm_params = (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

        # Break out strokes, stroke lengths and pen states
        S, Ns, p = separate_stroke_params(X)
        mask = create_stroke_mask(Ns, S.shape[1])

        # Compute and return the loss
        Lp = pen_loss(p, q)
        Ls = stroke_loss(S, mask, gmm_params)
        Lk = kullback_leibler_loss(mu, sigma_hat)
        return Lp + Ls + W_kl * Lk

    return _loss_fn


def pen_loss(p, q):
    """
    Computes the loss for the pen state categorical probabilities. Averaged across the batches.

    Args:
        p: Ground truth pen states, expected to have be (batch, seq_len, 3)
        q: Predicted categorical probabilities, expected to be (batch, seq_len, 3)
    """
    N_max = p.shape[1]
    loss = p * torch.log(q)
    loss = -1 * torch.sum(loss, dim=(1, 2)) / N_max
    return torch.mean(loss)


def bivariate_normal_pdf(mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    """
    Returns a callable for evaluating a bivariate normal PDF with given parameters at some point (x, y)
    
    Args:
        mu_x: Mean value for the x dimension
        mu_y: Mean value for the y dimension
        sigma_x: Standard deviation along the x dimension
        sigma_y: Standard deviation along the y dimension
        rho_xy: Covariance between the x and y dimension
    """

    def _pdf(x, y):
        xx = ((x - mu_x) / sigma_x)**2
        yy = ((y - mu_y) / sigma_y)**2
        xy = (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y)
        z = xx + yy -2*rho_xy*xy

        z_exp = torch.exp(-z/(2*(1-rho_xy**2)))
        z_norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy**2)

        return z_exp / z_norm
    
    return _pdf


def stroke_loss(S, mask, params):
    """
    Computes the reconstruction loss of the GMMs averaged across each batch.

    Args:
        S: The ground truth stroke data for each batch. A (batch_size, seq_len, 2) tensor.
        mask: A binary (batch_size, seq_len) tensor indicating which strokes should be counted as part of the sketch.
        params: A tuple of tensors (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy) each being (batch_size, num_mixtures, seq_len)
    """
    N_max = S.shape[1]

    # These need to be (batch_size, seq_len, 1)
    dx = S[..., 0].unsqueeze(-1)
    dy = S[..., 1].unsqueeze(-1)

    # Evaluate the PDF for each mixture component in each stroke and batch
    pdf = bivariate_normal_pdf(*params[1:])
    likelihoods = pdf(dx, dy)

    # Compute and return the loss
    loss = torch.sum(mask*torch.log(1e-5 + torch.sum(params[0] * likelihoods, dim=2)), dim=1)
    loss /= (-1 * N_max)
    return torch.mean(loss)


def kullback_leibler_loss(mu, sigma_hat):
    """Computes and returns the Kullback-Leibler loss against a standard normal distribution"""
    Nz = mu.shape[1]
    loss = 1 + sigma_hat - torch.pow(mu, 2) - torch.exp(sigma_hat)
    loss /= -1 * Nz
    return torch.sum(loss)
