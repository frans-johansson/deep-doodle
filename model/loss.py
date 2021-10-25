"""Loss function definitions"""

import numpy as np
import torch
from utils.data import create_stroke_mask, separate_stroke_params
from model import device


def sketch_rnn_loss(W_kl, kl_min, eta_min, R):
    """
    Generates and reterns the main loss function for training the SketchRNN VAE.
    This loss function computes three individual losses: Lp, Ls, and Lk representing the
    pen state, stroke likelihood from the learned GMM as well as the Kullback-Leibler Divergence loss
    and returns a weighted sum as the final loss as follows Lp + Ls + W_kl * Lk, as well as
    a dictionary of individual losses for logging.

    Args:
        W_kl: The weight to be given to the Lk term of the computed loss
        kl_min: Floor for the KL divergence loss term
        eta_min: Minimum value for eta in KL annealing
        R: Stepping parameter for KL annealing
    """
    step = 0  # This will serve as a semi-global variable keeping track of training steps

    def _loss_fn(X, Y, training=True):
        loss_dict = {}

        # Break out the learned probabilities and parameters for the latent space Z
        (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), mu, sigma_hat = Y
        gmm_params = (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

        # Break out strokes, stroke lengths and pen states
        S, Ns, p = separate_stroke_params(X)
        mask = create_stroke_mask(Ns, S.shape[1])

        # Compute and return the loss
        Lp = pen_loss(p, q)
        Ls = stroke_loss(S, mask, gmm_params)
        Lk = kullback_leibler_loss(mu, sigma_hat, kl_min)

        loss_dict.setdefault("Lp", float(Lp.item()))
        loss_dict.setdefault("Ls", float(Ls.item()))
        loss_dict.setdefault("Lk", float(Lk.item()))

        if training:
            # KL annealing computation
            nonlocal step
            step += 1
            eta = 1.0 - (1.0 - eta_min) * R**step
            Lk *= eta
            loss_dict.setdefault("Lk*eta", float(Lk.item()))

        loss = Lp + Ls + W_kl * Lk
        loss_dict.setdefault("loss", float(loss.item()))
        return loss, loss_dict

    return _loss_fn


def pen_loss(p, q):
    """
    Computes the loss for the pen state categorical probabilities. Averaged across the batches.

    Args:
        p: Ground truth pen states, expected to have be (batch, seq_len, 3)
        q: Predicted categorical probabilities, expected to be (batch, seq_len, 3)
    """
    N_max = p.shape[1]
    loss = p * torch.log(q + 1e-5)
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
        z_norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy**2 + 1e-5)

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


def kullback_leibler_loss(mu, sigma_hat, kl_min):
    """Computes and returns the Kullback-Leibler loss against a standard normal distribution"""
    loss = -torch.sum(1 + sigma_hat - mu**2 - torch.exp(sigma_hat)) \
        / (2. * mu.shape[0] * mu.shape[1])

    # Old implementation
    # Nz = mu.shape[1]
    # loss = 1 + sigma_hat - torch.pow(mu, 2) - torch.exp(sigma_hat)
    # loss /= -2 * Nz
    # loss = torch.mean(loss) 

    loss_min = torch.tensor(kl_min, device=device)
    return torch.max(loss, loss_min)
