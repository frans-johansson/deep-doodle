import torch
import torch.nn.functional as F
from numpy.random import default_rng
from model import device

rng = default_rng()


def sample_normal(mu, sigma):
    """Gradient-friendly way to sample a normal vector from an n-dimensional normal distribution"""
    assert (
        mu.shape == sigma.shape
    ), "Mean and standard deviation must have the same length"
    N = torch.from_numpy(rng.normal(0, 1, size=(mu.shape))).to(torch.float32).to(device)
    return mu + sigma * N


def normalized_gmm_params(params):
    """
    Returns a tuple of normalized GMM parameters given the unnormalized output of the SketchRNN model.
    
    Args:
        params: The unnormalized GMM parameters output by e.g. the SketchRNN model. Should have shape
            (batch, num_mixtures, seq_len, 6)

    Returns:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy
    """
    pi = F.softmax(params[..., 0], dim=1).transpose(1, 2)
    mu_x = params[..., 1].transpose(1, 2)
    mu_y = params[..., 2].transpose(1, 2)
    sigma_x = torch.exp(params[..., 3]).transpose(1, 2)
    sigma_y = torch.exp(params[..., 4]).transpose(1, 2)
    rho_xy = torch.tanh(params[..., 5]).transpose(1, 2)
    return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy


def sample_stroke_offset(mu, sigma, rho):
    """
    Sample the stroke offsets dx and dy from a multivariate normal distribution.

    Args:
        mu: A two-dimensional vector of mean values for dx and dy
        sigma: A two-dimensional vector of standard deviation values for dx and dy
        rho: A floating point value for the covariance between dx and dy
    """
    cov = torch.diag(sigma).to(device)
    cov[1, 0] = cov[0, 1] = rho
    s = rng.multivariate_normal(mean=mu.detach(), cov=cov.detach())
    return torch.from_numpy(s)


def sample_pen_state(q):
    """
    Samples the one-hot encoded pen state for the stroke-5 representation of pen strokes

    Args:
        q: Categorical probability values (q1, q2, q3) for each of the outcomes of p
    """
    ohc = torch.distributions.OneHotCategorical(probs=q)
    return ohc.sample().to(device)


def sample_stroke_gmm(weights, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
    """
    Samples a stroke offset dx and dy from a GMM with M mixture components.

    Args:
        TODO: Add args
    """
    # Select one of the mixtures by their weights
    c = torch.distributions.Categorical(probs=weights)
    m = c.sample()

    # Handle the sampling
    mu = torch.cat([mu_x[m], mu_y[m]])
    sigma = torch.cat([sigma_x[m], sigma_y[m]])
    
    stroke = sample_stroke_offset(
        mu, sigma, rho_xy[m]
    )
    pen = sample_pen_state(q)

    # Concatenate into one stroke-5 tensor
    return torch.cat([stroke, pen]).to(device)


def sample_sketch(params):
    """
    Samples a complete sketch sequence from the parameters output by e.g. the SkethRNN model
    
    Args:
        TODO: Add args
    """
    strokes = [
        sample_stroke_gmm(weights, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
        for weights, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q in params
    ]

