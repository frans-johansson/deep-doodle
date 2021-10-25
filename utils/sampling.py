import torch
from numpy.random import default_rng
from model import device

rng = default_rng()


def sample_normal(mu, sigma):
    """Gradient-friendly way to sample a normal vector from an n-dimensional normal distribution"""
    assert (
        mu.shape == sigma.shape
    ), "Mean and standard deviation must have the same length"
    # N = torch.from_numpy(rng.normal(0, 1, size=(mu.shape))).to(torch.float32).to(device)
    N = torch.normal(torch.zeros_like(mu), torch.ones_like(mu)).to(device)
    return mu + sigma * N


def _sample_stroke_offset(mu, sigma, rho):
    """
    Sample the stroke offsets dx and dy from a multivariate normal distribution.

    Args:
        mu: A two-dimensional vector of mean values for dx and dy
        sigma: A two-dimensional vector of standard deviation values for dx and dy
        rho: A floating point value for the covariance between dx and dy
    """
    cov = torch.diag(sigma).to(device)
    cov[1, 0] = cov[0, 1] = rho
    # s = rng.multivariate_normal(mean=mu.cpu().detach(), cov=cov.cpu().detach())
    s = torch.distributions.MultivariateNormal(loc=mu, scale_tril=torch.tril(cov)).sample()
    # return torch.from_numpy(s)
    return s


def _sample_pen_state(q):
    """
    Samples the one-hot encoded pen state for the stroke-5 representation of pen strokes

    Args:
        q: Categorical probability values (q1, q2, q3) for each of the outcomes of p
    """
    ohc = torch.distributions.OneHotCategorical(probs=q)
    return ohc.sample().to(device)


def sample_stroke(weights, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
    """
    Samples a single stroke offset dx and dy from a GMM with M mixture components, and
    a one-hot encoded pen state from a categorical distribution of three possible values. 

    Args:
        weights: An M dimensional tensor of mixture weights
        mu_x, mu_y, sigma_x, sigma_y, rho_xy: Each an M dimensional tensor with parameters for
            each of the mixture components
        q: A 3 dimensional tensor of categorical probabilities for the pen states

    Returns:
        A 5 dimensional tensor containing [dx, dy, p1, p2, p3], with dx and dy being stroke offsets
        and each p representing a different pen state (down, up and end-of-sequence).
    """
    # Select one of the mixtures by their weights
    c = torch.distributions.Categorical(probs=weights)
    m = c.sample()

    # Handle the sampling
    mu = torch.stack([mu_x[m], mu_y[m]])
    sigma = torch.stack([sigma_x[m], sigma_y[m]])
    
    stroke = _sample_stroke_offset(
        mu, sigma, rho_xy[m]
    )
    pen = _sample_pen_state(q)

    # Concatenate into one stroke-5 tensor
    return torch.cat([stroke, pen]).to(device)


def sample_sketch(params):
    """
    Samples a complete sketch sequence from the parameters output by e.g. the SkethRNN model
    
    Args:
        params: A tuple of (weights, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), each except q being
            a (batch, stroke, mixture) tensor with q being a (batch, stroke, 3) tensor
    """
    num_batches = params[0].shape[0]
    batches = []

    for batch in range(num_batches):
        batch_params = tuple([param[batch] for param in params])
        strokes = [
            sample_stroke(weights, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
            for weights, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q in zip(*batch_params)
        ]

        batches.append(torch.stack(strokes))
    
    return torch.stack(batches)

