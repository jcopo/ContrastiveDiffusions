import pdb
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, PyTreeDef

from diffuse.sde import SDE


class MixState(NamedTuple):
    """
    Represents the state of the mixture optimization algorithm.

    Attributes:
        means (PyTreeDef): The means of the mixture components.
        cov (PyTreeDef): Covariances components of the mixture
        mix_weights (PyTreeDef): The mixture weights.
        grad_state (GradState, optional): The gradient state. Defaults to GradState().
        info (INFO, optional): Hyperparameters
    """

    means: PyTreeDef
    cov: PyTreeDef
    mix_weights: PyTreeDef


def cdf_mixtr(mix_state: MixState, x: Array) -> Array:
    """
    Calculate the cumulative distribution function (CDF) of a mixture of Gaussian distributions.

    Args:
        mix_state (MixState): The state of the mixture model, including means, covariances,
                              and mixture weights.
        x (jnp.ndarray): The input values at which to evaluate the CDF.

    Returns:
        jnp.ndarray: The CDF values of the mixture distribution at the input points.

    Note:
        This function assumes that the mixture components are univariate Gaussian distributions.
        It uses jax.scipy.stats.norm.cdf for calculating individual CDFs.
    """
    means, covs, weights = mix_state
    stds = jnp.sqrt(covs)

    def single_cdf(mean, std, weight):
        return weight * jax.scipy.stats.norm.cdf((x - mean) / std)

    cdfs = jax.vmap(single_cdf)(means, stds, weights)
    return jnp.sum(cdfs, axis=0).squeeze()


def pdf_mixtr(mix_state: MixState, x: Array) -> Array:
    """
    Calculate the probability density function (PDF) of a multivariate normal distribution
    mixture given a state and input data.

    Args:
        state (MixState): The state of the mixture model, including means, cholesky factors,
                          mixture weights, and other parameters.
        x (Array): The input data.

    Returns:
        float: The PDF of the multivariate normal distribution mixture.
    """
    means, sigmas, weights = mix_state

    def pdf_multivariate_normal(mean, sigma):
        return jax.scipy.stats.multivariate_normal.pdf(x, mean, sigma)

    pdf = jax.vmap(pdf_multivariate_normal)(means, sigmas)
    return weights @ pdf


def rho_t(x: Array, t: Array, init_mix_state: MixState, sde: SDE) -> Array:
    """
    Compute p_t(x_t) where x_t follows the noising process defined by sde
    """
    means, covs, weights = transform_mixture_params(init_mix_state, sde, t)
    return pdf_mixtr(MixState(means, covs, weights), x)


def cdf_t(x: Array, t: Array, init_mix_state: MixState, sde: SDE) -> Array:
    """
    Compute cdf_t(x_t) where x_t follows the noising process defined by sde
    """
    means, covs, weights = transform_mixture_params(init_mix_state, sde, t)
    return cdf_mixtr(MixState(means, covs, weights), x)


def init_mixture(key, d=1):
    n_mixt = 3
    means = jax.random.uniform(key, (n_mixt, d), minval=-3, maxval=3)
    chol = jax.random.normal(key + 1, (n_mixt, d, d))
    covs = 0.1 * (chol @ chol.transpose(0, 2, 1))
    mix_weights = jax.random.uniform(key + 2, (n_mixt,))
    mix_weights /= jnp.sum(mix_weights)

    return MixState(means, covs, mix_weights)


def sampler_mixtr(key, state: MixState, N):
    """
    Sampler from the mixture
    """
    mu, sigma, weights = state
    d = mu.shape[-1]
    key1, key2 = jax.random.split(key)
    idx = jax.random.choice(key1, jnp.arange(len(weights)), shape=(N,), p=weights)
    noise = jax.random.normal(key2, shape=(N, d))

    chol = jnp.linalg.cholesky(sigma)
    noise_scaled = jnp.einsum("nij, ni->nj", chol[idx], noise)

    return mu[idx] + noise_scaled


xmax = 4
nbins = 200


def transform_mixture_params(state, sde, t):
    """
    Close form solution of VP SDE for Gaussian Mixture
    """
    means, covs, weights = state
    int_b = sde.beta.integrate(t, 0.0)
    alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
    means = alpha * means
    covs = alpha**2 * covs + beta * jnp.eye(covs.shape[-1])
    return means, covs, weights


def display_histogram(samples, ax):
    nb = samples.flatten().shape[0]
    h0, b = jnp.histogram(samples.flatten(), bins=nbins, range=[-xmax, xmax])
    h0 = h0 / nb * nbins / (2 * xmax)
    ax.bar(
        jnp.linspace(-xmax, xmax, nbins),
        h0,
        width=2 * xmax / (nbins - 1),
        align="center",
        color="red",
    )


def display_trajectories(Y, m):
    """
    Color shading to show where particles ends up
    m: number of trajectories to plot
    """
    P, N = Y.shape
    idxs = jnp.round(jnp.linspace(0, P - 1, m)).astype(jnp.int32)
    sorted_idx = jnp.argsort(Y[:, -1])
    I = sorted_idx[idxs]
    for i, idx in enumerate(I):
        color_marker = i / (m - 1)  # trcks where particle ends up

        plt.plot(
            Y[idx, :], c=[color_marker, 0, 1 - color_marker], alpha=0.3, linewidth=0.5
        )
