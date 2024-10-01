from tqdm import tqdm
from typing import Tuple
from torch.distributions import MultivariateNormal

import torch
from torch import Tensor

from utils.im_invp_utils import InverseProblem


def load_g_epsilon_net(mean: Tensor, cov: Tensor, n_steps: int):
    timesteps = torch.linspace(0, 999, n_steps).long()
    alphas_cumprod = torch.linspace(0.9999, 0.98, 1000)
    alphas_cumprod = torch.cumprod(alphas_cumprod, 0).clip(1e-10, 1)
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

    epsilon_net = EpsilonNetG(
        mean, cov, alphas_cumprod=alphas_cumprod, timesteps=timesteps
    )

    return epsilon_net


class EpsilonNetG:
    def __init__(
        self, mean: Tensor, cov: Tensor, alphas_cumprod: Tensor, timesteps: Tensor
    ):
        self.mean = mean
        self.cov = cov
        self.inv_cov = torch.inverse(cov)
        self.distribution = MultivariateNormal(mean, cov)

        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps
        self.I = torch.eye(len(self.mean))

    def forward(self, x, t):
        acp_t = self.alphas_cumprod[t]
        return -(1 - acp_t).sqrt() * self.score(x, t)

    def score(self, x, t):
        I = self.I
        mean, cov = self.mean, self.cov
        acp_t = self.alphas_cumprod[t]

        cov_t = (1 - acp_t) * I + acp_t * cov
        return -torch.linalg.solve(cov_t, (x - acp_t.sqrt() * mean))

    def predict_x0(self, x, t):
        acp_t = self.alphas_cumprod[t]
        return (x + (1 - acp_t) * self.score(x, t)) / acp_t.sqrt()

    def cov_x0(self, x, t):
        # computes `inverse((acp_t / (1 - acp_t)) * I + inverse(cov))`
        # without explicitly inverting cov
        acp_t = self.alphas_cumprod[t]
        return torch.linalg.solve(self.I + (acp_t / (1 - acp_t)) * self.cov, self.cov)

    def ideal_sample(self, n_samples=1):
        """Sample from the distribution using torch."""
        samples = self.distribution.sample((n_samples,))

        if n_samples == 1:
            return samples[0]

        return samples


class ConditionalEpsilonNetG(EpsilonNetG):
    """EpsilonNet with Gaussian prior conditioned through an Inverse Problem.

    Prior is ``N(mean, Cov)`` and the inverse problem is ``y = A x + obs_std * n``.
    """

    def __init__(
        self,
        mean: Tensor,
        cov: Tensor,
        alphas_cumprod: Tensor,
        timesteps: Tensor,
        inverse_prob: InverseProblem,
    ):
        A, obs, obs_std = inverse_prob.A, inverse_prob.obs, inverse_prob.std

        inv_cov_prior = torch.linalg.inv(cov)
        inv_Sigma = inv_cov_prior + A.T @ A / obs_std**2
        Sigma = inv_Sigma.inverse()

        m = A.T @ obs / obs_std**2 + inv_cov_prior @ mean

        super().__init__(Sigma @ m, Sigma, alphas_cumprod, timesteps)


class PosteriorLinearInvProb(MultivariateNormal):
    """Posterior distribution of Linear inverse problem under Gaussian prior.

    Defines ``p(x | obs)``, where::

        obs = A x + obs_std * n
        x ~ N(mean, cov)
    """

    def __init__(self, A, obs, obs_std, epsilon_net: EpsilonNetG):
        self.epsilon_net = epsilon_net

        mean, cov = epsilon_net.mean, epsilon_net.cov
        inv_cov = torch.inverse(cov)

        inv_Sigma = inv_cov + A.T @ A / obs_std**2
        Sigma = inv_Sigma.inverse()

        m = A.T @ obs / obs_std**2 + inv_cov @ mean

        super().__init__(loc=Sigma @ m, covariance_matrix=Sigma)

    def get_marginal_t(self, t):
        """Get p(x_t | obs)"""
        I = torch.eye(len(self.mean))
        acp_t = self.epsilon_net.alphas_cumprod[t]

        return MultivariateNormal(
            acp_t * self.mean, (1 - acp_t) * I + self.covariance_matrix
        )


def bridge_kernel_all_stats(
    ell: int,
    t: int,
    s: int,
    epsilon_net: EpsilonNetG,
    eta: float = 1.0,
) -> Tuple[float, float, float]:
    """s < t < ell

    Return
    ------
    coeff_xell, coeff_xs, std
    """
    alpha_cum_s_to_t = epsilon_net.alphas_cumprod[t] / epsilon_net.alphas_cumprod[s]
    alpha_cum_t_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[t]
    alpha_cum_s_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[s]
    std = (
        eta
        * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
        ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t**0.5) - coeff_xell * (alpha_cum_s_to_ell**0.5)
    return coeff_xell, coeff_xs, std


def compute_kl(dist_1: MultivariateNormal, dist_2: MultivariateNormal):
    """Compute between two Multivariate Gaussian KL(dist_1 || dist_2)."""

    # XXX Analytical implementation of the KL between two Multivariate Gaussian
    # mean_1, mean_2 = dist_1.mean, dist_2.mean
    # cov_1, cov_2 = dist_1.covariance_matrix, dist_2.covariance_matrix

    # n = len(mean_1)
    # inv_cov_2 = torch.inverse(cov_2)
    # diff_mean = mean_1 - mean_2

    # return 0.5 * (
    #     -(torch.logdet(cov_1) - torch.logdet(cov_2))
    #     + (torch.trace(inv_cov_2 @ cov_1) - n)
    #     + diff_mean @ inv_cov_2 @ diff_mean
    # )

    return torch.distributions.kl.kl_divergence(dist_1, dist_2)


def compute_W2_multivariate_normal(
    dist_1: MultivariateNormal, dist_2: MultivariateNormal
):
    mean_term = torch.norm(dist_1.mean - dist_2.mean) ** 2

    cov_1, cov_2 = dist_1.covariance_matrix, dist_2.covariance_matrix
    sqrt_cov_1 = torch_sqrtm(cov_1)

    trace_term = torch.trace(
        cov_1 + cov_2 - 2 * torch_sqrtm(sqrt_cov_1 @ cov_2 @ sqrt_cov_1)
    )

    return (mean_term + trace_term).sqrt()


def torch_sqrtm(M):
    """Compute the square root of symmetric matrix."""
    # NOTE: here, we use svd as it has proven to be more numerically stable
    # using eigh sometimes returns negative eigen value

    # XXX: implementation with eigh
    # eig_vals, P = torch.linalg.eigh(M)
    # return P @ (eig_vals.sqrt()[:, None] * P.T)

    U, S, Vh = torch.linalg.svd(M)
    return U @ (S[:, None].sqrt() * Vh)


def generate_cov(dim=2):
    """Generate a random covariance matrix."""
    M = torch.randn(size=(dim, dim))
    norm_M = M.norm(dim=0)

    # make it symmetric
    M_normalized = (1 / norm_M) * M
    cov = M_normalized @ M_normalized.T

    # make it positive definite
    eigvals = torch.linalg.eigvalsh(cov)
    cov.view(-1)[:: dim + 1] += eigvals.mean()

    return cov
