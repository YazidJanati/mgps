import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import torch
from torch.func import grad
from torch.distributions import (
    Distribution,
    MixtureSameFamily,
    Categorical,
    MultivariateNormal,
)


def generate_inverse_problem(
    prior: Distribution, dim: int, std_y: float, A: torch.Tensor = None
):

    if A is None:
        A = torch.randn((1, dim))
    obs = A @ prior.sample() + std_y * torch.randn((A.shape[0],))
    posterior = get_posterior(obs, prior, A, std_y)

    return obs, A, posterior


def gmm_25(dim: int):

    means = torch.tensor(
        [[8 * i, 8 * j] * (dim // 2) for i in range(-2, 3) for j in range(-2, 3)]
    ).float()
    n_mixtures = means.shape[0]
    covs = torch.eye(dim)[None, :].repeat(n_mixtures, 1, 1)
    weights = torch.rand(n_mixtures)
    weights = weights / weights.sum()

    return MixtureSameFamily(Categorical(weights), MultivariateNormal(means, covs))


def gmm_2(dim: int, loc0: float, loc1: float):
    means = torch.tensor([[loc0] * dim, [loc1] * dim])
    covs = torch.eye(dim)[None, :].repeat(2, 1, 1)
    weights = torch.rand(2)
    weights = weights / weights.sum()
    return MixtureSameFamily(Categorical(weights), MultivariateNormal(means, covs))


def sliced_wasserstein(
    dist_1: torch.Tensor, dist_2: torch.Tensor, n_slices: int = 100
) -> float:
    """Compute sliced Wasserstein distance between two distributions.

    Assumes that both ``dist_1`` and ``dist_2`` have the same dimension.

    Parameters
    ----------
    dist_1 : Tensor

    dist_2 : Tensor

    n_slices : int, default=100
        The number of the considered random projections.

    Return
    ------
    sw_distance : float
    """
    if dist_1.ndim > 2:
        dist_1 = dist_1.reshape(dist_1.shape[0], -1)
        dist_2 = dist_2.reshape(dist_2.shape[0], -1)

    projections = torch.randn(size=(n_slices, dist_1.shape[1]), device=dist_1.device)
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = projections @ dist_1.T
    dist_2_projected = projections @ dist_2.T

    dist_1_projected = dist_1_projected.cpu().numpy()
    dist_2_projected = dist_2_projected.cpu().numpy()

    return np.mean(
        [
            wasserstein_distance(u_values=d1, v_values=d2)
            for d1, d2 in zip(dist_1_projected, dist_2_projected)
        ]
    )


def generate_linear_inverse_problem(
    prior: Distribution, dim: int, std_y: float, A: torch.Tensor = None
):

    if A is None:
        A = torch.randn((1, dim))
    obs = A @ prior.sample() + std_y * torch.randn((A.shape[0],))
    posterior = get_posterior(obs, prior, A, std_y)
    log_pot = lambda x: -0.5 * torch.norm(obs - x @ A.T) ** 2 / std_y**2

    return obs.flatten(), GeneralH(H=A), posterior, log_pot


def get_posterior(obs, prior, A, noise_std):
    modified_means = []
    modified_covars = []
    weights = []

    assert obs.ndim == 1

    for loc, cov, log_weight in zip(
        prior.component_distribution.loc,
        prior.component_distribution.covariance_matrix,
        prior.mixture_distribution.logits,
    ):
        new_dist = gaussian_posterior(
            obs,
            A,
            torch.zeros_like(obs),
            torch.eye(obs.shape[0]) / (noise_std**2),
            loc,
            cov,
        )
        modified_means.append(new_dist.loc)
        modified_covars.append(new_dist.covariance_matrix)
        prior_x = MultivariateNormal(loc=loc, covariance_matrix=cov)
        log_constant = (
            -torch.linalg.norm(obs - A @ new_dist.loc) ** 2 / (2 * noise_std**2)
            + prior_x.log_prob(new_dist.loc)
            - new_dist.log_prob(new_dist.loc)
        )
        weights.append(log_weight + log_constant)
    weights = torch.tensor(weights)
    # weights = weights / weights.sum()
    weights = weights.softmax(0)
    cat = Categorical(weights)
    ou_norm = MultivariateNormal(
        loc=torch.stack(modified_means, dim=0),
        covariance_matrix=torch.stack(modified_covars, dim=0),
    )
    return MixtureSameFamily(cat, ou_norm)


def gaussian_posterior(
    y, likelihood_A, likelihood_bias, likelihood_precision, prior_loc, prior_covar
):
    prior_precision_matrix = torch.linalg.inv(prior_covar)
    posterior_precision_matrix = (
        prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
    )

    # posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)

    # NOTE: use cholesky to invert matrix to avoid numerical errors
    # that results in dissymmetry of the covariance
    posterior_covariance_matrix = torch.cholesky_inverse(
        torch.linalg.cholesky(posterior_precision_matrix)
    )
    posterior_mean = posterior_covariance_matrix @ (
        likelihood_A.T @ likelihood_precision @ (y - likelihood_bias)
        + prior_precision_matrix @ prior_loc
    )

    # posterior_mean = torch.linalg.solve(
    #     posterior_precision_matrix,
    #     likelihood_A.T @ likelihood_precision @ (y - likelihood_bias)
    #     + prior_precision_matrix @ prior_loc,
    # )

    posterior_covariance_matrix += 1e-3 * torch.eye(posterior_covariance_matrix.shape[0])

    # # NOTE: due to numerical error when inverting `posterior_precision_matrix`
    # # it might lose its symmetry (with small tolerance 1e-6) ---> force symmetric here
    posterior_covariance_matrix = (
        posterior_covariance_matrix.T + posterior_covariance_matrix
    ) / 2

    return MultivariateNormal(
        loc=posterior_mean, covariance_matrix=posterior_covariance_matrix
    )


def bwt_gm(x_t, t, s, prior, alphas_cumprod):
    ratio_acp = alphas_cumprod[t] / alphas_cumprod[s]
    prior_s = fwm_gm(prior, s, alphas_cumprod)
    return get_posterior(
        x_t, prior_s, ratio_acp.sqrt() * torch.eye(x_t.shape[0]), (1 - ratio_acp).sqrt()
    )


def fwm_gm(prior, t, alphas_cumprod):
    means, covs, weights = (
        prior.component_distribution.mean,
        prior.component_distribution.covariance_matrix,
        prior.mixture_distribution.probs,
    )
    return fwd_mixture(means, weights, alphas_cumprod, t, covs)


def fwd_mixture(
    means: torch.tensor,
    weights: torch.tensor,
    alphas_cumprod: torch.tensor,
    t: torch.tensor,
    covs: torch.tensor = None,
):
    n_mixtures = weights.shape[0]
    acp_t = alphas_cumprod[t]
    means = acp_t.sqrt() * means
    Id = torch.eye(means.shape[-1])[None, ...].repeat(n_mixtures, 1, 1)
    if covs is None:
        covs = Id
    else:
        covs = (1 - acp_t) * Id + acp_t * covs

    mvn = MultivariateNormal(means, covs)
    return MixtureSameFamily(Categorical(weights), mvn)


class EpsilonNetGM(torch.nn.Module):

    def __init__(self, means, weights, alphas_cumprod, cov=None):
        super().__init__()
        self.means = means
        self.weights = weights
        self.covs = cov
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x, t):
        acp_t = self.alphas_cumprod[t.to(int)]
        grad_logprob = grad(
            lambda x: fwd_mixture(
                self.means, self.weights, self.alphas_cumprod, t, self.covs
            )
            .log_prob(x)
            .sum()
        )
        return -((1 - acp_t) ** 0.5) * grad_logprob(x)


def display_samples(samples, posterior_samples):

    posterior_samples = posterior_samples.cpu()
    samples = samples.detach().cpu()
    alpha = 0.2
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
    ax[0].scatter(
        posterior_samples[:, 0],
        posterior_samples[:, 1],
        s=2.0,
        label="posterior samples",
        alpha=alpha,
    )
    ax[0].scatter(samples[:, 0], samples[:, 1], s=2.0, label="super_dps", alpha=alpha)
    ax[1].scatter(
        posterior_samples[:, 2],
        posterior_samples[:, 3],
        s=2.0,
        label="posterior samples",
        alpha=alpha,
    )
    ax[1].scatter(samples[:, 2], samples[:, 3], s=2.0, label="super_dps", alpha=alpha)

    for i in [0, 1]:
        ax[i].set_xlim(-25, 25)
        ax[i].set_ylim(-25, 25)
    plt.legend()
    plt.show()


# code copy/paste from DDRM
# https://github.com/bahjat-kawar/ddrm/blob/32b6b3ccfda532ba01c9cc5b6e7456c3a06a6ca2/functions/svd_replacement.py#L72


class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, : singulars.shape[0]] = temp[:, : singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))


# a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2:
            vshape = vshape * v.shape[2]
        if len(v.shape) > 3:
            vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape, 1)).view(
            v.shape[0], M.shape[0]
        )

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, : self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out


##########
