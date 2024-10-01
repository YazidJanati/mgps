from typing import Tuple
from tqdm import trange

import torch
from torch.distributions import MultivariateNormal

from utils.gauss_tmid.utils import (
    EpsilonNetG,
    ConditionalEpsilonNetG,
    bridge_kernel_all_stats,
)
from utils.im_invp_utils import InverseProblem


def conditional_ddpm(
    epsilon_net: EpsilonNetG,
    inverse_problem: InverseProblem,
    alpha: float,
    eta=1.0,
    verbose: bool = True,
):
    """"""
    # init conditional epsilon net
    epsilon_net = ConditionalEpsilonNetG(
        epsilon_net.mean,
        epsilon_net.cov,
        epsilon_net.alphas_cumprod,
        epsilon_net.timesteps,
        inverse_problem,
    )

    mean = epsilon_net.mean
    tmid_fn = lambda t: max(1, int(alpha * t))
    n_steps = len(epsilon_net.timesteps)

    I = torch.eye(len(epsilon_net.mean))
    loc = torch.zeros_like(mean)
    cov = torch.eye(len(mean), dtype=mean.dtype)

    for i in trange(n_steps - 1, 1, -1, disable=not verbose):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        t_mid = tmid_fn(t)
        t_mid = t_mid if t_mid <= t_prev else t_prev

        coeff_x_t, coeff_x_mid, std = bridge_kernel_all_stats(
            t, t_prev, s=t_mid, epsilon_net=epsilon_net, eta=eta
        )

        loc_t_t_prev, var_x_x_tprev, matrix_x_t_x_tprev = exact_conditional_backward(
            t,
            t_mid,
            x_t=loc,
            epsilon_net=epsilon_net,
            eta=eta,
        )

        loc = coeff_x_t * loc + coeff_x_mid * loc_t_t_prev

        matrix_x_t = coeff_x_t * I + coeff_x_mid * matrix_x_t_x_tprev
        cov = (
            matrix_x_t @ cov @ matrix_x_t.T
            + coeff_x_mid**2 * var_x_x_tprev
            + std**2 * I
        )

    # last diffusion step
    t_1 = epsilon_net.timesteps[1]
    jac_x_t = torch.autograd.functional.jacobian(
        lambda x: epsilon_net.predict_x0(x, t_1), torch.zeros_like(loc)
    )

    loc = epsilon_net.predict_x0(loc, t_1)
    cov = jac_x_t @ cov @ jac_x_t.T

    return loc, cov


def mgps_ddpm_dps(
    epsilon_net: EpsilonNetG,
    inverse_problem: InverseProblem,
    alpha: float,
    type_conditional="ddpm+dps",
    eta=1.0,
    verbose: bool = True,
):
    """
    Parameters
    ----------
    type_backward : str
        of the form "TYPE_BACKWARD + TYPE_POTENTIAL"
        where:
            - TYPE_BACKWARD = {'ddim', 'exact'}
            - TYPE_POTENTIAL = {'dps', 'exact}
    """
    obs, A, obs_std = inverse_problem.obs, inverse_problem.A, inverse_problem.std
    backward, potential = _get_backward_and_potential(type_conditional)

    mean = epsilon_net.mean
    tmid_fn = lambda t: max(1, int(alpha * t))
    n_steps = len(epsilon_net.timesteps)

    I = torch.eye(len(epsilon_net.mean))
    loc = torch.zeros_like(mean)
    cov = torch.eye(len(mean), dtype=mean.dtype)

    for i in trange(n_steps - 1, 1, -1, disable=not verbose):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        t_mid = tmid_fn(t)
        t_mid = t_mid if t_mid <= t_prev else t_prev

        coeff_x_t, coeff_x_mid, std = bridge_kernel_all_stats(
            t, t_prev, s=t_mid, epsilon_net=epsilon_net, eta=eta
        )

        loc_t_t_prev, var_x_x_tprev, matrix_x_t_x_tprev = conditional_backward(
            obs,
            A,
            obs_std,
            epsilon_net,
            t,
            t_mid,
            x_t=loc,
            backward_fn=backward,
            potential_fn=potential,
            eta=eta,
        )

        loc = coeff_x_t * loc + coeff_x_mid * loc_t_t_prev

        matrix_x_t = coeff_x_t * I + coeff_x_mid * matrix_x_t_x_tprev
        cov = (
            matrix_x_t @ cov @ matrix_x_t.T
            + coeff_x_mid**2 * var_x_x_tprev
            + std**2 * I
        )

    # last diffusion step
    t_1 = epsilon_net.timesteps[1]
    jac_x_t = torch.autograd.functional.jacobian(
        lambda x: epsilon_net.predict_x0(x, t_1), torch.zeros_like(loc)
    )

    loc = epsilon_net.predict_x0(loc, t_1)
    cov = jac_x_t @ cov @ jac_x_t.T

    return loc, cov


def exact_conditional_backward(
    t, t_mid, x_t, epsilon_net: ConditionalEpsilonNetG, eta=1.0
):
    r"""Compute conditional backward.

    Returns
    -------
    loc, cov, matrix_x_t
    """
    I = torch.eye(len(x_t))
    coeff_x_t, coeff_x_s, std = bridge_kernel_all_stats(
        t, t_mid, s=0, epsilon_net=epsilon_net, eta=eta
    )

    x_0_t = epsilon_net.predict_x0(x_t, t)
    loc = coeff_x_t * x_t + coeff_x_s * x_0_t

    jac_x_t = torch.autograd.functional.jacobian(
        lambda x: epsilon_net.predict_x0(x, t), torch.zeros_like(x_t)
    )
    matrix_x_t = coeff_x_t * I + coeff_x_s * jac_x_t

    return loc, std**2 * I, matrix_x_t


def conditional_backward(
    obs,
    A,
    obs_std,
    epsilon_net: EpsilonNetG,
    t,
    t_mid,
    x_t,
    backward_fn,
    potential_fn,
    eta=1.0,
):
    r"""Compute conditional backward.

    For DPS, the potential reads::

        hpot(x_s) = g(x_0_s(x_s))

    Approximate Posterior::

        p(x_s | y) \propo hpot(x_s) p_{s|t}(x_s | x_t)

    Returns
    -------
    loc, matrix_x_t, cov
    """

    # dps potential
    likelihood_A, bias, likelihood_precision = potential_fn(
        obs, A, obs_std, epsilon_net, t_mid
    )

    # backward
    prior_loc, prior_precision, matrix_x_t = backward_fn(
        t, t_mid, x_t, epsilon_net, eta=eta
    )

    # conjugate likelihood and prior
    posterior = gaussian_posterior(
        obs,
        likelihood_A,
        bias,
        likelihood_precision=likelihood_precision,
        prior_loc=prior_loc,
        prior_precision=prior_precision,
    )

    matrix_x_t = posterior.covariance_matrix @ prior_precision @ matrix_x_t

    return posterior.mean, posterior.covariance_matrix, matrix_x_t


def dps_potential_statistics(obs, A, obs_std, epsilon_net: EpsilonNetG, t_mid):
    I = torch.eye(len(obs))
    acp_t_mid = epsilon_net.alphas_cumprod[t_mid]

    cov_0_t_mid = epsilon_net.cov_x0(t_mid, t_mid)

    # potential
    A_dot_cov_0_t_mid = A @ cov_0_t_mid
    likelihood_A = (acp_t_mid.sqrt() / (1 - acp_t_mid)) * A_dot_cov_0_t_mid
    bias = A_dot_cov_0_t_mid @ torch.linalg.solve(epsilon_net.cov, epsilon_net.mean)
    likelihood_precision = (1 / obs_std**2) * I

    return likelihood_A, bias, likelihood_precision


def exact_potential_statistics(obs, A, obs_std, epsilon_net: EpsilonNetG, t_mid):
    """
    Returns
    -------
    likelihood_A, bias, likelihood_precision
    """
    I = torch.eye(len(obs))
    acp_t_mid = epsilon_net.alphas_cumprod[t_mid]

    # NOTE: cov_x0 doesn't depend on x hence pass in None
    cov_0_t_mid = epsilon_net.cov_x0(x=None, t=t_mid)

    # potential
    A_dot_cov_0_t_mid = A @ cov_0_t_mid
    likelihood_A = (acp_t_mid.sqrt() / (1 - acp_t_mid)) * A_dot_cov_0_t_mid
    bias = A_dot_cov_0_t_mid @ torch.linalg.solve(epsilon_net.cov, epsilon_net.mean)

    likelihood_cov = obs_std**2 * I + A_dot_cov_0_t_mid @ A.T
    likelihood_inv = torch.linalg.inv(likelihood_cov)

    return likelihood_A, bias, likelihood_inv


def ddim_backward_statistics(
    ell: int,
    t: int,
    x_ell,
    epsilon_net: EpsilonNetG,
    eta: float = 1.0,
) -> Tuple[float, float, float]:
    """s < t < ell

    p(x_t | x_ell) = q(x_t | x_ell, x_0_ell)

    Returns
    -------
    Tuple: loc, precision, matrix_x_ell
    """
    I = torch.eye(len(x_ell))
    acp_ell = epsilon_net.alphas_cumprod[ell]

    coeff_xell, coeff_xs, std = bridge_kernel_all_stats(
        ell, t, s=0, epsilon_net=epsilon_net, eta=eta
    )
    cov_0_ell = epsilon_net.cov_x0(x_ell, ell)
    x_0_ell = epsilon_net.predict_x0(x_ell, ell)

    loc = coeff_xell * x_ell + coeff_xs * x_0_ell
    matrix_x_ell = (
        coeff_xell * I + (coeff_xs * (acp_ell.sqrt() / (1 - acp_ell))) * cov_0_ell
    )

    return loc, (1 / std**2) * I, matrix_x_ell


def exact_backward_statistics(ell, t, x_ell, epsilon_net: EpsilonNetG, **kwargs):
    """
    Returns
    -------
    Tuple: loc, precision, matrix_x_ell
    """
    mean, cov = epsilon_net.mean, epsilon_net.cov

    I = torch.eye(len(x_ell))
    acp_t, acp_ell = epsilon_net.alphas_cumprod[t], epsilon_net.alphas_cumprod[ell]
    r = acp_ell / acp_t

    precision_t = torch.inverse((1 - acp_t) * I + acp_t * epsilon_net.cov)
    precision = precision_t + (r / (1 - r)) * I
    cov = torch.inverse(precision)

    loc = torch.linalg.solve(
        precision, (acp_t.sqrt() * (precision_t @ mean) + (r.sqrt() / (1 - r)) * x_ell)
    )

    matrix_x_ell = (r.sqrt() / (1 - r)) * cov

    return loc, precision, matrix_x_ell


def gaussian_posterior(
    y, likelihood_A, likelihood_bias, likelihood_precision, prior_loc, prior_precision
):
    prior_precision_matrix = prior_precision
    posterior_precision_matrix = (
        prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
    )

    posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)
    posterior_mean = posterior_covariance_matrix @ (
        likelihood_A.T @ likelihood_precision @ (y - likelihood_bias)
        + prior_precision_matrix @ prior_loc
    )

    return MultivariateNormal(
        loc=posterior_mean, covariance_matrix=posterior_covariance_matrix
    )


def _get_backward_and_potential(type_conditional: str):
    type_conditional = type_conditional.replace(" ", "")
    type_conditional = type_conditional.lower()

    # XXX we don't check that `type_conditional` has the right form
    type_backward, type_potential = type_conditional.split("+")

    return (
        globals()[f"{type_backward}_backward_statistics"],
        globals()[f"{type_potential}_potential_statistics"],
    )
