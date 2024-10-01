from tqdm import trange
from typing import Tuple, Callable

import torch
from torch import Tensor
from posterior_samplers.diffusion_utils import EpsilonNet, ddim_step
from utils.utils import display
from utils.im_invp_utils import InverseProblem


def dps(
    initial_noise: torch.Tensor,
    inverse_problem: InverseProblem,
    epsilon_net: EpsilonNet,
    gamma: float = 1.0,
    eta: float = 1.0,
    noise_type: str = "gaussian",
    poisson_rate: float = 1e-1,
    display_im: bool = False,
    display_freq: int = 20,
) -> Tensor:
    """DPS algorithm as described in [1].

    This is an implement of the Algorithm 1.

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    inverse_problem : Tuple
        observation, degradation operator, and standard deviation of noise.

    epsilon_net: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    gamma : float or Tensor
        The step size of the gradient (zeta in Algorithm 1).
        Refer to Appendix D.1 for different choices of the step size.

    noise_type : str, default 'gaussian'
        The type of the noise, either 'gaussian' or 'poisson'.

    poisson_rate : float, default 0.1
        If ``noise_type='poisson'``, the intensity of the noise.

    References
    ----------
    .. [1] Chung, Hyungjin, et al. "Diffusion posterior sampling for general noisy inverse problems."
        arXiv preprint arXiv:2209.14687 (2022).
    """
    obs, H_func = inverse_problem.obs, inverse_problem.H_func
    shape = (initial_noise.shape[0], *(1,) * len(initial_noise.shape[1:]))

    # Define potential and error functions depending on the noise type
    if noise_type == "gaussian":
        pot_func = lambda x: -torch.norm(obs.reshape(1, -1) - H_func.H(x)) ** 2.0
        error = lambda x: torch.norm(obs.reshape(1, -1) - H_func.H(x), dim=-1)
    elif noise_type == "poisson":
        rate = poisson_rate
        obs = rate * (obs.reshape(1, -1) + 1.0) / 2.0

        pot_func = lambda x: -(
            torch.norm((obs - rate * H_func((x + 1.0) / 2.0)) / (obs + 1e-3).sqrt())
            ** 2.0
        )
        error = lambda x: torch.norm(obs - rate * H_func((x + 1.0) / 2.0), dim=-1)
    else:
        raise ValueError(f"Unknown ``noise_type``. Got {noise_type}")

    sample = initial_noise.clone()
    for i in trange(len(epsilon_net.timesteps) - 1, 1, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        sample.requires_grad_()
        # NOTE: in the case of Gaussian noise, `pot_val` and `grad_norm` can be
        # computed with one call of the denoiser
        # as error is proportional to sqrt of log potential
        x_0t = epsilon_net.predict_x0(sample, t)
        error_val = error(x_0t).reshape(*shape)
        pot_val = pot_func(x_0t)
        grad_pot = torch.autograd.grad(pot_val, sample)[0]

        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=t,
            t_prev=t_prev,
            eta=eta,
            e_t=x_0t,
        )

        with torch.no_grad():
            grad_pot = (gamma / error_val) * grad_pot
            sample = sample + grad_pot

        if display_im and i % display_freq == 0:
            for j in range(sample.shape[0]):
                img = epsilon_net.predict_x0(sample[[j]], t_prev)
                display(img.clamp(-1, 1))

    return epsilon_net.predict_x0(sample, epsilon_net.timesteps[1])
