from tqdm import trange
from typing import Tuple, Callable

import torch
from torch import Tensor
from posterior_samplers.diffusion_utils import EpsilonNet, ddim_step
from utils.utils import display
from utils.im_invp_utils import InverseProblem


def psld(
    initial_noise: torch.Tensor,
    inverse_problem: InverseProblem,
    epsilon_net: EpsilonNet,
    gamma: float = 1.0,
    omega: float = 0.1,
    eta: float = 1.0,
    display_im: bool = False,
    display_freq: int = 20,
) -> Tensor:
    """PSLD algorithm as described in [1].

    This is an implement of the Algorithm 2 in [1].
    Official implementation is available in https://github.com/LituRout/PSLD.
    In particular, https://github.com/LituRout/PSLD/blob/d734647bbc1ed0b1171521a804fc744973779f8c/stable-diffusion/ldm/models/diffusion/psld.py#L188-L338

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    inverse_problem : Tuple
        observation, degradation operator, and standard deviation of noise.

    epsilon_net: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    gamma : float
        denoted by eta in the algo, stepsize associated with the constraint

    omega : float
        gamma in the algorithm, stepsize associated with likelihood.

    eta : float, default=1
        DDIM hyperparameter. If ``eta=1``, the sampling algorithm is DDPM.

    References
    ----------
    .. [1] Rout, Litu, et al. "Solving linear inverse problems provably via
        posterior sampling with latent diffusion models."
        Advances in Neural Information Processing Systems 36 (2024).
    """
    x_shape = (3, 256, 256)
    n_samples = initial_noise.shape[0]
    obs, H_func = inverse_problem.obs, inverse_problem.H_func

    Ht_obs = H_func.Ht(obs)

    def encoder_fn(x):
        encoded = epsilon_net.net.differentiable_encode(x.view(n_samples, *x_shape))
        return encoded

    def decoder_fn(z):
        return epsilon_net.net.differentiable_decode(z)

    z_t = initial_noise.clone()
    for i in trange(len(epsilon_net.timesteps) - 1, 1, -1):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]

        z_t = z_t.requires_grad_()
        z_0t = epsilon_net.predict_x0(z_t, t)
        decoded_z_0t = decoder_fn(z_0t).view(n_samples, -1)
        H_dot_decoded_z_0t = H_func.H(decoded_z_0t.view(n_samples, *x_shape))

        # compute errors
        # NOTE: the official implementation of the algorithm uses norm (without square)
        # yet the algorithm features the norm square
        ll_error = torch.norm(obs - H_dot_decoded_z_0t)
        gluing_error = torch.norm(
            z_0t - encoder_fn(Ht_obs + decoded_z_0t - H_func.Ht(H_dot_decoded_z_0t))
        )

        error = omega * ll_error + gamma * gluing_error
        error.backward()
        grad = z_t.grad

        with torch.no_grad():
            z_t = ddim_step(
                x=z_t, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=z_0t
            )
            z_t = z_t - grad

            if display_im and i % display_freq == 0:
                for j in range(z_t.shape[0]):
                    img = epsilon_net.predict_x0(z_t[[j]], t_prev)
                    display(img.clamp(-1, 1))

    # last denoising step
    return epsilon_net.predict_x0(z_t, epsilon_net.timesteps[1])
