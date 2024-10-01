import torch
from functools import partial

from utils.im_invp_utils import InverseProblem
from ldm.models.diffusion.ddpm import LatentDiffusion
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.resample.utils import DDIMSampler, get_conditioning_method


def resample(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    inverse_problem: InverseProblem,
    noise_type: str = "gaussian",
    max_optimization_iters: int = 2000,
    sigma_scale: float = 40.0,
    scale: float = 0.3,
    eta: float = 1.0,
) -> torch.Tensor:
    """Wrapper around Resample algorithm [1].

    Source of the implementation https://github.com/soominkwon/resample

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    epsilon_net: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    inverse_problem : instance of InverseProblem
        Object that defines the inverse problem.

    noise_type : str, default='gaussian'
        Either "gaussian" or "poisson".

    max_optimization_iters: int, default=2000
        The maximum number of optimization iterations to perform to ensure
        data consistency.

    sigma_scale : float, default=40
        A scaler that multiply the variance of the ``stochastic_resample``.

    scale : float, default=0.3
        Posterior Sampling steps as defined in DPS [2].

    eta : float, default=1
        The coefficient that multiplies DDIM variance.


    References
    ----------
    .. [1] Song, Bowen, et al. "Solving inverse problems with latent diffusion
        models via hard data consistency." arXiv preprint arXiv:2307.08123 (2023).
    .. [2] Chung, Hyungjin, et al. "Diffusion posterior sampling for general noisy inverse problems."
        arXiv preprint arXiv:2209.14687 (2022).
    """
    # check model abides by LDM
    latent_model = getattr(epsilon_net.net, "net", None)
    if not isinstance(latent_model, LatentDiffusion):
        raise ValueError("ReSample algorithm is only compatible with `LatentDiffusion`")

    # BUG currently ReSample doesn't throws an index out of bound when
    # the number diffusion steps is different than 500
    # c.f. https://github.com/soominkwon/resample/issues/5
    if len(epsilon_net.timesteps) != 500:
        raise ValueError(
            "ReSample algorithm supports only 500 diffusion steps.\n"
            "Change Diffusion steps to 500."
        )

    # inverse problem
    H_funcs, y, std = inverse_problem.H_func, inverse_problem.obs, inverse_problem.std

    # The tolerance for solving the optimization problem for data consistency problems
    eps = std

    # init
    n_samples = initial_noise.shape[0]
    sampler = DDIMSampler(epsilon_net)
    measurement_cond_fn = get_conditioning_method(
        "ps", epsilon_net.net.net, H_funcs.H, noiser=noise_type
    ).conditioning
    # NOTE: scale is not account for in the script provided in official implementation
    # it is also hard coded in the algorithm
    measurement_cond_fn = partial(measurement_cond_fn, scale=scale)

    # Instantiate sampler
    sample_fn = partial(
        sampler.posterior_sampler,
        measurement_cond_fn=measurement_cond_fn,
        operator_fn=H_funcs.H,
        S=len(epsilon_net.timesteps),
        cond_method="resample",
        conditioning=None,
        ddim_use_original_steps=True,
        batch_size=n_samples,
        shape=[3, 64, 64],
        verbose=False,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        eta=eta,
        eps=eps,
        max_optimization_iters=max_optimization_iters,
        sigma_scale=sigma_scale,
    )

    # solve problem
    samples_ddim, _ = sample_fn(measurement=y)

    return samples_ddim
