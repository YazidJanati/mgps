from tqdm import trange
import torch
from torch import Tensor

from posterior_samplers.diffusion_utils import EpsilonNet
from ddrm.functions.svd_replacement import H_functions

from utils.utils import display
from utils.im_invp_utils import InverseProblem


def diffpir(
    initial_noise: Tensor,
    epsilon_net: EpsilonNet,
    inverse_problem: InverseProblem,
    lmbd: float,
    zeta: float,
    n_reps: int = 1,
) -> Tensor:
    """DiffPIR algorithm as described in [1].

    Refer to Table 3 for setting the hyperparameters ``lmbd`` and ``zeta``.

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    epsilon_net: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    H_func :
        Inverse problem operator.

    y : Tensor
        The observation.

    std : float
        The standard deviation.

    lmbd : float
        Hyperparameter.

    zeta : float
        Hyperparameter.

    n_reps : int, default = 1
        The number of times to repeat a diffusion step.

    References
    ----------
    .. [1] Zhu, Yuanzhi, et al. "Denoising diffusion models for plug-and-play image restoration."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    """
    obs, H_func, std = inverse_problem.obs, inverse_problem.H_func, inverse_problem.std
    epsilon_net.requires_grad_(False)
    acp = epsilon_net.alphas_cumprod
    timesteps = epsilon_net.timesteps

    n_samples = initial_noise.shape[0]
    zeta = torch.tensor(zeta, dtype=initial_noise.dtype)

    x_t = torch.randn_like(initial_noise)
    for it, i in enumerate(trange(len(timesteps) - 1, 1, -1), start=1):
        t, t_prev = timesteps[i], timesteps[i - 1]
        acp_t, acp_t_prev = acp[t], acp[t_prev]

        for _ in range(n_reps):
            # --- step 1
            x_0t= epsilon_net.predict_x0(x_t, t)

            # --- step 2
            # solve data fitting problem
            rho_t = std**2 * lmbd / ((1 - acp_t) / acp_t)

            if _is_svd_decomposed(H_func):
                x_0t = _argmin_quadratic_problem(
                    A=H_func,
                    gamma=rho_t,
                    b=rho_t * x_0t.view(n_samples, -1) + H_func.Ht(obs),
                )
            else:
                # NOTE: Using this approximation results in diverging algorithm
                x_0t = _approximate_argmin_quadratic_problem(x_0t, H_func, obs, rho_t)

            x_0t = x_0t.view(*initial_noise.shape)

            # --- step 3
            e_t_y = (x_t - acp_t.sqrt() * x_0t) / (1 - acp_t).sqrt()
            noise = (1 - zeta).sqrt() * e_t_y + zeta.sqrt() * torch.randn_like(x_t)

            x_t = acp_t_prev.sqrt() * x_0t + (1 - acp_t_prev).sqrt() * noise

    return x_t


def _argmin_quadratic_problem(A: H_functions, gamma: float, b: Tensor) -> Tensor:
    """Solve for x the problem ``(gamma * I + A.T @ A) x = b``"""
    singulars = A.singulars()

    out = A.Vt(b)
    out[:, : len(singulars)] /= gamma + singulars**2
    out[:, len(singulars) :] /= gamma
    out = A.V(out)

    return out


def _approximate_argmin_quadratic_problem(
    x_0t: Tensor, H_func: H_functions, obs: Tensor, rho_t
):
    x_0t.requires_grad_()

    loss = ((obs - H_func.H(x_0t)) ** 2).sum()
    loss.backward()

    with torch.no_grad():
        return x_0t - x_0t.grad / (2 * rho_t)


def _is_svd_decomposed(H_func: H_functions):
    required_methods = ("V", "singulars")

    for method in required_methods:
        method_fn = getattr(H_func, method)

        # NOTE: since, we want to check wether operator is defined through SVD decomposition
        # ignore other errors, they better get catch in the algo
        try:
            method_fn(None)
        except NotImplementedError:
            return False
        except:
            pass

    return True
