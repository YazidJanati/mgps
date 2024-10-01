import tqdm
import torch
from posterior_samplers.diffusion_utils import ddim_step, EpsilonNetMCGD
from ddrm.functions.denoising import efficient_generalized_steps


from mcg_diff.sgm import ScoreModel
from mcg_diff.particle_filter import mcg_diff
from mcg_diff.scripts.viz_gaussian import get_optimal_timesteps_from_singular_values


class EpsilonNetDDRM(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        t = t.to(int)
        return self.unet(x, t)


def ddrm(initial_noise, inverse_problem, epsilon_net, etaB, etaA, etaC):

    device = initial_noise.device
    obs, H_func, std = inverse_problem.obs, inverse_problem.H_func, inverse_problem.std
    ddrm_timesteps = epsilon_net.timesteps.clone()
    ddrm_timesteps[-1] = ddrm_timesteps[-1] - 1
    betas = 1 - epsilon_net.alphas_cumprod[1:] / epsilon_net.alphas_cumprod[:-1]
    ddrm_samples = efficient_generalized_steps(
        x=initial_noise,
        b=betas,
        seq=ddrm_timesteps.cpu(),
        model=EpsilonNetDDRM(unet=epsilon_net.net),
        y_0=obs[None, ...].to(device),
        H_funcs=H_func,
        sigma_0=std,
        etaB=1.0,
        etaA=0.85,
        etaC=1.0,
        device=device,
        classes=None,
        cls_fn=None,
    )
    return ddrm_samples[0][-1]
