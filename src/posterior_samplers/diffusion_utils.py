import torch
from torch.distributions import Distribution

import tqdm
from typing import Tuple

from utils.utils import load_yaml, fwd_mixture
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from diffusers import DDPMPipeline
from torch.func import grad
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from local_paths import REPO_PATH


class UNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        return self.unet(x, torch.tensor([t]))[:, :3]


class LDM(torch.nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t):
        return self.net.model(x, torch.tensor([t]))

    def decode(self, z):
        if hasattr(self.net, "decode_first_stage"):
            return self.net.decode_first_stage(z)
        else:
            raise NotImplementedError

    def differentiable_decode(self, z):
        if hasattr(self.net, "differentiable_decode_first_stage"):
            return self.net.differentiable_decode_first_stage(z)
        else:
            raise NotImplementedError

    def differentiable_encode(self, x):
        z = self.net.differentiable_encode_first_stage(x)

        # NOTE: this operation involves scaling the encoding with factor
        # in our case this factor is 1., but perhaps it is not for other models
        # c.f. `get_first_stage_encoding` in ldm/models/diffusion/ddpm.py
        z = self.net.get_first_stage_encoding(z)

        return z

    def encode(self, x):
        z = self.net.encode_first_stage(x)
        # NOTE: see differentiable_encoder for info
        z = self.net.get_first_stage_encoding(z)
        return z


class EpsilonNetGM(torch.nn.Module):

    def __init__(self, means, weights, alphas_cumprod, cov=None):
        super().__init__()
        self.means = means
        self.weights = weights
        self.covs = cov
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x, t):
        # if len(t) == 1 or t.dim() == 0:
        #     acp_t = self.alphas_cumprod[t.to(int)]
        # else:
        #     acp_t = self.alphas_cumprod[t.to(int)][0]
        acp_t = self.alphas_cumprod[t.to(int)]
        grad_logprob = grad(
            lambda x: fwd_mixture(
                self.means, self.weights, self.alphas_cumprod, t, self.covs
            )
            .log_prob(x)
            .sum()
        )
        return -((1 - acp_t) ** 0.5) * grad_logprob(x)


class EpsilonNetMCGD(torch.nn.Module):

    def __init__(self, H_funcs, unet, dim):
        super().__init__()
        self.unet = unet
        self.H_funcs = H_funcs
        self.dim = dim

    def forward(self, x, t):
        x_normal_basis = self.H_funcs.V(x).reshape(-1, *self.dim)
        # .repeat(x.shape[0]).to(x.device)
        t_emb = torch.tensor(t).to(x.device)
        eps = self.unet(x_normal_basis, t_emb)
        eps_svd_basis = self.H_funcs.Vt(eps)
        return eps_svd_basis


class EpsilonNet(torch.nn.Module):
    def __init__(self, net, alphas_cumprod, timesteps):
        super().__init__()
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps

    def forward(self, x, t):
        return self.net(x, torch.tensor(t))

    def predict_x0(self, x, t):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        return (x - (1 - acp_t) ** 0.5 * self.forward(x, t)) / (acp_t**0.5)

    def score(self, x, t):
        acp_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - acp_t) ** 0.5

    def decode(self, z):
        return self.net.decode(z)

    def differentiable_decode(self, z):
        return self.net.differentiable_decode(z)

    # def value_and_grad_predx0(self, x, t):
    #     x = x.requires_grad_()
    #     pred_x0 = self.predict_x0(x, t)
    #     grad_pred_x0 = torch.autograd.grad(pred_x0.sum(), x)[0]
    #     return pred_x0, grad_pred_x0

    # def value_and_jac_predx0(self, x, t):
    #     def pred(x):
    #         return self.predict_x0(x, t)

    #     pred_x0 = self.predict_x0(x, t)
    #     return pred_x0, vmap(jacrev(pred))(x)


# TODO: fix shape handling
class EpsilonNetSVD(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, shape, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device
        self.shape = shape

    def forward(self, x, t):
        # shape = (x.shape[0], 3, int(np.sqrt((x.shape[-1] // 3))), -1)
        x = self.H_func.V(x.to(self.device)).reshape(self.shape)
        return self.H_func.Vt(self.net(x, t))


class EpsilonNetSVDGM(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, t):
        x = self.H_func.V(x.to(self.device))
        return self.H_func.Vt(self.net(x, t))


def load_ldm_model(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_gmm_epsilon_net(prior: Distribution, dim: int, n_steps: int):
    timesteps = torch.linspace(0, 999, n_steps).long()
    alphas_cumprod = torch.linspace(0.9999, 0.98, 1000)
    alphas_cumprod = torch.cumprod(alphas_cumprod, 0).clip(1e-10, 1)
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

    means, covs, weights = (
        prior.component_distribution.mean,
        prior.component_distribution.covariance_matrix,
        prior.mixture_distribution.probs,
    )

    epsilon_net = EpsilonNet(
        net=EpsilonNetGM(means, weights, alphas_cumprod, covs),
        alphas_cumprod=alphas_cumprod,
        timesteps=timesteps,
    )

    return epsilon_net


def load_epsilon_net(model_id: str, n_steps: int, device: str):
    hf_models = {
        "celebahq": "google/ddpm-celebahq-256",
    }
    pixelsp_models = {
        "ffhq": REPO_PATH / "configs/ffhq_model.yaml",
        "imagenet": REPO_PATH / "configs/imagenet_model.yaml",
    }
    ldm_models = {
        "ffhq_ldm": REPO_PATH / "configs/latent-diffusion/ffhq-ldm-vq-4.yaml",
    }

    timesteps = torch.linspace(0, 999, n_steps).long()

    if model_id in hf_models:
        hf_id = "google/ddpm-celebahq-256"
        pipeline = DDPMPipeline.from_pretrained(hf_id).to(device)
        model = pipeline.unet
        model = model.requires_grad_(False)
        model = model.eval()

        timesteps = torch.linspace(0, 999, n_steps).long()
        alphas_cumprod = pipeline.scheduler.alphas_cumprod.clip(1e-6, 1)
        alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

        return EpsilonNet(
            net=UNet(model), alphas_cumprod=alphas_cumprod, timesteps=timesteps
        )

    if model_id in pixelsp_models:

        # NOTE code verified at https://github.com/openai/guided-diffusion
        # and adapted from https://github.com/DPS2022/diffusion-posterior-sampling

        model_config = pixelsp_models[model_id]
        diffusion_config = REPO_PATH / "configs/diffusion_config.yaml"

        model_config = load_yaml(model_config)
        diffusion_config = load_yaml(diffusion_config)

        sampler = create_sampler(**diffusion_config)
        model = create_model(**model_config)

        # by default set model to eval mode and disable grad on model parameters
        model = model.eval()
        model.requires_grad_(False)

        alphas_cumprod = torch.tensor(sampler.alphas_cumprod).float().clip(1e-6, 1)
        alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

        net = UNet(model)
        return EpsilonNet(
            net=net,
            alphas_cumprod=alphas_cumprod,
            timesteps=timesteps,
        )

    if model_id in ldm_models:
        cfg_path = OmegaConf.load(ldm_models[model_id])
        ckpt_path = cfg_path.model.params.unet_config.ckpt_path
        model = load_ldm_model(cfg_path, ckpt_path, device)

        model = model.eval()
        model.requires_grad_(False)
        alphas_cumprod = torch.tensor(model.alphas_cumprod).float().clip(1e-6, 1)
        alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

        return EpsilonNet(
            net=LDM(model), alphas_cumprod=alphas_cumprod, timesteps=timesteps
        )


def bridge_kernel_statistics(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    """s < t < ell"""
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
    return coeff_xell * x_ell + coeff_xs * x_s, std


def bridge_kernel_all_stats(
    ell: int,
    t: int,
    s: int,
    epsilon_net: EpsilonNet,
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


def sample_bridge_kernel(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float = 1.0,
):
    mean, std = bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    return mean + std * torch.randn_like(mean)


def ddim_statistics(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return bridge_kernel_statistics(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim_step(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return sample_bridge_kernel(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim(
    initial_noise_sample: torch.Tensor, epsilon_net: EpsilonNet, eta: float = 1.0
) -> torch.Tensor:
    """
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)
    :param initial_noise_sample: Initial "noise"
    :param timesteps: List containing the timesteps. Should start by 999 and end by 0
    :param score_model: The score model
    :param eta: the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    :return:
    """
    sample = initial_noise_sample
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=t,
            t_prev=t_prev,
            eta=eta,
        )
    sample = epsilon_net.predict_x0(sample, epsilon_net.timesteps[1])

    return epsilon_net.decode(sample) if hasattr(epsilon_net.net, "decode") else sample
