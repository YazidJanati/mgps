# %%
# %load_ext autoreload
# %autoreload 2

from dataclasses import dataclass
import sys
import os
import time
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import time

import torch

from posterior_samplers.dps import dps
from posterior_samplers.ddnm import ddnm_plus
from posterior_samplers.diffpir import diffpir
from posterior_samplers.mgps import mgps_half
from posterior_samplers.resample.algo import resample
from posterior_samplers.pgdm import pgdm
from posterior_samplers.ddrm import ddrm
from posterior_samplers.reddiff import reddiff
from posterior_samplers.psld import psld

from utils.utils import display
from utils.experiments_tools import update_sampler_cfg, get_gpu_memory_consumption
from utils.metrics import LPIPS, PSNR, SSIM
from utils.im_invp_utils import generate_invp, Hsimple
from posterior_samplers.diffusion_utils import EpsilonNetSVD
from posterior_samplers.diffusion_utils import load_epsilon_net
from utils.im_invp_utils import InverseProblem
import yaml
import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt

from local_paths import REPO_PATH


device = "cuda:1"
seed = 620

torch.set_default_device(device)
torch.cuda.empty_cache()

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


@dataclass
class test:
    sampler = "mgps"
    nsteps = 50
    dataset = "ffhq"
    im_idx = "00223"
    task = "outpainting_half"
    noise_type = "gaussian"
    std = 0.05  # 0.05
    poisson_rate = 0.1
    nsamples = 1
    alpha = 0.5
    optimizer = "adam"
    lr = 3e-2
    gamma = 0.07
    threshold = 700


test_cfg = test()


@hydra.main(config_path="configs/experiments/", config_name="config")
def run_sampler(cfg: DictConfig):

    sampler = {
        "mgps": mgps_half,
        "pgdm": pgdm,
        "dps": dps,
        "reddiff": reddiff,
        "diffpir": diffpir,
        "ddnm": ddnm_plus,
        "resample": resample,
        "ddrm": ddrm,
        "psld": psld,
    }[cfg.sampler.name]

    print(f"Running {cfg.task} with {cfg.sampler.nsteps} steps...")

    epsilon_net = load_epsilon_net(cfg.dataset, cfg.sampler.nsteps, device=cfg.device)
    lpips, ssim, psnr = LPIPS(), SSIM(), PSNR()

    dataset = cfg.dataset.split("_ldm")[0]
    im = test_cfg.im_idx + ".png" if dataset == "ffhq" else test_cfg.im_idx + ".jpg"
    obs, obs_img, x_orig, H_func, ip_type, log_pot_fn = generate_invp(
        model=dataset,
        im_idx=im,
        task=cfg.task,
        noise_type=cfg.noise_type,
        obs_std=cfg.std,
        poisson_rate=cfg.poisson_rate,
        device=cfg.device,
    )

    display(obs_img.detach().cpu(), title="Observation")
    display(x_orig.cpu(), title="Ground-truth")

    shape = (3, 64, 64) if cfg.dataset.endswith("ldm") else x_orig.shape

    # NOTE: resample algorithm applies decoding internally
    if cfg.dataset.endswith("ldm") and cfg.sampler.name not in ("resample", "psld"):
        log_pot = lambda z: log_pot_fn(epsilon_net.differentiable_decode(z))
        H_fn = Hsimple(fn=lambda z: H_func.H(epsilon_net.differentiable_decode(z)))
    else:
        log_pot = log_pot_fn
        H_fn = H_func

    inverse_problem = InverseProblem(
        obs=obs, H_func=H_fn, std=cfg.std, log_pot=log_pot, task=cfg.task
    )

    initial_noise = torch.randn(test_cfg.nsamples, *shape)
    start_time = time.perf_counter()
    samples = sampler(
        initial_noise=initial_noise,
        inverse_problem=inverse_problem,
        epsilon_net=epsilon_net,
        **cfg.sampler.parameters,
    )
    end_time = time.perf_counter()

    if cfg.dataset.endswith("ldm"):
        samples = epsilon_net.decode(samples)

    samples = samples.clamp(-1.0, 1.0)

    for i in range(test_cfg.nsamples):
        display(
            samples[i],
            title=f"{test_cfg.sampler}-{test_cfg.task}-{test_cfg.dataset}-reconstruction-{i}",
        )
        plt.show()

    lpips, ssim, psnr = LPIPS(), SSIM(), PSNR()

    x_orig = x_orig.to(device)

    print(f"{test_cfg.sampler} metrics")
    print(f"lpips: {lpips.score(samples, x_orig)}")
    print(f"ssim: {ssim.score(samples, x_orig)}")
    print(f"psnr: {psnr.score(samples, x_orig)}")
    print("===================")
    print(f"runtime: {end_time - start_time}")
    print(f"GPU: {get_gpu_memory_consumption(device)}")


# # XXX for interactive window
# sys.argv = [sys.argv[0], f"sampler={test_cfg.sampler}"]

if __name__ == "__main__":
    run_sampler()
