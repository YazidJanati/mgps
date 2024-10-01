import torch
from red_diff.algos.reddiff import REDDIFF
from omegaconf import DictConfig
from red_diff.models.classifier_guidance_model import ClassifierGuidanceModel, Diffusion


def reddiff(
    initial_noise,
    inverse_problem,
    epsilon_net,
    deg,
    awd,
    cond_awd,
    grad_term_weight,
    obs_weight,
    eta,
    lr,
    denoise_term_weight,
    sigma_x0,
):
    "Wrapper over the original code of REDDIFF"
    obs, H_func, std = inverse_problem.obs, inverse_problem.H_func, inverse_problem.std

    clfg = ClassifierGuidanceModel(
        model=epsilon_net.net,
        classifier=None,
        diffusion=Diffusion(device=initial_noise.device),
        cfg=None,
    )

    dict_cfg = DictConfig(
        {
            "algo": {
                "deg": deg,
                "awd": awd,
                "cond_awd": cond_awd,
                "grad_term_weight": grad_term_weight,
                "obs_weight": obs_weight,
                "eta": eta,
                "lr": lr,
                "denoise_term_weight": denoise_term_weight,
                "sigma_x0": sigma_x0,
            }
        }
    )
    rdiff = REDDIFF(clfg, cfg=dict_cfg, H=H_func)
    return rdiff.sample(
        initial_noise, None, epsilon_net.timesteps, y_0=obs.reshape(1, -1)
    ).detach()
