import os
from pathlib import Path
import json
from omegaconf import OmegaConf
from local_paths import REPO_PATH
import torch
import matplotlib.pyplot as plt
import PIL
import numpy as np
import random
import subprocess


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def get_gpu_memory_consumption(device: str) -> int:
    """Get the current gpu usage.

    Code adapted from:
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Parameters
    ----------
    device : str
        name of the device, for example: 'cuda:0'

    Returns
    -------
    usage: int
        memory usage in MB.

    Notes
    -----
    - Normally this function should be called during the execution of a scripts but
      it is possible to call it at the end as GPU computation is cached.
    """
    # get device id
    try:
        device_id = int(device.replace("cuda:", ""))
    except ValueError:
        raise ValueError(f"Expected device to be of the form 'cuda:ID', got {device}")

    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

    memory = gpu_memory_map.get(device_id, None)
    if memory is None:
        available_devices = [f"cuda:{i}" for i in gpu_memory_map]
        raise ValueError(
            "Unknown device name.\n"
            f"Expected device to be {available_devices}\n"
            f"got {device}"
        )

    return memory


def update_sampler_cfg(cfg):
    """
    This function exists because hydra doesn't allow dynamic interpolation with nested files.
    The dataset/task specific parameters contained in add_cfg cannot be overridden with command line
    """
    context_parameters = getattr(cfg.sampler, "context_parameters", None)
    if context_parameters is None:
        return

    for s in ("dataset", "task"):
        context = getattr(context_parameters, s, None)
        s_val = getattr(cfg, s)

        if context is None:
            continue

        if hasattr(context, s_val):
            cfg.sampler.parameters.update(context[s_val])


def save_experiment(save_folder, results, log_data):

    os.makedirs(save_folder, exist_ok=True)

    run_number = 1
    while os.path.exists(os.path.join(save_folder, f"run{run_number}")):
        run_number += 1

    run_folder = save_folder / f"run{run_number}"
    os.makedirs(run_folder)
    results_path = os.path.join(run_folder, "results.json")
    log_path = os.path.join(run_folder, "log_data.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)


def save_im(x, save_path):
    sample = x.squeeze(0).cpu().permute(1, 2, 0)
    sample = (sample + 1.0) * 127.5
    sample = sample.numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_pil)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(save_path)
    plt.close()
