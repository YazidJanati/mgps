import torch
import PIL
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from torch.func import grad
from typing import Tuple
import yaml
from dataclasses import dataclass
from torch.distributions import Distribution
from ddrm.functions.svd_replacement import H_functions
from utils.DiffJPEG.utils import quality_to_factor, diff_round
from utils.DiffJPEG.compression import compress_jpeg
from utils.DiffJPEG.decompression import decompress_jpeg
from local_paths import REPO_PATH
from PIL import Image
import math
from utils.DiffJPEG.DiffJPEG import DiffJPEG
from guided_diffusion.operators_wrappers import OPERATORS
from typing import Callable


@dataclass
class InverseProblem:
    obs: torch.Tensor = None
    H_func: Callable[[torch.Tensor], torch.Tensor] = None
    std: float = None
    log_pot: Callable[[torch.Tensor], float] = None
    task: str = None
    noise_type: str = None
    A: torch.Tensor = None


class JPEG(H_functions):

    def __init__(self, jpeg_op):
        super(JPEG).__init__()
        self.jpeg = jpeg_op

    def H(self, x):
        """
        x is in [-1, 1]
        """
        return (2 * self.jpeg((x + 1.0) / 2.0) - 1.0).reshape(x.shape[0], -1)

    def H_pinv(self, x):
        return x


class Identity(H_functions):

    def __init__(self):
        super(Identity).__init__()

    def H(self, x):
        return x.reshape(x.shape[0], -1)

    def H_pinv(self, x):
        return x


class Hsimple(H_functions):

    def __init__(self, fn):
        super(Hsimple).__init__()
        self.fn = fn

    def H(self, x):
        return self.fn(x)


def generate_invp(
    model: str,
    im_idx: str,
    task: str,
    obs_std: float,
    device: float,
):
    """Generate inverse problem.

    Supported tasks are
        - Inpainting:
            - inpainting_center
            - inpainting_middle
            - outpainting_half
            - outpainting_top
        - Blurring:
            - blur
            - blur_svd (SVD version of blur)
            - motion_blur
            - nonlinear_blur
        - JPEG dequantization
            - jpeg{QUALITY}
        - Super Resolution:
            - sr4
            - sr16
        - Others:
            - phase_retrieval
            - high_dynamic_range
    """
    ip_type = "jpeg" if task.startswith("jpeg") else "linear"
    im_dir = REPO_PATH / "assets"
    image = Image.open(im_dir / f"images/{model}/{im_idx}")

    im = torch.tensor(np.array(image)).type(torch.FloatTensor).to(device)

    x_orig = ((im - 127.5) / 127.5).squeeze(0)
    x_orig = x_orig.permute(2, 0, 1)
    D_OR = x_orig.shape
    if task.startswith("jpeg"):
        jpeg_quality = int(task.replace("jpeg", ""))
        operator = DiffJPEG(
            height=256, width=256, differentiable=True, quality=jpeg_quality
        ).to(device)
        H_func = JPEG(operator)

    elif task == "denoising":
        H_func = Identity()

    elif task in OPERATORS:
        H_func = OPERATORS[task](device=device)

    else:
        H_func = torch.load(
            im_dir / f"operators/{task}.pt",
            map_location=device,
        )

    obs = H_func.H(x_orig.unsqueeze(0))
    obs = obs + obs_std * torch.randn_like(obs)
    obs = obs.to(device)

    if task == "phase_retrieval":
        obs_img = H_func.H_pinv(obs)
        hw = int((obs_img.shape[1] / 3) ** (1 / 2))
        obs_img = obs_img.reshape(3, hw, hw)
    elif task == "blur_svd":
        hw = int((obs.shape[1] / 3) ** (1 / 2))

        obs_img = obs.reshape(3, hw, hw)
        obs_img = obs_img.clamp(-1.0, 1.0)
    else:
        # clamp(-1, 1) for pretty image plots
        # NOTE: obs_img is not used when solving inverse problem
        obs_img = H_func.H_pinv(obs).reshape(D_OR)
        obs_img = obs_img.clamp(-1.0, 1.0)

    def log_pot(x):
        diff = obs.reshape(1, -1) - H_func.H(x)
        return -0.5 * torch.norm(diff) ** 2 / obs_std**2

    return obs, obs_img.cpu(), x_orig.cpu(), H_func, ip_type, log_pot


def generate_inpainting(
    anchor_left_top: torch.Tensor,
    sizes: torch.Tensor,
    original_shape: Tuple[int, int, int],
):
    """

    :param anchor_left_top:
    :param sizes:
    :param original_shape: (x, y, n_channels)
    :return:
    """
    A_per_channel = torch.eye(original_shape[0] * original_shape[1])
    mask = torch.ones(original_shape[:2])
    mask[anchor_left_top[0] : anchor_left_top[0] + sizes[0], :][
        :, anchor_left_top[1] : anchor_left_top[1] + sizes[1]
    ] = 0
    return (
        A_per_channel[mask.flatten() == 1, :],
        A_per_channel[mask.flatten() == 0],
        mask,
    )


def display(x, save_path=None, title=None):
    sample = x.squeeze(0).cpu().permute(1, 2, 0)
    sample = (sample + 1.0) * 127.5
    sample = sample.numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_pil)
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.show()
    if save_path is not None:
        fig.savefig(save_path + ".png")


def check_image(tensor):
    assert (
        torch.max(tensor) <= 1.0 and torch.min(tensor) >= -1.0
    ), "Output images should be (-1, 1.)"


def normalize_tensor(tensor):
    check_image(tensor)
    return (tensor + 1.0) / 2.0


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
