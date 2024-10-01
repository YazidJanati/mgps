import torch, yaml

from nonlinear_blurring.kernel_wizard import KernelWizard
from local_paths import REPO_PATH

# load configs
with open(REPO_PATH / "src/nonlinear_blurring/option_generate_blur_default.yml") as f:
    OPT = yaml.safe_load(f)["KernelWizard"]


class NonLinearBlurModel(KernelWizard):
    """Wrapper around ``KernelWizard`` that loads ``GOPRO_wVAE`` model."""

    def __init__(self, device: int = "cpu"):
        super().__init__(OPT)

        # load weights
        self.load_state_dict(
            torch.load(OPT["pretrained"], map_location=device),
        )

        self.eval()
        self.requires_grad_(False)
