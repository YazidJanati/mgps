import torch
from torch import Tensor
from ddrm.functions.svd_replacement import H_functions, Colorization

from nonlinear_blurring import NonLinearBlurModel
from guided_diffusion.measurements import (
    GaussialBlurOperator,
    MotionBlurOperator,
    PhaseRetrievalOperator,
)


OPERATORS = {}


def register_operator(name: str):
    def wrapper(cls):
        if OPERATORS.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        OPERATORS[name] = cls
        return cls

    return wrapper


@register_operator("colorization")
class Colorization256(Colorization):
    def __init__(self, device):
        super().__init__(img_dim=256, device=device)


@register_operator("blur")
class Blur(H_functions):
    """Wrapper around ``GaussialBlurOperator`` that follows ``H_functions`` API.

    Notes
    -----
    Default parameters were borrowed from
    https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/configs/gaussian_deblur_config.yaml
    """

    def __init__(self, kernel_size=61, intensity=3.0, device="cpu"):
        super().__init__()

        self.degradation = GaussialBlurOperator(kernel_size, intensity, device)
        self.degradation.conv.requires_grad_(False)

    def H(self, x: Tensor):
        n_samples = x.shape[0]
        return self.degradation.forward(x).view(n_samples, -1)

    def Ht(self, x: Tensor):
        n_samples = x.shape[0]
        return self.degradation.transpose(x).view(n_samples, -1)

    # HACK: this is not the pseudo inverse of the operator
    # As long as we are using this method to display the degraded image (observation)
    def H_pinv(self, x: Tensor):
        return x


@register_operator("motion_blur")
class MotionBlur(H_functions):
    """Wrapper around ``MotionBlurOperator`` that follows ``H_functions`` API.

    Notes
    -----
    Default parameters were borrowed from
    https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/configs/motion_deblur_config.yaml
    """

    def __init__(self, kernel_size=61, intensity=0.5, device="cpu"):
        super().__init__()

        self.degradation = MotionBlurOperator(kernel_size, intensity, device)
        self.degradation.conv.requires_grad_(False)

    def H(self, x: Tensor):
        n_samples = x.shape[0]
        return self.degradation.forward(x).view(n_samples, -1)

    def Ht(self, x: Tensor):
        n_samples = x.shape[0]
        return self.degradation.transpose(x).view(n_samples, -1)

    # HACK: this is not the pseudo inverse of the operator
    # As long as we are using this method to display the degraded image (observation)
    def H_pinv(self, x: Tensor):
        return x


@register_operator("phase_retrieval")
class PhaseRetrieval(H_functions):
    """Wrapper around ``PhaseRetrievalOperator`` that follows ``H_functions`` API.

    Notes
    -----
    Default parameters were borrowed from
    https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/configs/phase_retrieval_config.yaml
    """

    def __init__(self, oversample=2.0, device="cpu"):
        super().__init__()

        self.degradation = PhaseRetrievalOperator(oversample, device)

    def H(self, x: Tensor):
        n_samples = x.shape[0]
        return self.degradation.forward(x).view(n_samples, -1)

    # HACK: this is not the pseudo inverse of the operator
    # As long as we are using this method to display the degraded image (observation)
    def H_pinv(self, x: Tensor):
        return x


@register_operator("high_dynamic_range")
class HDR(H_functions):
    """High Dynamic Range operator that follows ``H_functions`` API."""

    def __init__(self, scale: float = 2.0, device: str = "cpu") -> None:
        super().__init__()
        self.scale = scale
        self.device = device

    def H(self, x: Tensor):
        n_samples = x.shape[0]
        scaled_x = x * self.scale
        out = torch.clip(scaled_x, -1, 1)

        # NOTE: sometimes `x` might be non-contiguous and hence we can't change its view
        # without data-copying it, which is less computationally efficient.
        # Avoiding forcing `reshape`.
        if scaled_x.is_contiguous():
            return out.view(n_samples, -1)
        else:
            return out.reshape(n_samples, -1)

    # HACK: this is not the pseudo inverse of the operator
    # As long as we are using this method to display the degraded image (observation)
    def H_pinv(self, x: Tensor):
        return x


@register_operator(name="nonlinear_blur")
class NonlinearBlurOperator(H_functions):
    """Nonlinear Deblurring operator that follows ``H_functions`` API.

    Notes
    -----
    Implementation adapted from
    https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/measurements.py#L174-L200
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.blur_model = NonLinearBlurModel(device)

    def H(self, x: Tensor) -> Tensor:
        n_samples = x.shape[0]
        random_kernel = 1.2 * torch.randn(size=(1, 512, 2, 2), device=self.device)

        x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(x, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]

        return blurred.view(n_samples, -1)

    # HACK: this is not the pseudo inverse of the operator
    # As long as we are using this method to display the degraded image (observation)
    def H_pinv(self, x: Tensor):
        return x
