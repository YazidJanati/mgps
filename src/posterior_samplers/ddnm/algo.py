import torch
from tqdm import tqdm

from utils.utils import display
from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.ddnm.utils import get_special_methods
from utils.im_invp_utils import InverseProblem

class_num = 951


@torch.no_grad()
def ddnm(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    inverse_problem: InverseProblem,
    eta: float = 0.85,
    config=None,
) -> torch.Tensor:
    """DDNM algorithm without noise as described in [1].

    Parameters
    ----------
    x : Tensor
        initial noise

    model: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    inverse_problem : Tuple
        observation, degradation operator, and standard deviation of noise.

    eta : float
        Hyperparameter, default to 0.85 as in
        https://github.com/wyhuai/DDNM/blob/main/README.md#quick-start

    config : Dict
        configuration.

    References
    ----------
    .. [1] Wang, Yinhuai, Jiwen Yu, and Jian Zhang.
        "Zero-shot image restoration using denoising diffusion null-space model."
        arXiv preprint arXiv:2212.00490 (2022).
    """
    y, A_funcs = inverse_problem.obs, inverse_problem.H_func

    model = epsilon_net
    x = initial_noise
    device = x.device

    # deduce b from alphas_cumprod
    # b ---> betas in DDPM
    # c.f. https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/guided_diffusion/diffusion.py#L588
    b = 1 - model.alphas_cumprod[1:] / model.alphas_cumprod[:-1]

    # use default configs of DDNM as in
    # https://github.com/wyhuai/DDNM/blob/main/configs/celeba_hq.yml
    if config is None:
        config = {
            "diffusion": {
                "num_diffusion_timesteps": len(model.alphas_cumprod),
            },
            "time_travel": {
                "T_sampling": len(model.timesteps),
                "travel_length": 1,
                "travel_repeat": 1,
            },
        }

    # setup iteration variables
    skip = (
        config["diffusion"]["num_diffusion_timesteps"]
        // config["time_travel"]["T_sampling"]
    )
    n = x.size(0)
    x0_preds = []
    xs = [x]

    # generate time schedule
    times = get_schedule_jump(
        config["time_travel"]["T_sampling"],
        config["time_travel"]["travel_length"],
        config["time_travel"]["travel_repeat"],
    )
    time_pairs = list(zip(times[:-1], times[1:]))

    # reverse diffusion sampling
    for it, (i, j) in enumerate(tqdm(time_pairs), start=1):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        if j < i:  # normal sampling
            t = (torch.ones(n) * i).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at = compute_alpha(b, t.long(), x)
            at_next = compute_alpha(b, next_t.long(), x)
            xt = xs[-1].to(device)

            et = model(xt, t[0].to(dtype=torch.int32))

            # NOTE we don't currently account for conditional sampling
            # to add support of it see
            # https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/functions/svd_ddnm.py#L46-L52

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            x0_t_hat = x0_t - A_funcs.H_pinv(
                A_funcs.H(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
            ).reshape(*x0_t.size())

            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta**2) ** 0.5)
            xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

            x0_preds.append(x0_t.to("cpu"))
            xs.append(xt_next.to("cpu"))

            # # XXX uncomment to view evolution of images
            # if it % 50 == 0:
            #     img = model.predict_x0(xt_next[[0]], next_t.long()[0])
            #     display(img)

        else:  # time-travel back
            next_t = (torch.ones(n) * j).to(x.device)
            at_next = compute_alpha(b, next_t.long(), x)
            x0_t = x0_preds[-1].to(device)

            xt_next = (
                at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
            )

            xs.append(xt_next.to("cpu"))

    return xt_next


@torch.no_grad()
def ddnm_plus(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    inverse_problem: InverseProblem,
    eta: float = 0.85,
    config=None,
) -> torch.Tensor:
    """DDNM algorithm with noise as described in [1].

    Parameters
    ----------
    initial_noise : Tensor
        initial noise

    model: Instance of EpsilonNet
        Noise predictor coming from a diffusion model.

    inverse_problem : Tuple
        observation, degradation operator, and standard deviation of noise.

    eta : float
        Hyperparameter, default to 0.85 as in
        https://github.com/wyhuai/DDNM/blob/main/README.md#quick-start

    config : Dict
        configuration.

    References
    ----------
    .. [1] Wang, Yinhuai, Jiwen Yu, and Jian Zhang.
        "Zero-shot image restoration using denoising diffusion null-space model."
        arXiv preprint arXiv:2212.00490 (2022).
    """
    y, A_funcs, sigma_y = (
        inverse_problem.obs,
        inverse_problem.H_func,
        inverse_problem.std,
    )
    model = epsilon_net
    x = initial_noise
    device = x.device

    # make SVD operator compatible with DDNM
    Lambda_func, Lambda_noise_func = get_special_methods(A_funcs)

    # deduce b from alphas_cumprod
    # b ---> betas in DDPM
    # c.f. https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/guided_diffusion/diffusion.py#L588
    b = 1 - model.alphas_cumprod[1:] / model.alphas_cumprod[:-1]

    # use default configs of DDNM as in
    # https://github.com/wyhuai/DDNM/blob/main/configs/celeba_hq.yml
    if config is None:
        config = {
            "diffusion": {
                "num_diffusion_timesteps": len(model.alphas_cumprod),
            },
            "time_travel": {
                "T_sampling": len(model.timesteps),
                "travel_length": 1,
                "travel_repeat": 1,
            },
        }

    # setup iteration variables
    skip = (
        config["diffusion"]["num_diffusion_timesteps"]
        // config["time_travel"]["T_sampling"]
    )
    n = x.size(0)
    x0_preds = []
    xs = [x]

    # generate time schedule
    times = get_schedule_jump(
        config["time_travel"]["T_sampling"],
        config["time_travel"]["travel_length"],
        config["time_travel"]["travel_repeat"],
    )
    time_pairs = list(zip(times[:-1], times[1:]))

    # reverse diffusion sampling
    for it, (i, j) in tqdm(enumerate(time_pairs)):
        i, j = i * skip, j * skip
        if j < 0:
            j = -1

        if j < i:  # normal sampling
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long(), x)
            at_next = compute_alpha(b, next_t.long(), x)
            xt = xs[-1].to(device)

            et = model(xt, t[0].to(dtype=torch.int32))

            # NOTE we don't currently account for conditional sampling
            # to add support of it see
            # https://github.com/wyhuai/DDNM/blob/00b58eac7843a4c99114fd8fa42da7aa2b6808af/functions/svd_ddnm.py#L46-L52

            # Eq. 12
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # NOTE to make it compatible with data other then RGB images
            sigma_t = (1 - at_next).sqrt()[0, *[0 for _ in x.shape[1:]]]

            # Eq. 17
            x0_t_hat = x0_t - Lambda_func(
                A_funcs.H_pinv(
                    A_funcs.H(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1),
                at_next.sqrt()[0, *[0 for _ in x.shape[1:]]],
                sigma_y,
                sigma_t,
                eta,
            ).reshape(*x0_t.size())

            # Eq. 51
            xt_next = at_next.sqrt() * x0_t_hat + Lambda_noise_func(
                torch.randn_like(x0_t).reshape(x0_t.size(0), -1),
                at_next.sqrt()[0, *[0 for _ in x.shape[1:]]],
                sigma_y,
                sigma_t,
                eta,
                et.reshape(et.size(0), -1),
            ).reshape(*x0_t.size())

            x0_preds.append(x0_t.to("cpu"))
            xs.append(xt_next.to("cpu"))

            # # XXX uncomment to view evolution of images
            # if it % 50 == 0:
            #     img = model.predict_x0(xt_next[[0]], next_t.long()[0])
            #     display(img)
        else:  # time-travel back
            next_t = (torch.ones(n) * j).to(x.device)
            at_next = compute_alpha(b, next_t.long(), x)
            x0_t = x0_preds[-1].to(device)

            xt_next = (
                at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
            )

            xs.append(xt_next.to("cpu"))

    return xt_next


# form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts


def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


def compute_alpha(beta, t, x):
    # x is passed in to make the right view of alpha
    shape = [1 for _ in x.shape[1:]]
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, *shape)
    return a


def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)
