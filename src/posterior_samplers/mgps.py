import torch
from tqdm import tqdm
from functools import partial
from typing import Callable

from posterior_samplers.diffusion_utils import EpsilonNet
from posterior_samplers.diffusion_utils import sample_bridge_kernel
from posterior_samplers.diffusion_utils import bridge_kernel_statistics

from utils.utils import display
from utils.im_invp_utils import InverseProblem


def _elbo(
    vmean: torch.Tensor,
    vlogstd: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    t: int,
    t_mid: int,
):
    acp_t, acp_tmid = (
        epsilon_net.alphas_cumprod[t],
        epsilon_net.alphas_cumprod[t_mid],
    )
    ratio_acp = acp_t / acp_tmid

    x_tmid = vmean + vlogstd.exp() * torch.randn_like(vmean)
    e_tmid = epsilon_net.predict_x0(x_tmid, t_mid)

    with torch.no_grad():
        score_tmid = (-x_tmid + acp_tmid.sqrt() * e_tmid) / (1 - acp_tmid)

    log_pot_val = log_pot(x_tmid) if t_mid == 1 else log_pot(e_tmid)
    log_fwd = (
        -0.5
        * (
            (x_t - ratio_acp.sqrt() * vmean) ** 2 + ratio_acp * (2 * vlogstd).exp()
        ).sum()
        / (1 - ratio_acp)
    )
    kl_div = -log_pot_val - vlogstd.sum() - log_fwd - (x_tmid * score_tmid).sum()

    return kl_div


def _rb_elbo(
    vmean: torch.Tensor,
    vlogstd: torch.Tensor,
    mean_prior: torch.Tensor,
    logstd_prior: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    t: int,
):
    """
    'Rao-Blackwellized' elbo using the Gaussian approximation of the backward transition
    """
    kl_prior = kl_mvn(vmean, vlogstd, mean_prior, logstd_prior)
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)

    if t > 1:
        pred_x0 = epsilon_net.predict_x0(vsample, t)
        int_log_pot_est = -log_pot(pred_x0)
    else:
        int_log_pot_est = -log_pot(vsample)

    return int_log_pot_est + kl_prior


def mgps_vi_tmid_step(
    i: int,
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    optimizer: "str",
    lr: float,
    gradient_steps_fn: Callable[[int], float],
    t: int,
    t_prev: int,
    t_mid: int,
    warm_start: bool,
):
    n_gradient_steps = gradient_steps_fn(i)

    t0 = 1 if warm_start else 0
    init_mean_tmid, std_prior = bridge_kernel_statistics(
        x_t, pred_x0, epsilon_net, t, t_mid, t0
    )
    logstd_tmid = torch.tensor(std_prior).log() * torch.ones_like(x_t)

    vmean, vlogstd = (
        init_mean_tmid.requires_grad_(),
        logstd_tmid.clone().requires_grad_(),
    )

    pred_x0t = epsilon_net.predict_x0(x_t, t)
    mean_tmid, std_tmid = bridge_kernel_statistics(
        x_t, pred_x0t, epsilon_net, t, t_mid, 0
    )
    logstd_tmid = std_tmid.log()

    kl_fn = partial(
        _rb_elbo,
        mean_prior=mean_tmid,
        logstd_prior=logstd_tmid,
        epsilon_net=epsilon_net,
        log_pot=log_pot,
        t=t_mid,
    )

    optim = torch.optim.Adam(params=[vmean, vlogstd], lr=lr)

    if optimizer == "adam":
        for _ in range(n_gradient_steps):
            optim.zero_grad()
            kl_div = kl_fn(vmean, vlogstd)
            kl_div.backward()
            optim.step()
    elif optimizer == "sgd":
        for _ in range(n_gradient_steps):
            vmean.requires_grad_(), vlogstd.requires_grad_()
            kl_div = kl_fn(vmean, vlogstd)
            mean_grad, logstd_grad = torch.autograd.grad(kl_div, (vmean, vlogstd))

            vmean = normalized_grad_step(vmean, mean_grad, lr=lr)
            vlogstd = normalized_grad_step(vlogstd, logstd_grad, lr=lr)

    vmean.detach_(), vlogstd.detach_()
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)

    return vsample


def mgps_vi_t0_step(
    i: int,
    x_tmid: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    optimizer: str,
    lr: float,
    gradient_steps_fn: Callable[[int], float],
    t_mid: int,
):
    n_gradient_steps = gradient_steps_fn(i)

    mean_t1, std_t1 = bridge_kernel_statistics(
        x_ell=x_tmid,
        x_s=epsilon_net.predict_x0(x_tmid, t_mid),
        epsilon_net=epsilon_net,
        ell=t_mid,
        t=1,
        s=0,
    )
    logstd_t1 = torch.tensor(std_t1).log() * torch.ones_like(x_tmid)

    vmean, vlogstd = (
        mean_t1.clone().requires_grad_(),
        logstd_t1.clone().requires_grad_(),
    )

    kl_fn = partial(
        _elbo, epsilon_net=epsilon_net, log_pot=log_pot, x_t=x_tmid, t=t_mid, t_mid=1
    )

    optim = torch.optim.Adam(params=[vmean, vlogstd], lr=lr)

    if optimizer == "adam":
        for _ in range(n_gradient_steps):
            optim.zero_grad()
            kl_div = kl_fn(vmean, vlogstd)
            kl_div.backward()
            optim.step()

    elif optimizer == "sgd":
        for _ in range(n_gradient_steps):
            vmean.requires_grad_(), vlogstd.requires_grad_()
            kl_div = kl_fn(vmean, vlogstd)
            mean_grad, logstd_grad = torch.autograd.grad(kl_div, (vmean, vlogstd))

            vmean = normalized_grad_step(vmean, mean_grad, lr=lr)
            vlogstd = normalized_grad_step(vlogstd, logstd_grad, lr=lr)

    vmean.detach_(), vlogstd.detach_()
    vsample = vmean + vlogstd.exp() * torch.randn_like(vmean)
    return vsample


def mgps_vi_step(
    i: int,
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    epsilon_net: EpsilonNet,
    log_pot: Callable[[torch.Tensor], torch.Tensor],
    optimizer: str,
    lr: float,
    gradient_steps_fn: Callable[[int], float],
    t: int,
    t_prev: int,
    tmid_fn: int,
    warm_start: bool,
):
    t_mid = tmid_fn(t_prev)
    t_mid = t_mid if t_mid <= t_prev else t_prev

    vsample_tmid = mgps_vi_tmid_step(
        i=i,
        x_t=x_t,
        pred_x0=pred_x0,
        epsilon_net=epsilon_net,
        log_pot=log_pot,
        optimizer=optimizer,
        gradient_steps_fn=gradient_steps_fn,
        lr=lr,
        t=t,
        t_prev=t_prev,
        t_mid=t_mid,
        warm_start=warm_start,
    )

    if warm_start:
        pred_x0 = mgps_vi_t0_step(
            i=i,
            x_tmid=vsample_tmid,
            epsilon_net=epsilon_net,
            log_pot=log_pot,
            optimizer=optimizer,
            gradient_steps_fn=gradient_steps_fn,
            lr=lr,
            t_mid=t_mid,
        )

        x_tprev = sample_bridge_kernel(
            x_ell=x_t, x_s=pred_x0, epsilon_net=epsilon_net, ell=t, t=t_prev, s=1
        )
    else:
        x_tprev = sample_bridge_kernel(
            x_ell=x_t,
            x_s=vsample_tmid,
            epsilon_net=epsilon_net,
            ell=t,
            t=t_prev,
            s=t_mid,
        )
        pred_x0 = epsilon_net.predict_x0(vsample_tmid, t_mid)

    return x_tprev, pred_x0


def mgps_half(
    initial_noise: torch.Tensor,
    epsilon_net: EpsilonNet,
    inverse_problem: InverseProblem,
    alpha_fn: float,
    gradient_steps_fn: dict,
    optimizer: str,
    lr: float,
    threshold: float,
    display_im: bool = False,
) -> torch.Tensor:

    log_pot = inverse_problem.log_pot
    n_steps = len(epsilon_net.timesteps)

    def tmid_fn(i):
        for rule in alpha_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i}):
                return int(rule["return"] * i)

    def grad_steps_fn(i):
        for rule in gradient_steps_fn.conditions:
            condition = rule.condition
            if condition == "default" or eval(condition, {"i": i, "n_steps": n_steps}):
                return rule["return"]

    x_tprev = initial_noise
    n_samples = x_tprev.shape[0]
    t = epsilon_net.timesteps[-1]
    pred_x0 = epsilon_net.predict_x0(initial_noise, t)

    for i in tqdm(range(n_steps - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        warm_start = t >= threshold
        x_tprev, pred_x0 = mgps_vi_step(
            i=i,
            x_t=x_tprev,
            pred_x0=pred_x0,
            epsilon_net=epsilon_net,
            log_pot=log_pot,
            optimizer=optimizer,
            gradient_steps_fn=grad_steps_fn,
            lr=lr,
            t=t,
            t_prev=t_prev,
            tmid_fn=tmid_fn,
            warm_start=warm_start,
        )

        # # XXX uncomment to view image process
        # if i % 20 == 0:
        #     if display_im:
        #         print(f"{i} / {n_steps}")
        #         for j in range(n_samples):
        #             display(pred_x0[j].clamp(-1, 1))

    return epsilon_net.predict_x0(x_tprev, epsilon_net.timesteps[1])


def kl_mvn(
    v_mean: torch.Tensor,
    v_logstd: torch.Tensor,
    mean: torch.Tensor,
    logstd: torch.Tensor,
):
    # NOTE `logstd` must be of shape (1,) and `v_logstd` of shape v_mea
    assert v_mean.shape == v_logstd.shape

    return 0.5 * (
        -2 * v_logstd.sum()
        + (torch.norm(v_mean - mean) ** 2.0 + (2.0 * v_logstd).exp().sum())
        / (2.0 * logstd).exp()
    )


@torch.no_grad()
def normalized_grad_step(var: torch.Tensor, var_grad: torch.Tensor, lr: float):
    """Apply a normalized gradient step on ``var``.

    Formula of the update::

        var = var - (lr / norm(grad)) * grad

    Note
    ----
    ``var`` must a be a leaf tensor with ``requires_grad=True``
    """
    # NOTE this is the eps used in Adam solver to prevent denominator from being zero
    eps = 1e-8
    n_samples = var.shape[0]
    shape = (n_samples, *(1,) * len(var.shape[1:]))

    grad_norm = torch.norm(var_grad.reshape(n_samples, -1), dim=-1).reshape(*shape)

    return var - (lr / (eps + grad_norm)) * var_grad
