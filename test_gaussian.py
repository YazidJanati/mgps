# %%
from typing import Dict, Callable
from dataclasses import dataclass

import torch
from torch.distributions import MultivariateNormal

from utils.gauss_tmid.utils import (
    load_g_epsilon_net,
    PosteriorLinearInvProb,
    compute_kl,
    generate_cov,
    compute_W2_multivariate_normal,
)
from utils.gauss_tmid.g_mgps import mgps_ddpm_dps
from utils.im_invp_utils import InverseProblem

import matplotlib.pyplot as plt


torch.manual_seed(135)
dim = 100
dtype = torch.float64
torch.set_default_dtype(dtype)


METRICS: Dict[str, Callable[[MultivariateNormal, MultivariateNormal], float]] = {
    "KL": compute_kl,
    "W2": compute_W2_multivariate_normal,
}


@dataclass
class Config:
    mean = torch.zeros(size=(dim,), dtype=dtype)
    cov = generate_cov(dim)
    obs_std = 1e-1
    A = torch.randn(size=(dim // 2, dim), dtype=dtype)
    n_steps = 300
    metric_name = "W2"


metric = METRICS[Config.metric_name]


epsilon_net = load_g_epsilon_net(Config.mean, Config.cov, Config.n_steps)
x_0 = epsilon_net.ideal_sample()

mean, cov = Config.mean, Config.cov
A = Config.A
obs_std = Config.obs_std

obs = A @ x_0
obs = obs + obs_std * torch.randn_like(obs)

inv_prob = InverseProblem(obs=obs, std=obs_std, A=A)

# plot prior and x_0
fig, ax = plt.subplots()

prior_samples = epsilon_net.ideal_sample(1000)

ax.scatter(prior_samples[:, 0], prior_samples[:, 1], alpha=0.3)
ax.scatter(x_0[[0]], x_0[[1]], label="x_0", color="red", marker="x")
ax.set_aspect(1)

ax.legend()
ax.set_title("Prior")

# %%
# View approximate true and approximate posterior distributions

colors = {"true post": "orange", "approx post": "green"}

# solve the problem
type_conditional = "ddim+dps"  # "exact+exact"
alpha = 0.7

mean, cov = mgps_ddpm_dps(
    epsilon_net,
    inv_prob,
    alpha,
    type_conditional,
)

# approximate posterior
approximate_posterior = MultivariateNormal(mean, cov)
posterior = PosteriorLinearInvProb(A, obs, obs_std, epsilon_net)

# plot results dimension idx_x and idx_y
idx_x, idx_y = 0, 1

# size of the markers in scatter plots
s = 10

fig = plt.figure()

gs = fig.add_gridspec(2, 2, width_ratios=(3, 1), height_ratios=(1, 3))

fig = plt.figure()
ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
ax.set(aspect=1)

# assign axes
# x-y position with-height
ax_histx = ax.inset_axes([0, 1.12, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.15, 0, 0.25, 1], sharey=ax)


ax.scatter(
    prior_samples[:, idx_x], prior_samples[:, idx_y], label="p_0", alpha=0.3, s=s
)

for name, distribution in zip(colors, (posterior, approximate_posterior)):

    samples = distribution.sample((1000,))

    ax.scatter(
        samples[:, idx_x],
        samples[:, idx_y],
        label=name,
        color=colors[name],
        s=s,
        alpha=0.3,
    )

    ax_histx.hist(
        samples[:, idx_x],
        bins="auto",
        density=False,
        color=colors[name],
        alpha=0.7,
    )
    ax_histy.hist(
        samples[:, idx_y],
        bins="auto",
        density=False,
        color=colors[name],
        orientation="horizontal",
        alpha=0.7,
    )

ax.scatter(x_0[[idx_x]], x_0[[idx_y]], label="x_0", color="red", s=20, marker="x")

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_aspect(1, adjustable="box")


fig.legend(ncols=4, bbox_to_anchor=(0.85, 1.1))
fig.suptitle(
    f"{type_conditional} with {alpha=}",
    x=0.5,
    y=1.15,
)

print(
    f"KL = {compute_kl(approximate_posterior, posterior)}\n"
    f"W2 = {compute_W2_multivariate_normal(approximate_posterior, posterior)}",
)

# %%

#################
#################
# metric evolution across configurations
# (ddim, exact) + (dps + exact)
#################
#################

# %%
# solve the problem

results = dict()
arr_alphas = torch.linspace(0.0, 1.0, 20 + 2)[1:-1]

for i, type_backward in enumerate(("ddim", "exact")):
    for j, type_potential in enumerate(("dps", "exact")):

        type_conditional = f"{type_backward} + {type_potential}"

        print(f"{type_conditional:=^45}")

        arr_kl = []
        for alpha in arr_alphas:

            vmean, vvar = mgps_ddpm_dps(
                epsilon_net, inv_prob, alpha, type_conditional=type_conditional
            )
            approximate_posterior = MultivariateNormal(vmean, vvar)

            arr_kl.append(metric(approximate_posterior, posterior))

        results[type_conditional] = arr_kl


# %%
# plot metrics across different alphas and configuration
fig, axes = plt.subplots(2, 2, sharex=True, sharey="row")

for i, type_backward in enumerate(("ddim", "exact")):
    for j, type_potential in enumerate(("dps", "exact")):
        ax = axes[i, j]
        type_conditional = f"{type_backward} + {type_potential}"

        ax.plot(arr_alphas, results[type_conditional])
        ax.set_title(type_conditional)
        ax.set_yscale("log")

fig.supxlabel("alpha")
fig.supylabel(Config.metric_name)

fig.suptitle(f"With {dim=}")
