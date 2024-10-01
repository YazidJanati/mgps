import tqdm
import torch
from posterior_samplers.diffusion_utils import ddim_step, EpsilonNetMCGD, EpsilonNetSVD
from ddrm.functions.denoising import efficient_generalized_steps
from utils.utils import display


def pgdm(initial_noise, inverse_problem, epsilon_net, eta=1.0):
    """
    obs = D^{-1} U^T y
    """
    obs, H_func, std, task = (
        inverse_problem.obs,
        inverse_problem.H_func,
        inverse_problem.std,
        inverse_problem.task,
    )

    if task.startswith("jpeg"):

        def pot_fn(x, t):
            diff = (obs - H_func.H(x)).detach()
            return (diff * x.reshape(x.shape[0], -1)).sum()

        sample = initial_noise
    else:
        Ut_y, diag = H_func.Ut(obs), H_func.singulars()
        epsilon_net = EpsilonNetSVD(
            net=epsilon_net.net,
            alphas_cumprod=epsilon_net.alphas_cumprod,
            timesteps=epsilon_net.timesteps,
            H_func=H_func,
            shape=initial_noise.shape,
            device=initial_noise.device,
        )

        def pot_fn(x, t):
            rsq_t = 1 - epsilon_net.alphas_cumprod[t]
            diag_cov = diag**2 + (std**2 / rsq_t)
            return (
                -0.5
                * torch.norm((Ut_y - diag * x[:, : diag.shape[0]]) / diag_cov.sqrt())
                ** 2.0
            )

        sample = initial_noise.reshape(initial_noise.shape[0], -1)
    for i in tqdm.tqdm(range(len(epsilon_net.timesteps) - 1, 1, -1)):
        t, t_prev = epsilon_net.timesteps[i], epsilon_net.timesteps[i - 1]
        sample = sample.requires_grad_()
        xhat_0 = epsilon_net.predict_x0(sample, t)
        acp_t, acp_tprev = (
            torch.tensor([epsilon_net.alphas_cumprod[t]]),
            torch.tensor([epsilon_net.alphas_cumprod[t_prev]]),
        )
        # grad_pot = grad_pot_fn(sample, t)
        grad_pot = pot_fn(xhat_0, t)
        grad_pot = torch.autograd.grad(grad_pot, sample)[0]
        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=xhat_0
        ).detach()

        if task.startswith("jpeg"):
            grad_weight = acp_t.sqrt()
            # grad_weight = acp_t * acp_tprev * (1 - acp_t) * (1 - acp_tprev)
        # HACK: this weight proved to work well in practice
        # namely alleviate the numerical errors in the SVD of the operator
        elif "blur" in task:
            # grad_weight = acp_t * acp_tprev * (1 - acp_t) * (1 - acp_tprev)
            grad_weight = acp_tprev.sqrt() * acp_t.sqrt()
        else:
            grad_weight = acp_tprev.sqrt() * acp_t.sqrt()

        sample += grad_weight * grad_pot

        # # XXX: Debug uncomment to view evolution of reconstruction
        # if i % 50 == 0:
        #     e_t = epsilon_net.predict_x0(sample[[0]], t_prev)
        #     img = H_func.V(e_t).reshape(*initial_noise.shape[1:])
        #     display(img)

    return (
        epsilon_net.predict_x0(sample, epsilon_net.timesteps[1])
        if task.startswith("jpeg")
        else (
            H_func.V(epsilon_net.predict_x0(sample, epsilon_net.timesteps[1]))
            .reshape(initial_noise.shape)
            .detach()
        )
    )
