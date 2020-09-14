import torch
import torch.nn as nn
from enum import Enum

from numpy import pi
from . import util


class Sampler(Enum):
    HMC = 1


def collect_gradients(log_prob, params):
    params.grad = torch.autograd.grad(log_prob.sum(), params)[0]
    return params


def gibbs(
    params, mass=None,
):

    if mass is None:
        dist = torch.distributions.Normal(
            torch.zeros_like(params), torch.ones_like(params)
        )
    else:
        if len(mass.shape) == 2:
            dist = torch.distributions.MultivariateNormal(
                torch.zeros_like(params), mass
            )
        elif len(mass.shape) == 1:
            dist = torch.distributions.Normal(torch.zeros_like(params), mass)
    return dist.sample()


def leapfrog(
    params, momentum, log_prob_func, steps=10, step_size=0.1, inv_mass=None,
):
    # params: shape (N, sdim)

    def params_grad(p):
        p = p.detach().requires_grad_()
        log_prob = log_prob_func(p)
        p = collect_gradients(log_prob, p)
        return p.grad

    ret_params = []
    ret_momenta = []
    momentum += 0.5 * step_size * params_grad(params)
    for n in range(steps):
        if inv_mass is None:
            params = params + step_size * momentum
        else:
            # Assum G is diag here so 1/Mass = G inverse
            if len(inv_mass.shape) == 2:
                params = params + step_size * torch.matmul(
                    inv_mass, momentum.view(-1, 1)
                ).view(-1)
            else:
                params = params + step_size * inv_mass * momentum
        p_grad = params_grad(params)
        momentum += step_size * p_grad
        ret_params.append(params.clone())
        ret_momenta.append(momentum.clone())
    # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
    ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
    # import pdb; pdb.set_trace()
    return ret_params, ret_momenta


def acceptance(h_old, h_new):
    return (-h_new + h_old).float()


def hamiltonian(
    params, momentum, log_prob_func, inv_mass=None,
):

    log_prob = log_prob_func(params)
    potential = -log_prob

    if inv_mass is None:
        kinetic = 0.5 * torch.sum(momentum ** 2, dim=-1)

    else:
        if len(inv_mass.shape) == 2:
            # Have not checked for parallel
            kinetic = 0.5 * torch.matmul(
                momentum.view(1, -1), torch.matmul(inv_mass, momentum.view(-1, 1))
            ).view(-1)
        else:
            kinetic = 0.5 * inv_mass * torch.sum(momentum ** 2, dim=-1)
    hamiltonian = potential + kinetic

    return hamiltonian


def sample(
    log_prob_func,
    params_init,
    num_samples=10,
    num_steps_per_sample=10,
    step_size=0.1,
    burn=0,
    inv_mass=None,
    srange=None,
):

    # Invert mass matrix once (As mass is used in Gibbs resampling step)
    mass = None
    if inv_mass is not None:
        if len(inv_mass.shape) == 2:
            mass = torch.inverse(inv_mass)
        elif len(inv_mass.shape) == 1:
            mass = 1 / inv_mass

    params = params_init.clone().requires_grad_()
    ret_params = [[c.clone()] for c in params]

    list(torch.split(params.clone(), 1, dim=0))
    num_rejected = [0 for _ in range(len(params))]

    hmcSamples = [[] for _ in range(len(params))]

#     util.progress_bar_init("Sampling HMC", num_samples, "Samples")

    for n in range(num_samples):
#         util.progress_bar_update(n)
        
        momentum = gibbs(params, mass=mass,)

        ham = hamiltonian(params, momentum, log_prob_func, inv_mass=inv_mass,)

        leapfrog_params, leapfrog_momenta = leapfrog(
            params,
            momentum,
            log_prob_func,
            steps=num_steps_per_sample,
            step_size=step_size,
            inv_mass=inv_mass,
        )

        params = leapfrog_params[-1].detach().requires_grad_()
        momentum = leapfrog_momenta[-1]

        new_ham = hamiltonian(params, momentum, log_prob_func, inv_mass=inv_mass,)

        rho = torch.min(torch.zeros_like(ham), acceptance(ham, new_ham))

        condition1 = (rho > torch.log(torch.rand(len(rho)))).float()
        condition2 = torch.prod(
            torch.cat(
                [
                    (
                        (srange[i, 0] <= params[:, i]).float()
                        * (params[:, i] <= srange[i, 1]).float()
                    ).view(1, -1)
                    for i in range(len(srange))
                ]
            ),
            dim=0,
        )

        condition = condition1 * condition2

        for c in range(len(params)):
            if condition[c]:
                if n > burn:
                    ret_params[c].append(leapfrog_params[-1][c])
                    hmcSamples[c].append(leapfrog_params[-1][c])
            else:
                num_rejected[c] += 1
                params[c] = ret_params[c][-1]
        
#     util.progress_bar_end(
# #         "Acceptance Rate {:.2f}".format(1 - num_rejected / num_samples)
#     )  # need to adapt for burn

    return hmcSamples

