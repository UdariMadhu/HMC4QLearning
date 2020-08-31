import torch
import torch.nn as nn
from enum import Enum

from numpy import pi
from . import util


class Sampler(Enum):
    HMC = 1
    RMHMC = 2
    HMC_NUTS = 3
    # IMPORTANCE = 3
    # MH = 4


class Integrator(Enum):
    EXPLICIT = 1
    IMPLICIT = 2
    S3       = 3


class Metric(Enum):
    HESSIAN = 1
    SOFTABS = 2
    JACOBIAN_DIAG = 3

def collect_gradients(log_prob, params):
    if isinstance(log_prob, tuple):
        log_prob[0].backward()
        params_list = list(log_prob[1])
        # params = util.flatten(params_list)
        params = torch.cat([p.flatten() for p in params_list])
        params.grad = torch.cat([p.grad.flatten() for p in params_list])
    else:
        params.grad = torch.autograd.grad(log_prob,params)[0]
        # log_prob.backward()
        # import pdb; pdb.set_trace()
    return params


def gibbs(params, sampler=Sampler.HMC, log_prob_func=None, jitter=None, normalizing_const=1., softabs_const=None, mass=None, metric=Metric.HESSIAN):
    if sampler == Sampler.RMHMC:
        dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), fisher(params, log_prob_func, jitter, normalizing_const, softabs_const, metric)[0])
    elif mass is None:
        dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    else:
        if len(mass.shape) == 2:
            dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), mass)
        elif len(mass.shape) == 1:
            dist = torch.distributions.Normal(torch.zeros_like(params), mass)
    return dist.sample()


def leapfrog(params, momentum, log_prob_func, steps=10, step_size=0.1, jitter=0.01, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, fixed_point_threshold=1e-20, fixed_point_max_iterations=6, jitter_max_tries=10, inv_mass=None, ham_func=None, sampler=Sampler.HMC, integrator=Integrator.IMPLICIT, metric=Metric.HESSIAN, debug=False):
    if sampler == Sampler.HMC:
        def params_grad(p):
            p = p.detach().requires_grad_()
            log_prob = log_prob_func(p)
            # log_prob.backward()
            p = collect_gradients(log_prob, p)
            return p.grad
        ret_params = []
        ret_momenta = []
        momentum += 0.5 * step_size * params_grad(params)
        for n in range(steps):
            if inv_mass is None:
                params = params + step_size * momentum
            else:
                #Assum G is diag here so 1/Mass = G inverse
                if len(inv_mass.shape) == 2:
                    params = params + step_size * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
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

    else:
        raise NotImplementedError()


def acceptance(h_old, h_new):
    # if isinstance(h_old, tuple):
    #     return float(-torch.log(h_new[0]) + torch.log(h_old[0]))
    # else:
    # return float(-torch.log(h_new) + torch.log(h_old))
    return float(-h_new + h_old)


def hamiltonian(params, momentum, log_prob_func, jitter=0.01, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, inv_mass=None, ham_func=None, sampler=Sampler.HMC, integrator=Integrator.EXPLICIT, metric=Metric.HESSIAN):

    if sampler == Sampler.HMC:
        log_prob = log_prob_func(params)

        if util.has_nan_or_inf(log_prob):
            print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
            raise util.LogProbError()

        potential = -log_prob
        if inv_mass is None:
            kinetic = 0.5 * torch.dot(momentum, momentum)
        else:
            if len(inv_mass.shape) == 2:
                kinetic = 0.5 * torch.matmul(momentum.view(1,-1),torch.matmul(inv_mass,momentum.view(-1,1))).view(-1)
            else:
                kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)
        hamiltonian = potential + kinetic
        # hamiltonian = hamiltonian

    else:
        raise NotImplementedError()
    # if not tup:
    return hamiltonian
    # else:
    #     model_parameters = hamiltonian[1]
    #     return hamiltonian[0], model_parameters


def sample(log_prob_func, params_init, num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=0, jitter=None, inv_mass=None, normalizing_const=1., softabs_const=None, explicit_binding_const=100, fixed_point_threshold=1e-5, fixed_point_max_iterations=1000, jitter_max_tries=10, sampler=Sampler.HMC, integrator=Integrator.IMPLICIT, metric=Metric.HESSIAN, debug=False, desired_accept_rate=0.8):

    if params_init.dim() != 1:
        raise RuntimeError('params_init must be a 1d tensor.')

    if burn >= num_samples:
        raise RuntimeError('burn must be less than num_samples.')

    # Invert mass matrix once (As mass is used in Gibbs resampling step)
    mass = None
    if inv_mass is not None:
        if len(inv_mass.shape) == 2:
            mass = torch.inverse(inv_mass)
        elif len(inv_mass.shape) == 1:
            mass = 1/inv_mass

    params = params_init.clone().requires_grad_()
    ret_params = [params.clone()]
    num_rejected = 0
    # if sampler == Sampler.HMC:
    util.progress_bar_init('Sampling ({}; {})'.format(sampler, integrator), num_samples, 'Samples')
    for n in range(num_samples):
        util.progress_bar_update(n)
        try:
            momentum = gibbs(params, sampler=sampler, log_prob_func=log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, metric=metric, mass=mass)

            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric, inv_mass=inv_mass)

            leapfrog_params, leapfrog_momenta = leapfrog(params, momentum, log_prob_func, sampler=sampler, integrator=integrator, steps=num_steps_per_sample, step_size=step_size, inv_mass=inv_mass, jitter=jitter, jitter_max_tries=jitter_max_tries, fixed_point_threshold=fixed_point_threshold, fixed_point_max_iterations=fixed_point_max_iterations, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, metric=metric, debug=debug)

            params = leapfrog_params[-1].detach().requires_grad_()
            momentum = leapfrog_momenta[-1]
            new_ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric, inv_mass=inv_mass)



            # new_ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric)
            rho = min(0., acceptance(ham, new_ham))
            if debug:
                print('Current Hamiltoninian: {}, Proposed Hamiltoninian: {}'.format(ham,new_ham))

            if rho >= torch.log(torch.rand(1)) and torch.dot(params,params)<=1:
#             if rho >= torch.log(torch.rand(1)):
                if debug:
                    print('Accept rho: {}'.format(rho))
                # ret_params.append(params)
                if n > burn:
                    ret_params.extend(leapfrog_params)
            else:
                num_rejected += 1
                params = ret_params[-1]
#                 if n > burn:
#                     leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
#                     ret_params.extend(leapfrog_params) # append the current sample to the chain
                if debug:
                    print('REJECT')

        except util.LogProbError:
            num_rejected += 1
            params = ret_params[-1]
            if n > burn:
                leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                ret_params.extend(leapfrog_params)
            if debug:
                print('REJECT')

        # gc.collect()

    util.progress_bar_end('Acceptance Rate {:.2f}'.format(1 - num_rejected/num_samples)) #need to adapt for burn
    if debug:
        return list(map(lambda t: t.detach(), ret_params)), step_size
    else:
        return list(map(lambda t: t.detach(), ret_params))


    def log_prob_func(params):
        # model.zero_grad()
        # params is flat
        # Below we update the network weights to be params
        params_unflattened = util.unflatten(model, params)

        i_prev = 0
        l_prior = torch.zeros_like( params[0], requires_grad=True) # Set l2_reg to be on the same device as params
        for weights, index, shape, dist in zip(model.parameters(), params_flattened_list, params_shape_list, dist_list):
            # weights.data = params[i_prev:index+i_prev].reshape(shape)
            w = params[i_prev:index+i_prev]
            l_prior = dist.log_prob(w).sum() + l_prior
            i_prev += index

        # Sample prior if no data
        if x is None:
            # print('hi')
            return l_prior#/y.shape[0]


        output = fmodel(x,params=params_unflattened)

        if model_loss is 'binary_class':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out *(crit(output, y))
        elif model_loss is 'multi_class_linear_output':
    #         crit = nn.MSELoss(reduction='mean')
            crit = nn.CrossEntropyLoss(reduction='sum')
    #         crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out *(crit(output, y.long().view(-1)))
            # ll = - tau_out *(torch.nn.functional.nll_loss(output, y.long().view(-1)))
        elif model_loss is 'multi_class_log_softmax_output':
            ll = - tau_out *(torch.nn.functional.nll_loss(output, y.long().view(-1)))

        elif model_loss is 'regression':
            # crit = nn.MSELoss(reduction='sum')
            ll = - 0.5 * tau_out * ((output - y) ** 2).sum(0)#sum(0)

        elif callable(model_loss):
            # Assume defined custom log-likelihood.
            ll = - model_loss(output, y).sum(0)
        else:
            raise NotImplementedError()
        if predict:
            return ll + l_prior, output
        else:
            return ll + l_prior

    return log_prob_func


