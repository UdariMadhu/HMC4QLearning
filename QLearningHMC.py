# The code for using HMC for Q-learning with continuous state space

import numpy as np
from scipy.stats import multivariate_normal
import argparse
import math
import torch

import sys

sys.path.append("HMCSampling")
from RunHMC import getHMCsamples, getlogprobs
import torch
import torch.nn as nn
import hamiltorch


EPSILON = 0.05
GAMMA = 0.95


def unfold(c, d):
    """
        Map an index to [x, y, z, ...] co-ordinates
        c (list): indexes in the targeted space
        d (list/tuple/array): shape of targeted n-dimensional space

        Note: index follows python zero-start convention i.e, c: [2, 10] will return 3rd and 11th item.
    """
    return [list(np.ndindex(*d))[i] for i in c]


def foldParallel(c, d):
    """
        Map an array of index in [x, y, z, ...] co-ordinates to an int. Folding row-wise.
        c (array): indexes in the n-dimensional space
        d (list/tuple/array): shape of targeted n-dimensional space

        Note: retuned index follows python zero-start convention i.e, c:[0, 0, 0,] will return 0 not 1.
    """
    p = np.array([np.prod(d[i + 1 :]) for i, _ in enumerate(d)])
    return np.sum(c * p, -1).astype(np.int32)


def build_spaces(args):
    srange = np.reshape(args.srange, (args.sdim, -1))
    arange = np.reshape(args.arange, (args.adim, -1))

    # Dicretization of [a, b] will include a, but exlude b when range if not multiple of step-size
    statespace = np.meshgrid(*[np.linspace(a, b, args.ssize) for (a, b) in srange])
    actionspace = np.meshgrid(*[np.linspace(a, b, args.asize) for (a, b) in arange])

    return statespace, actionspace


def get_state_transition(states, actions, statespace, biasmat, args):
    """
        Generate transition probability for each state-action pair
        Generating bias based on the states action pair
    """
    nstates = statespace.size
    ST = np.zeros([len(states), nstates])

    mp = np.meshgrid(
        *[np.arange(-args.ssize, args.ssize + 1) for _ in range(args.sdim)]
    )
    xp = np.concatenate([e.reshape(-1, 1) for e in mp], axis=-1)
    index_origin = np.array(unfold([len(xp) // 2], mp[0].shape))

    # sample probs for 2 x state_space_size in each dimension
    p = multivariate_normal.pdf(
        xp, mean=np.zeros(args.sdim), cov=args.scov * np.eye(args.sdim)
    )

    neworigins = unfold(states, statespace.shape)  # add noise later

    bias = biasmat[states, actions]
    neworigins = (np.array(neworigins) + bias) % args.ssize
    left = index_origin - neworigins

    x = np.array(list(np.ndindex(*[args.ssize for i in range(args.sdim)])))
    x = x + np.reshape(left, (-1, 1, args.sdim))
    x = foldParallel(x.reshape(-1, args.sdim), mp[0].shape)

    ST = np.reshape(p[x], ST.shape)
    ST /= np.sum(ST, axis=-1, keepdims=True)
    return ST


def sampling_policy(cs, q):
    """
        cs: current states
        q: Q-matrix (num_states x num_actions)
    """

    eps = np.random.rand(len(cs))
    s = q[cs]  # entries from q-matrix for current states

    return np.where(
        eps > EPSILON, np.argmax(s, axis=-1), np.random.randint(0, s.shape[-1], len(s))
    )


def next_state_value(cs, ca, P):
    """
        cs: current state
        ca: current action
        P: transition probability
    """
    pass


def main():
    parser = argparse.ArgumentParser("Q-learning with HMC")
    parser.add_argument("--sdim", type=int, default=2, help="dimension of state space")
    parser.add_argument("--adim", type=int, default=2, help="dimension of action space")
    parser.add_argument(
        "--srange",
        nargs="+",
        type=float,
        help="range of state space. --srange -10 10 -20 20 -30 30 -40 40 will set range of\
            1st, 2nd, 3rd, 4th dimension to [-10, 10], [-20, 20], [-30, 30], [-40, 40]",
    )
    parser.add_argument("--arange", nargs="+", type=float)
    parser.add_argument(
        "--ssize",
        type=int,
        help="Number of discrete values in each dimension of state space",
    )
    parser.add_argument(
        "--asize",
        type=int,
        help="Number of discrete values in each dimension of action space",
    )
    parser.add_argument(
        "--scov",
        type=float,
        nargs="+",
        help="Convarince of Gaussian in state space. Keep ~1/2 of --ssize",
    )
    parser.add_argument(
        "--max-bias",
        type=int,
        help="Maximum bias (shift) in the gaussian distribution for state-transition. Keep --ssize",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--seed", type=int, default=27082020)

    # HMC
    parser.add_argument(
        "--stepsize", type=float, default=3, help="step size for generating trajectory"
    )
    parser.add_argument(
        "--trlen", type=int, default=3, help="number of steps in the trajectory"
    )
    parser.add_argument("--burn", type=int, default=0, help="number of burn samples")
    parser.add_argument(
        "--hmcsample", type=int, default=200, help="number of samples in hmc"
    )
    parser.add_argument("--cSig", type=float, default=50, help="cutoff parameter for sigmoid")
    parser.add_argument("--HMCseed", type=int, default=123)
    parser.add_argument(
        "--mode", type=str, default="complete", choices=("complete", "hmc", "iid")
    )

    args = parser.parse_args()
    args.scov = np.array(args.scov)

    print(args)

    # setup
    hamiltorch.set_random_seed(args.HMCseed)
    params_init = torch.zeros(args.sdim)

    # create state-space and action-space
    statespace, actionspace = build_spaces(args)
    Q = np.zeros([statespace[0].size, actionspace[0].size])
    R = np.random.rand(statespace[0].size, actionspace[0].size)
    B = np.random.randint(
        -args.max_bias,
        args.max_bias,
        (statespace[0].size, actionspace[0].size, args.sdim),
    )
    cs = np.random.randint(0, statespace[0].size, args.samples)  # current states

    # run agent for specific steps
    for _ in range(args.steps):
        
        # randomly choosing state-action pairs
        cs = np.random.randint(0, statespace[0].size, args.samples) # current states
        ca = np.random.randint(0, actionspace[0].size, args.samples) # current states
        
#         ca = sampling_policy(cs, Q)  # current actions
        cr = R[cs, ca]  # current rewards

        T = get_state_transition(
            cs, ca, statespace[0], B, args
        )  # state-transition matrix
        
        Q_max = np.max(Q, axis=-1)
        
        if args.mode == "hmc":
            hmcorigin = (
                np.array(unfold(cs, statespace[0].shape)) + B[cs, ca]
            ) % args.ssize

            hmcorigin = [hmcorigin[:, i] for i in range(hmcorigin.shape[-1])]
            statesrange = np.reshape(args.srange, (args.sdim, -1))

            hmcstdev = torch.tensor(
                np.sqrt(args.scov)
                * (statesrange[:, 1] - statesrange[:, 0])
                * np.eye(args.sdim)
            ).float()
            hmcorigin = torch.tensor(
                np.array([s[hmcorigin] for s in statespace])
            ).transpose(1, 0)

            hmcoords = []
            for i, _ in enumerate(cs):
                params_hmc = hamiltorch.sample(
                    log_prob_func=getlogprobs(
                        c=args.cSig,
                        mean=hmcorigin[i].float(),
                        stddev=hmcstdev,
                        srange=torch.tensor(statesrange).float(),
                    ),
                    params_init=params_init,
                    num_samples=args.hmcsample,
                    step_size=args.stepsize,
                    num_steps_per_sample=args.trlen,
                    burn=args.burn,
                    srange=torch.tensor(statesrange).float(),
                )
                coords_hmc = getHMCsamples(params_hmc, args.trlen).data.numpy()

                # map to closest and convert to co-ordinates
                sr = np.array([np.linspace(a, b, args.ssize) for (a, b) in statesrange])
                coords_hmc = np.array(
                    [
                        np.argmin(np.abs(coords_hmc[:, i : i + 1] - sr[i : i + 1]), -1)
                        for i in range(args.sdim)
                    ]
                ).T

                hmcoords.append(coords_hmc)

            hmcoords = [foldParallel(i, statespace[0].shape) for i in hmcoords]

            # update with HMC
            update = cr + GAMMA * np.array(
                [
                    np.sum(T[row][hmcoords[row]] * Q_max[hmcoords[row]])
                    for row in range(T.shape[0])
                ]
            )
        
        if args.mode == "iid":
            ts = [np.random.choice(len(p), args.hmcsample, p=p) for p in T]
            update = cr + GAMMA * np.array(
                [
                    np.sum(T[row][ts[row]] * Q_max[ts[row]])
                    for row in range(T.shape[0])
                ]
            )
            
        # update step without hmc
        if args.mode == "complete":
            update = cr + GAMMA * np.sum(
                T
                * np.concatenate(
                    [np.max(Q, axis=-1, keepdims=True).T for _ in range(args.samples)]
                ),
                axis=-1,
            )

        print(
            "L2 norm of difference in Q-matrix: {:.3f}".format(
                np.linalg.norm(Q[cs, ca] - update)
            )
        )

        Q[cs, ca] = update
        cs = [np.random.choice(len(p), p=p) for p in T]  # sample next state

    rank = np.linalg.matrix_rank(Q)
    print("Rank", rank)


if __name__ == "__main__":
    main()