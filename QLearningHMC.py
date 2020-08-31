# The code for using HMC for Q-learning with continuous state space

import numpy as np
from scipy.stats import multivariate_normal
import argparse

EPSILON = 0.05
GAMMA = 0.1


def unfold(c, d):
    """
        Map an index to [x, y, z, ...] co-ordinates
        c (list): indexes in the targeted space
        d (list/tuple/array): shape of targeted n-dimensional space

        Note: index follows python zero-start convention i.e, c: [2, 10] will return 3rd and 11th item.
    """
    return [list(np.ndindex(*d))[i] for i in c]


def fold(c, d):
    """
        Map an index in [x, y, z, ...] co-ordinates to an int. Folding row-wise.
        c (list/tuple/array): index in the n-dimensional space
        d (list/tuple/array): shape of targeted n-dimensional space

        Note: retuned index follows python zero-start convention i.e, c:[0, 0, 0,] will return 0 not 1.
    """
    assert np.prod(
        [i < j for (i, j) in zip(c, d)]
    ), "index start of zero, thus can't be equal to shape in any dim"
    return np.sum([k * np.prod(d[i + 1 :]) for i, k in enumerate(c)]).astype(np.int32)


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

    sa = np.c_[states, actions]  # state-action pair

    mp = np.meshgrid(
        *[np.arange(-args.ssize, args.ssize + 1) for _ in range(args.sdim)]
    )
    xp = np.concatenate([e.reshape(-1, 1) for e in mp], axis=-1)
    index_origin = unfold([len(xp) // 2], mp[0].shape)[0]

    # sample probs for 2 x state_space_size in each dimension
    p = multivariate_normal.pdf(
        xp, mean=np.zeros(args.sdim), cov=args.scov * np.eye(args.sdim)
    )

    neworigins = unfold(states, statespace.shape)  # add noise later

    bias = biasmat[states, actions]
    neworigins = (np.array(neworigins) + bias) % args.ssize

    # ST =
    for i, _ in enumerate(states):
        m = np.meshgrid(
            *[
                np.arange(o - e, args.ssize + o - e)
                for e, o in zip(neworigins[i], index_origin)
            ]
        )
        x = np.concatenate([e.reshape(-1, 1) for e in m], axis=-1)
        x = [fold(e, mp[0].shape) for e in x]
        ST[i] = p[x]

    ST /= np.sum(ST, axis=-1, keepdims=True)
    np.random.seed(args.seed)  # switch back to original seed
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
        type=int,
        help="Convarince of Gaussian in state space. Keep ~1/2 of --ssize",
    )
    parser.add_argument(
        "--max-bias",
        type=int,
        help="Maximum bias (shift) in the gaussian distribution for state-transition. ",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--seed", type=int, default=27082020)

    args = parser.parse_args()
    print(args)

    # setup
    np.random.seed(args.seed)

    # create state-space and action space
    statespace, actionspace = build_spaces(args)
    Q = np.ones([statespace[0].size, actionspace[0].size])
    R = np.random.rand(statespace[0].size, actionspace[0].size)
    B = np.random.randint(
        -args.max_bias, args.max_bias, (statespace[0].size, actionspace[0].size)
    )
    cs = np.random.randint(0, statespace[0].size, args.samples)  # current states

    # run agent for specific steps
    for _ in range(args.steps):
        ca = sampling_policy(cs, Q)  # current actions
        cr = R[cs, ca]  # current rewards

        T = get_state_transition(
            cs, ca, statespace[0], B, args
        )  # state-transition matrix

        # update step
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


if __name__ == "__main__":
    main()
