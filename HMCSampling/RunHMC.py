# The code for running HMC
import numpy as np
import torch
import hamiltorch
import argparse

hamiltorch.set_random_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_prob(omega, c=None, mean=None, stddev=None, srange=None):
    """
        Sampling from a trucated Gaussian distribution
        mean: shape-(N, sdim)
        cov: shape-(sdim, sdim), assuming that its diagonal

        Note: This function is explicitly optimizied for isotropic distribution.
    """
    cov = stddev ** 2

    # custom-logprop
    k = len(cov)
    d = cov.diag()
    normal_log = (
        -k / 2 * np.log(2 * np.pi)
        - 0.5 * torch.log(torch.prod(d))
        - 0.5 * torch.sum((omega - mean) ** 2 / d, dim=-1)
    )
    # normal_log = torch.distributions.MultivariateNormal(mean, cov).log_prob(omega)

    if c:
        s = torch.nn.functional.logsigmoid(c * (omega - srange[:, 0].view(1, -1))).sum(
            -1
        ) + torch.nn.functional.logsigmoid(c * (srange[:, 1].view(1, -1) - omega)).sum(
            -1
        )
        return normal_log + s
    else:
        return normal_log


def getlogprobs(c, mean, stddev, srange):
    """
        Wrapper for passing parameters of trucated Gaussian distribution
    """

    def f(omega):
        return log_prob(omega, c, mean, stddev, srange)

    return f


def getHMCsamples(params_hmc):
    """
        Generating HMC samples
    """
    params_hmc = [torch.cat(p).reshape(len(p), -1) for p in params_hmc]

    return params_hmc


def main():
    parser = argparse.ArgumentParser("Q-learning with HMC")
    parser.add_argument("--sdim", type=int, default=2, help="dimension of state space")
    parser.add_argument(
        "--srange",
        nargs="+",
        type=float,
        help="range of state space. --srange -10 10 -20 20 -30 30 -40 40 will set range of\
            1st, 2nd, 3rd, 4th dimension to [-10, 10], [-20, 20], [-30, 30], [-40, 40]",
    )
    parser.add_argument(
        "--ssize",
        type=int,
        help="Number of discrete values in each dimension of state space",
    )
    parser.add_argument(
        "--scov", type=float, nargs="+", help="Convarince of Gaussian ",
    )
    parser.add_argument(
        "--mean", type=float, nargs="+", help="Mean of Gaussian ",
    )
    parser.add_argument("--samples", type=int, default=100)

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
    parser.add_argument(
        "--cSig", type=float, default=50, help="cutoff parameter for sigmoid"
    )
    parser.add_argument("--HMCseed", type=int, default=123)

    args = parser.parse_args()
    print(args)

    args.scov = np.array(args.scov)

    n = 100

    # setup
    hamiltorch.set_random_seed(args.HMCseed)
    params_init = torch.zeros(n, args.sdim)

    statesrange = np.reshape(args.srange, (args.sdim, -1))

    hmcstdev = torch.tensor(
        np.sqrt(args.scov) * (statesrange[:, 1] - statesrange[:, 0]) * np.eye(args.sdim)
    ).float()

    #     hmcmean = torch.tensor(args.mean).float()
    #     hmcmean = torch.cat([torch.zeros(1, args.sdim), torch.ones(1, args.sdim)], dim = 0)
    hmcmean = torch.zeros_like(params_init)

    params_hmc = hamiltorch.sample(
        log_prob_func=getlogprobs(
            c=args.cSig,
            mean=hmcmean,
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

    coords_hmc = getHMCsamples(params_hmc)

    # IID sampling
    #     targetDis = torch.distributions.MultivariateNormal(
    #         hmcmean, hmcstdev ** 2
    #     )

    #     sample_iid = targetDis.sample((20 * args.hmcsample,))

    #     ind_iid = torch.prod(
    #                     torch.cat(
    #                         [
    #                            ((statesrange[i, 0] <= sample_iid[:, i]).float() * (sample_iid[:, i] <= statesrange[i, 1]).float()).view(1, -1)
    #                             for i in range(args.sdim)
    #                         ]
    #                     ), dim=0
    #                 ).nonzero()[:, 0]

    #     print(ind_iid.shape, len(coords_hmc[:, 0]))
    #     coords_iid = sample_iid[ind_iid[: len(coords_hmc[:, 0])]]

    #     coords_hmc = torch.tensor(coords_hmc)

    # Round off HMC and IID samples

    #     discritise = 1
    #     coords_hmc = torch.round(coords_hmc * 10 ** discritise) / 10 ** discritise
    #     coords_iid = torch.round(coords_iid * 10 ** discritise) / 10 ** discritise

    #     print(coords_hmc.shape)
    #     print(coords_iid.shape)

    # print(coords_hmc)

    print("True mean:            ", hmcmean)
    print(
        "HMC mean:            ",
        torch.cat([c.mean(0).data.unsqueeze(0) for c in coords_hmc]),
    )
    #     print("IID mean:            ", coords_iid.mean(0))

    hmcEmpMean = torch.cat([c.mean(0).data.unsqueeze(0) for c in coords_hmc])


#     print(
#         "HMC mean norm:            ", hmcEmpMean
#     )
#     print(
#         "IID mean norm:            ", torch.dot(coords_iid.mean(0), coords_iid.mean(0))
#     )


if __name__ == "__main__":
    main()

