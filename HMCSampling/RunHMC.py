# The code for running HMC

import torch
import hamiltorch
import argparse

hamiltorch.set_random_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_prob(omega, c=None, mean=None, stddev=None, srange=None):
    """
        Sampling from a trucated Gaussian distribution
    """

    if len(stddev.shape) == 1:
        cov = torch.diag(stddev ** 2)
        print("Create Covariance matrix, assuming input was stddev")
    else:
        cov = stddev ** 2

    normal_log = torch.distributions.MultivariateNormal(mean, cov).log_prob(omega)
    if c:
        s = torch.nn.functional.logsigmoid(
            c * (omega - srange[:, 0])
        ) + torch.nn.functional.logsigmoid(c * (srange[:, 1] - omega))
        return (normal_log + s).sum()
    else:
        return normal_log.sum()


def getlogprobs(c, mean, stddev, srange):
    """
        Wrapper for passing parameters of trucated Gaussian distribution
    """

    def f(omega):
        return log_prob(omega, c, mean, stddev, srange)

    return f


def getHMCsamples(params_hmc, trjlen):
    """
        Generating HMC samples
    """
    coords_all_hmc = torch.cat(params_hmc).reshape(len(params_hmc), -1)
    numSample = len(coords_all_hmc[:, 1]) // trjlen
    index = trjlen * torch.arange(numSample)

    return coords_all_hmc[index]


def main():
    parser = argparse.ArgumentParser("HMC Sampling")
    parser.add_argument(
        "--dim", type=int, default=3, help="dimension of probability space"
    )
    parser.add_argument(
        "--stepsize", type=float, default=3, help="step size for generating trajectory"
    )
    parser.add_argument(
        "--trlen", type=int, default=3, help="number of steps in the trajectory"
    )
    parser.add_argument("--burn", type=int, default=0, help="number of burn samples")
    parser.add_argument("--sample", type=int, default=200, help="number of samples")
    parser.add_argument("--cSig", type=float, help="cutoff parameter for sigmoid")
    parser.add_argument(
        "--mean", nargs="+", type=float, help="mean of the Probability distribution"
    )
    parser.add_argument(
        "--stddev",
        nargs="+",
        type=float,
        help="variance of the Probability distribution",
    )

    args = parser.parse_args()
    print(args)

    args.mean = torch.tensor(args.mean)
    args.stddev = torch.tensor(args.stddev)
    # mean = torch.tensor([0.0, 0.0, 0.0])
    # stddev = torch.tensor([0.5, 1.0, 2.0])

    # Run HMC
    hamiltorch.set_random_seed(123)
    params_init = torch.zeros(args.dim)
    #     mean = torch.tensor(args.mean)
    #     stddev = torch.tensor(args.stddev)

    params_hmc = hamiltorch.sample(
        log_prob_func=getlogprobs(c=args.cSig, mean=args.mean, stddev=args.stddev),
        params_init=params_init,
        num_samples=args.sample,
        step_size=args.stepsize,
        num_steps_per_sample=args.trlen,
        burn=args.burn,
    )

    # Get samples
    coords_hmc = getHMCsamples(params_hmc, args.trlen)

    # IID sampling
    targetDis = torch.distributions.MultivariateNormal(
        args.mean, args.stddev.diag() ** 2
    )
    sample_iid = targetDis.sample((8 * args.sample,))
    norm_iid = torch.norm(sample_iid, dim=-1) ** 2
    ind_iid = torch.where(norm_iid <= 1)[0]
    ind_iid = ind_iid[: len(coords_hmc[:, 0])]
    coords_iid = sample_iid[ind_iid]

    # Round off HMC and IID samples
    discritise = 1
    coords_hmc = torch.round(coords_hmc * 10 ** discritise) / 10 ** discritise
    coords_iid = torch.round(coords_iid * 10 ** discritise) / 10 ** discritise

    print(coords_hmc)

    print("True mean:            ", args.mean)
    print("HMC mean:            ", coords_hmc.mean(0))
    print("IID mean:            ", coords_iid.mean(0))

    print(
        "HMC mean norm:            ", torch.dot(coords_hmc.mean(0), coords_hmc.mean(0))
    )
    print(
        "IID mean norm:            ", torch.dot(coords_iid.mean(0), coords_iid.mean(0))
    )


if __name__ == "__main__":
    main()

