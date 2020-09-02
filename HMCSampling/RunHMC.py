# The code for running HMC

import torch
import hamiltorch
import argparse

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = torch.tensor([0.,0.,0.])
stddev = torch.tensor([0.5,1.,2.])


def log_prob(omega):
    """
        Sampling a multi-varaite Gaussian
    """
    return torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega).sum()


def log_prob(omega, c=None):
    """
        Sampling from a trucated Gaussian distribution
    """
    normal_log = torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega)
    
    if c:
        sigmoid_log = torch.nn.functional.logsigmoid(c*(1-torch.norm(omega) ** 2))
        return (normal_log + sigmoid_log).sum()
    else:
        return normal_log.sum()


def getlogprobs(c):
    """
        Wrapper for passing parameters of trucated Gaussian distribution
    """
    def f(omega):
        return log_prob(omega, c)
    return f


def main():
    parser = argparse.ArgumentParser("HMC Sampling")
    parser.add_argument("--dim", type=int, default=3, help="dimension of probability space")
    parser.add_argument("--stepsize", type=float, default=3, help="step size for generating trajectory")
    parser.add_argument("--trlen", type=int, default=3, help="number of steps in the trajectory")
    parser.add_argument("--burn", type=int, default=0, help="number of burn samples")
    parser.add_argument("--sample", type=int, default=200, help="number of samples")
    parser.add_argument("--cSig", type=float, help="cutoff parameter for sigmoid")
    parser.add_argument("--mean", nargs="+", type=float, help="mean of the Probability distribution")
    parser.add_argument("--stddev", nargs="+", type=float, help="variance of the Probability distribution")
 
    args = parser.parse_args()
    print(args)
    
    # Run HMC
    hamiltorch.set_random_seed(123)
    params_init = torch.zeros(args.dim)
#     mean = torch.tensor(args.mean)
#     stddev = torch.tensor(args.stddev) 
    
    params_hmc = hamiltorch.sample(log_prob_func=getlogprobs(c=args.cSig), params_init=params_init, num_samples=args.sample,
                               step_size=args.stepsize, num_steps_per_sample=args.trlen, burn=args.burn)

    # Get samples
    coords_all_hmc = torch.cat(params_hmc).reshape(len(params_hmc),-1)
    numSample = len(coords_all_hmc[:,1]) // args.trlen
    index = args.trlen*torch.arange(numSample)
    coords_hmc=coords_all_hmc[index]
    
    # IID sampling
    targetDis = torch.distributions.MultivariateNormal(mean, stddev.diag()**2)
    sample_iid = targetDis.sample((8 * args.sample,))
    norm_iid = torch.norm(sample_iid, dim=-1) ** 2
    ind_iid = torch.where(norm_iid <= 1)[0]
    ind_iid = ind_iid[:len(coords_hmc[:,0])]
    coords_iid = sample_iid[ind_iid]
    
    # Round off HMC and IID samples
    discritise = 1
    coords_hmc = torch.round(coords_hmc*10 **discritise)/10 **discritise
    coords_iid=torch.round(coords_iid*10 **discritise)/10 **discritise
    
    print(coords_hmc)
    
    print('True mean:            ',mean)
    print('HMC mean:            ',coords_hmc.mean(0))
    print('IID mean:            ',coords_iid.mean(0))
    
    print('HMC mean norm:            ',torch.dot(coords_hmc.mean(0),coords_hmc.mean(0)))
    print('IID mean norm:            ',torch.dot(coords_iid.mean(0),coords_iid.mean(0)))

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    