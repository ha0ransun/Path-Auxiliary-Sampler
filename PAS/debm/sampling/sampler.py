from random import sample
import numpy as np
from SIP.debm.sampling import gwg_sampler, multistep_sampler, block_samplers, gibbs_sampler, nb_sampler


def get_sampler(args, sampler=None, data_dim=None):
    sampler = sampler or args.sampler
    data_dim = data_dim or np.prod(args.input_size)
    if args.input_type == "binary":
        if sampler == "gwg":
            sampler = gwg_sampler.DiffSampler(data_dim, 1,
                                              fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif sampler.startswith("gwg-"):
            n_hops = int(sampler.split('-')[1])
            sampler = gwg_sampler.MultiDiffSampler(data_dim, 1, approx=True, temp=2., n_samples=n_hops)
        elif sampler.startswith("msfast-"):
            radius = int(sampler.split('-')[1])
            sampler = multistep_sampler.MSFastSampler(radius)
        elif sampler.startswith("mspath-"):
            radius = int(sampler.split('-')[1])
            sampler = multistep_sampler.MSPathSampler(radius)
        elif sampler.startswith("mscorrect-"):
            radius = int(sampler.split('-')[1])
            sampler = multistep_sampler.MSPathCorrectSampler(radius)
        elif sampler.startswith("nb-"):
            radius = int(sampler.split('-')[1])
            sampler = nb_sampler.NBSampler(radius)
        elif sampler.startswith("nba"):
            sampler = nb_sampler.NBASampler()
        elif sampler == 'dim-gibbs':
            sampler = gibbs_sampler.PerDimGibbsSampler(data_dim)
        elif sampler == "rand-gibbs":
            sampler = gibbs_sampler.PerDimGibbsSampler(data_dim, rand=True)
        elif "bg-" in sampler:
            block_size = int(sampler.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(data_dim, block_size)
        elif "hb-" in sampler:
            block_size, hamming_dist = [int(v) for v in sampler.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(data_dim, block_size, hamming_dist)
        else:
            raise ValueError("Invalid sampler...")
    else:
        if sampler == "gwg":
            sampler = gwg_sampler.DiffSamplerMultiDim(data_dim, 1, approx=True, temp=2.)
        elif sampler.startswith("mscorrect-"):
            radius = int(sampler.split('-')[1])
            sampler = multistep_sampler.MSPathCatSampler(radius)
        else:
            raise ValueError("Invalid sampler...")        
    return sampler
