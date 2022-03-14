from random import sample
import numpy as np
import torch
from PAS.sampling import MH_sampler, block_sampler



def get_sampler(args, sampler=None, data_dim=None, device=torch.device('cpu'), log_g=None):
    sampler = sampler
    data_dim = data_dim or np.prod(args.input_size)
    print(100 * '#')
    print(sampler)
    if log_g is None:
        if args.g_func == 'sqrt2':
            print('g(t) = sqrt(t)')
            log_g = lambda x: x / 2.0
        elif args.g_func == 'tdtp1':
            print('g(t) = t / (t + 1)')
            log_g = lambda x: x - torch.logaddexp(x, torch.zeros_like(x))
        else:
            print(f'unkonwn g func {args.g_func}, use g(t) = sqrt(t)')
            log_g = lambda x: x / 2.0
    if args.input_type == "binary":
        if sampler.startswith("rw-"):
            radius = int(sampler.split('-')[1])
            sampler = MH_sampler.RandomWalkSampler(args=args, U=radius)
        elif sampler.startswith("gwg-"):
            radius = int(sampler.split('-')[1])
            sampler = MH_sampler.GibbsWithGradientSampler(args=args, U=radius, log_g=log_g)
        elif sampler.startswith("pas-"):
            radius = int(sampler.split('-')[1])
            sampler = MH_sampler.PathAuxiliarySampler(args=args, U=radius, log_g=log_g)
        elif sampler.startswith("pafs-"):
            radius = int(sampler.split('-')[1])
            sampler = MH_sampler.PathAuxiliaryFastSampler(args=args, U=radius, log_g=log_g)
        elif "bg-" in sampler:
            block_size = int(sampler.split('-')[1])
            sampler = block_sampler.BlockGibbsSampler(data_dim, block_size)
        elif "hb-" in sampler:
            block_size, hamming_dist = [int(v) for v in sampler.split('-')[1:]]
            sampler = block_sampler.HammingBallSampler(data_dim, block_size, hamming_dist)
        else:
            raise ValueError("Invalid sampler...")
    return sampler
