import numpy as np
import torch
import random
from PAS.common.config import cmd_args
from PAS.sampling.sampler import get_sampler
from PAS.model import Ising
import matplotlib.pyplot as plt
from matplotlib import cm
import os, pickle
import time
import tensorflow_probability as tfp
from tqdm import tqdm


def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_visible = args.p ** 2
    args.input_type = 'binary'

    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    temps = ['bg-1', 'bg-2', 'hb-10-1', 'gwg-1', 'gwg-3', 'pafs-3']

    hops, times, energys, traces, ess = {}, {}, {}, {}, {}
    model = Ising(p=args.p, device=device)
    m = 50 * args.p
    for temp in temps:
        sampler = get_sampler(args, sampler=temp, data_dim=args.n_visible)
        x = model.init_dist.sample((args.n_test_samples,)).to(device)
        hops[temp], energys[temp], traces[temp] = [], [], []
        cur_time = 0.
        progress_bar = tqdm(range(args.n_steps))

        for i in progress_bar:
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            cur_hops = (x != xhat).float().sum(-1).mean().item()
            hops[temp].append(cur_hops)
            x = xhat

            with torch.no_grad():
                energy = - model(x)
                trace = model.trace(x)
            energys[temp].append(energy.cpu().numpy())
            traces[temp].append(trace.cpu().numpy())

            if i % args.print_every == 0:
                progress_bar.set_description(
                    "temp {}, itr = {}, energy = {:.4f}, hop-dist = {:.4f}".format(temp, i, energy.mean().item(),
                                                                                   cur_hops))
        energys[temp] = np.stack(energys[temp], 1)
        traces[temp] = np.stack(traces[temp], 1)
        ess[temp] = get_ess(energys[temp].T, 0.5)
        times[temp] = cur_time
        print(f"{temp}: \t ess = {ess[temp].mean()} +/- {ess[temp].std()}")


    if not os.path.isdir('results'):
        os.mkdir('results')
    plt.savefig(f'results/ising-{args.p}.pdf', bbox_inches="tight")
    with open(f'results/results-{args.p}.pkl', 'wb') as handle:
        pickle.dump([hops, times, energys, traces, ess], handle)


if __name__ == '__main__':
    main(cmd_args)