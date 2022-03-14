import numpy as np
import torch
import random
import os
from tqdm import tqdm
from SIP.common.config import cmd_args
from SIP.debm.model.ebm import BernoulliRBM
from SIP.debm.data_util.data_loader import load_mnist
from SIP.debm.experiment.rbm_sampling import mmd
from SIP.debm.sampling.sampler import get_sampler
import matplotlib.pyplot as plt
import pickle

import tensorflow_probability as tfp
import time


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
    args.n_visible = 784
    args.input_type = 'binary'
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu)

    train_loader, test_loader, plot, viz = load_mnist(args)

    init_data = []
    for x, _ in train_loader:
        init_data.append(x)
    init_data = torch.cat(init_data, 0)
    init_mean = init_data.mean(0).clamp(.01, .99)

    model = BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
    model.to(device)

    if args.model_dump is not None:
        print('loading model from', args.model_dump)
        model.load_state_dict(torch.load(args.model_dump))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

        # train!
        for e in range(1):
            for itr, (x, _) in enumerate(train_loader):
                x = x.to(device)
                xhat = model.gibbs_sample(v=x, n_steps=args.cd)

                d = model.logp_v_unnorm(x)
                m = model.logp_v_unnorm(xhat)

                obj = d - m
                loss = -obj.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itr % args.print_every == 0:
                    print("{} {} | log p(data) = {:.4f}, log p(model) = {:.4f}, "
                          "diff = {:.4f}".format(e, itr, d.mean(), m.mean(), (d - m).mean()))

        torch.save(model.state_dict(), 'results/mnist_%d.ckpt' % args.seed)
        import sys
        sys.exit()
    gt_samples = model.gibbs_sample(n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        plot("{}/ground_truth.png".format(args.save_dir), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())

    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)
    log_mmds = {}
    log_mmds['gibbs'] = []
    ars = {}
    hops = {}
    ess = {}
    times = {}
    chains = {}
    chain = []

    times['gibbs'] = []
    start_time = time.time()
    progress_bar = tqdm(range(args.n_steps))
    for i in progress_bar:
        if i % args.print_every == 0:
            stat = kmmd.compute_mmd(new_samples, gt_samples)
            log_stat = stat.log10().item()
            log_mmds['gibbs'].append(log_stat)
            progress_bar.set_description("gibbs {} {} {}".format(i, stat, stat.log10()))
            times['gibbs'].append(time.time() - start_time)
        new_samples = model.gibbs_sample(new_samples, 1)
        if i % args.subsample == 0:
            if args.ess_statistic == "dims":
                chain.append(new_samples.cpu().numpy()[0][None])
            else:
                xc = new_samples[0][None]
                h = (xc != gt_samples).float().sum(-1)
                chain.append(h.detach().cpu().numpy()[None])
    chain = np.concatenate(chain, 0)
    chains['gibbs'] = chain
    ess['gibbs'] = get_ess(chain, args.burn_in)
    print("ess = {} +/- {}".format(ess['gibbs'].mean(), ess['gibbs'].std()))

    # temps = ['bg-1', 'bg-2', 'hb-10-1', 'gwg', 'gwg-3', 'gwg-5', 'mscorrect-3', 'mscorrect-5']
    temps = ['nb-2', 'nb-3', 'nb-5', 'mscorrect-2', 'mscorrect-3', 'mscorrect-5', 'gwg-2', 'gwg-3', 'gwg-5']
    for temp in temps:
        sampler = get_sampler(args, sampler=temp, data_dim=784)

        x = model.init_dist.sample((args.n_test_samples,)).to(device)

        log_mmds[temp] = []
        ars[temp] = []
        hops[temp] = []
        times[temp] = []
        chain = []
        cur_time = 0.
        progress_bar = tqdm(range(args.n_steps))
        for i in progress_bar:
            # do sampling and time it
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x[0][None]
                    h = (xc != gt_samples).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            if i % args.plot_every == 0 and plot is not None:
                plot("{}/temp_{}_samples_{}.png".format(args.save_dir, temp, i), x)

            if i % args.print_every == 0:
                hard_samples = x
                stat = kmmd.compute_mmd(hard_samples, gt_samples)
                log_stat = stat.log10().item()
                log_mmds[temp].append(log_stat)
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
                progress_bar.set_description("temp {}, itr = {}, log-mmd = {:.4f}, hop-dist = {:.4f}".format(temp, i, log_stat, cur_hops))
        chain = np.concatenate(chain, 0)
        ess[temp] = get_ess(chain, args.burn_in)
        chains[temp] = chain
        log = "temp {}, succ = {:.4f}, ess = {:.2f} +/- {:.2f}".format(temp, sampler.succ / sampler.count,
                                                               ess[temp].mean(), ess[temp].std())
        print(log)
        with open(os.path.join(args.save_dir, f'result_{args.seed}.txt'), 'a') as handle:
            handle.write(log + '\n')

    ess_temps = temps
    plt.clf()
    plt.boxplot([ess[temp] for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess.png".format(args.save_dir))

    plt.clf()
    plt.boxplot([np.log(ess[temp]) for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/log_ess.png".format(args.save_dir))

    plt.clf()
    plt.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in ess_temps], labels=ess_temps, showfliers=False)
    plt.savefig("{}/ess_per_sec.png".format(args.save_dir))

    plt.clf()
    for temp in temps + ['gibbs']:
        plt.plot(log_mmds[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/results.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(ars[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/ars.png".format(args.save_dir))

    plt.clf()
    for temp in temps:
        plt.plot(hops[temp], label="{}".format(temp))

    plt.legend()
    plt.savefig("{}/hops.png".format(args.save_dir))

    for temp in temps:
        plt.clf()
        plt.plot(chains[temp][:, 0])
        plt.savefig("{}/trace_{}.png".format(args.save_dir, temp))

    with open("{}/results.pkl".format(args.save_dir), 'wb') as f:
        results = {
            'ess': ess,
            'hops': hops,
            'log_mmds': log_mmds,
            'chains': chains,
            'times': times
        }
        pickle.dump(results, f)


if __name__ == '__main__':
    main(cmd_args)
