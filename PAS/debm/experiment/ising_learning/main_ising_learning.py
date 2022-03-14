import numpy as np
import torch
import sys
import random
from SIP.common.config import cmd_args
import SIP.debm.model.ebm as ebm
from SIP.debm.data_util.data_loader import get_from_file
from SIP.debm.experiment.rbm_sampling import mmd
from SIP.debm.sampling.sampler import get_sampler
import SIP.debm.sampling.gibbs_sampler as samplers
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


def l1(module):
    loss = 0.
    for p in module.parameters():
        loss += p.abs().sum()
    return loss


def generate_data(args, device):
    if args.data_model == "lattice_potts":
        model = ebm.LatticePottsModel(args.dim, args.n_state, args.sigma)
        sampler = samplers.PerDimMetropolisSampler(model.data_dim, args.n_out, rand=False)
    elif args.data_model == "lattice_ising":
        model = ebm.LatticeIsingModel(args.dim, args.sigma)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
    elif args.data_model == "lattice_ising_3d":
        model = ebm.LatticeIsingModel(args.dim, args.sigma, lattice_dim=3)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
        print(model.sigma)
        print(model.G)
        print(model.J)
    elif args.data_model == "er_ising":
        model = ebm.ERIsingModel(args.dim, args.degree, args.sigma)
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=False)
        print(model.G)
        print(model.J)
    else:
        raise ValueError
    torch.set_num_threads(1)
    model = model.to(device)
    samples = model.init_sample(args.n_samples).to(device)
    print("Generating {} samples from:".format(args.n_samples))
    print(model)
    for _ in tqdm(range(args.gt_steps)):
        samples = sampler.step(samples, model).detach()

    return samples.detach().cpu(), model


def main(args):
    logger = open("{}/log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu)

    if args.data_file is not None:
        train_loader, test_loader, plot, viz = get_from_file(args)
    else:
        data, data_model = generate_data(args, device)
        my_print("we have created your data, but what have you done for me lately?????")
        with open("{}/data.pkl".format(args.save_dir), 'wb') as f:
            pickle.dump(data, f)
        if args.data_model == "er_ising":
            ground_truth_J = data_model.J.detach().cpu()
            with open("{}/J.pkl".format(args.save_dir), 'wb') as f:
                pickle.dump(ground_truth_J, f)
        sys.exit()
    args.input_type = 'binary'
    if args.model == "lattice_potts":
        model = ebm.LatticePottsModel(int(args.dim), int(args.n_state), 0., 0., learn_sigma=True)
        buffer = model.init_sample(args.buffer_size)
    elif args.model == "lattice_ising":
        model = ebm.LatticeIsingModel(int(args.dim), 0., 0., learn_sigma=True)
        buffer = model.init_sample(args.buffer_size)
    elif args.model == "lattice_ising_3d":
        model = ebm.LatticeIsingModel(int(args.dim), .2, learn_G=True, lattice_dim=3)
        ground_truth_J = model.J.clone().to(device)
        model.G.data = torch.randn_like(model.G.data) * .01
        model.sigma.data = torch.ones_like(model.sigma.data)
        buffer = model.init_sample(args.buffer_size)
        plt.clf()
        plt.matshow(ground_truth_J.detach().cpu().numpy())
        plt.savefig("{}/ground_truth.png".format(args.save_dir))
    elif args.model == "lattice_ising_2d":
        model = ebm.LatticeIsingModel(int(args.dim), args.sigma, learn_G=True, lattice_dim=2)
        ground_truth_J = model.J.clone().to(device)
        model.G.data = torch.randn_like(model.G.data) * .01
        model.sigma.data = torch.ones_like(model.sigma.data)
        buffer = model.init_sample(args.buffer_size)
        plt.clf()
        plt.matshow(ground_truth_J.detach().cpu().numpy())
        plt.savefig("{}/ground_truth.png".format(args.save_dir))
    elif args.model == "er_ising":
        model = ebm.ERIsingModel(int(args.dim), 2, learn_G=True)
        model.G.data = torch.randn_like(model.G.data) * .01
        buffer = model.init_sample(args.buffer_size)
        with open(args.graph_file, 'rb') as f:
            ground_truth_J = pickle.load(f)
            plt.clf()
            plt.matshow(ground_truth_J.detach().cpu().numpy())
            plt.savefig("{}/ground_truth.png".format(args.save_dir))
        ground_truth_J = ground_truth_J.to(device)
    elif args.model == "rbm":
        model = ebm.BernoulliRBM(args.dim, args.n_hidden)
        buffer = model.init_dist.sample((args.buffer_size,))
    else:
        raise NotImplementedError

    model.to(device)
    buffer = buffer.to(device)

    def get_J():
        j = model.J
        return (j + j.t()) / 2

    sampler = get_sampler(args, data_dim=model.data_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr, weight_decay=args.weight_decay)

    itr = 0
    sigmas = []
    sq_errs = []
    rmses = []
    while itr < args.n_iters:
        for x in train_loader:        
            x = x[0].to(device)

            for k in range(args.sampling_steps):
                buffer = sampler.step(buffer.detach(), model).detach()

            logp_real = model(x).squeeze().mean()
            logp_fake = model(buffer).squeeze().mean()

            obj = logp_real - logp_fake
            loss = -obj
            loss += args.l1 * get_J().abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.G.data *= (1. - torch.eye(model.G.data.size(0))).to(model.G)

            if itr % args.print_every == 0:
                my_print("({}) log p(real) = {:.4f}, log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}".format(itr,
                                                                                              logp_real.item(),
                                                                                              logp_fake.item(),
                                                                                              obj.item(),
                                                                                              sampler._hops))
                if args.model in ("lattice_potts", "lattice_ising"):
                    my_print("\tsigma true = {:.4f}, current sigma = {:.4f}".format(args.sigma,
                                                                                    model.sigma.data.item()))
                else:
                    sq_err = ((ground_truth_J - get_J()) ** 2).sum()
                    rmse = ((ground_truth_J - get_J()) ** 2).mean().sqrt()
                    my_print("\t err^2 = {:.4f}, rmse = {:.4f}".format(sq_err, rmse))
                    #print(ground_truth_J)
                    #print(get_J())

            itr += 1

            if itr > args.n_iters:
                if args.model in ("lattice_potts", "lattice_ising"):
                    final_sigma = model.sigma.data.item()
                    with open("{}/sigma.txt".format(args.save_dir), 'w') as f:
                        f.write(str(final_sigma))
                else:
                    sq_err = ((ground_truth_J - get_J()) ** 2).sum().item()
                    rmse = ((ground_truth_J - get_J()) ** 2).mean().sqrt().item()
                    with open("{}/sq_err.txt".format(args.save_dir), 'w') as f:
                        f.write(str(sq_err))
                    with open("{}/rmse.txt".format(args.save_dir), 'w') as f:
                        f.write(str(rmse))
                sys.exit()


if __name__ == '__main__':
    main(cmd_args)
