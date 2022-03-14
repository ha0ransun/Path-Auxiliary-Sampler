from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import pickle as cp


cmd_opt = argparse.ArgumentParser(description="Argparser for mcmc", allow_abbrev=False)

cmd_opt.add_argument("--data_dir", default=".", help="data dir")
cmd_opt.add_argument("--dataset_name", default=None, help="dataset name")
cmd_opt.add_argument("--save_dir", default=".", help="save dir")
cmd_opt.add_argument('--model', type=str, default='mlp-256')
cmd_opt.add_argument('--buffer_init', type=str, default='mean')
cmd_opt.add_argument('--model_dump', type=str, default=None)
cmd_opt.add_argument('--base_dist', action='store_true')
cmd_opt.add_argument('--eval_only', action='store_true')
cmd_opt.add_argument('--vis_only', action='store_true')
cmd_opt.add_argument("--n_iters", default=50000, type=int, help="num iterations")
cmd_opt.add_argument("--warmup_iters", default=10000, type=int, help="warmup iterations")

cmd_opt.add_argument('--reinit_freq', type=float, default=0.0)
cmd_opt.add_argument('--weight_decay', type=float, default=.0)
cmd_opt.add_argument('--p_control', type=float, default=0.0)
cmd_opt.add_argument('--sampler', type=str, default='gwg')
cmd_opt.add_argument('--eval_sampling_steps', type=int, default=100)
cmd_opt.add_argument('--l2', type=float, default=0.0)
cmd_opt.add_argument('--ema', type=float, default=0.999)
cmd_opt.add_argument("--buffer_size", default=1000, type=int, help="pcd buffer size")
cmd_opt.add_argument("--seed", default=1, type=int, help="random seed")
cmd_opt.add_argument("--eval_every", default=10, type=int, help="eval every x epoch")
cmd_opt.add_argument("--plot_every", default=2, type=int, help="plot every x epoch")
cmd_opt.add_argument("--sampling_steps", default=50, type=int, help="number of mcmc steps per gradient update")
cmd_opt.add_argument("--batch_size", default=100, type=int, help="batch size")
cmd_opt.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
cmd_opt.add_argument("--gpu", default=-1, type=int, help="device")

cmd_opt.add_argument('--proj_dim', type=int, default=4)
cmd_opt.add_argument('--g_func', type=str, default=None)
cmd_opt.add_argument('--ess_ratio', type=float, default=0.5)

# rbm def
cmd_opt.add_argument('--n_steps', type=int, default=40000)
cmd_opt.add_argument('--n_samples', type=int, default=500)
cmd_opt.add_argument('--n_test_samples', type=int, default=100)
cmd_opt.add_argument('--gt_steps', type=int, default=10000)

cmd_opt.add_argument('--n_hidden', type=int, default=25)
cmd_opt.add_argument('--n_visible', type=int, default=100)
cmd_opt.add_argument('--print_every', type=int, default=10)
# for rbm training
cmd_opt.add_argument('--rbm_lr', type=float, default=.001)
cmd_opt.add_argument('--cd', type=int, default=100)
cmd_opt.add_argument('--img_size', type=int, default=28)
# for ess
cmd_opt.add_argument('--subsample', type=int, default=1)
cmd_opt.add_argument('--burn_in', type=float, default=.1)
cmd_opt.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])

# for ising

cmd_opt.add_argument('--dim', type=int, default=10)
cmd_opt.add_argument('--p', type=int, default=20)
cmd_opt.add_argument('--sigma', type=float, default=.1)
cmd_opt.add_argument('--l1', type=float, default=0)
cmd_opt.add_argument('--data_model', type=str, default='rbm')
cmd_opt.add_argument('--data_file', type=str, default=None)

# for fhmm
cmd_opt.add_argument('--L', type=int, default=100)
cmd_opt.add_argument('--K', type=int, default=10)
cmd_opt.add_argument('--alpha', type=float, default=.1)
cmd_opt.add_argument('--beta', type=float, default=.8)

cmd_args, _ = cmd_opt.parse_known_args()

print(cmd_args)
