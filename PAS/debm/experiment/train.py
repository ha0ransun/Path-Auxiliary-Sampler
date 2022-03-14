import numpy as np
import torch
import random
import copy
import os
import sys
from tqdm import tqdm
import torchvision
from SIP.debm.data_util.raw_dataset import load_raw_dataset
from SIP.debm.model.dnn import get_model
from SIP.debm.model.ebm import EBM
from SIP.debm.experiment.replay_buffer import ReplayBuffer
from SIP.debm.sampling.sampler import get_sampler
from SIP.debm.data_util.data_loader import get_data_loader
import SIP.debm.sampling.ais as ais


def compute_loss(x, x_fake, args, model):
    logp_real = model(x).squeeze()            
    if args.p_control > 0:
        grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                        create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
        grad_reg = (grad_ld ** 2. / 2.).mean() * args.p_control
    else:
        grad_reg = 0.0
    logp_fake = model(x_fake).squeeze()
    obj = logp_real.mean() - logp_fake.mean()
    loss = -obj + grad_reg + args.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())
    return loss, obj.item(), logp_real.mean().item(), logp_fake.mean().item()


def main_loop(args, save_fig, eval_metric):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.gpu)

    x_train, x_val, x_test = load_raw_dataset(args)
    x_train, x_val, x_test = torch.from_numpy(x_train), torch.from_numpy(x_val).float(), torch.from_numpy(x_test).float()

    buffer = ReplayBuffer(args, x_train, device)
    net = get_model(args)
    if args.base_dist:
        model = EBM(net, buffer.init_mean, is_binary=args.input_type == 'binary')
    else:
        model = EBM(net, is_binary=args.input_type == 'binary')
    ema_model = copy.deepcopy(model)
    model.to(device)
    ema_model.to(device)

    if args.model_dump is not None:
        model_dump = os.path.join(args.save_dir, args.model_dump)
        print('loading from', model_dump)
        d = torch.load(model_dump)
        model.load_state_dict(d['model'])
        ema_model.load_state_dict(d['ema_model'])

    sampler = get_sampler(args)    
    if args.eval_only:
        test_loader = get_data_loader(x_test, args, phase='test')
        sampler = get_sampler(args, sampler='gwg')
        _, test_ll, _ = ais.evaluate(ema_model, buffer.init_dist, sampler,
                                                 test_loader, device,
                                                 args.eval_sampling_steps,
                                                 args.batch_size)
        test_ll = eval_metric(test_ll)
        print('EMA test metric: %.6f' % test_ll)
        sys.exit()
    if args.vis_only:
        test_loader = get_data_loader(x_test, args, phase='test')
        x_fake, buffer_inds = buffer.sample(args.batch_size)
        for _ in tqdm(range(args.sampling_steps)):
            x_fake_new = sampler.step(x_fake.detach(), model).detach()
            x_fake = x_fake_new
        save_fig(args, x_fake, os.path.join(args.save_dir, 'visualize_sampler-%s-steps-%d.png' % (args.sampler, args.sampling_steps)))
        for x in test_loader:
            save_fig(args, x, os.path.join(args.save_dir, 'visualize_data.png'))
            break
        sys.exit()
    val_loader = get_data_loader(x_val, args, phase='valid')
    train_loader = get_data_loader(x_train, args, phase='train')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    itr = 0
    n_epochs = 0
    best_val_ll = -1e10    
    while itr < args.n_iters:
        pbar = tqdm(train_loader)
        for x in pbar:
            x = x.to(device)
            if itr < args.warmup_iters:
                lr = args.learning_rate * float(itr) / args.warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            x_fake, buffer_inds = buffer.sample(args.batch_size)
            for _ in range(args.sampling_steps):
                x_fake_new = sampler.step(x_fake.detach(), model).detach()
                x_fake = x_fake_new
            buffer.update(x_fake.detach().cpu(), buffer_inds)
            optimizer.zero_grad()
            x.requires_grad_()
            loss, obj, score_real, score_fake = compute_loss(x, x_fake, args, model)
            loss.backward()
            optimizer.step()
            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)
            log = 'iter: %d, lr: %.2e, obj: %.2f, score_true: %.2f, score_fake: %.2f' % (itr, lr, obj, score_real, score_fake)
            pbar.set_description(log)
            itr += 1
        n_epochs += 1
        if n_epochs % args.plot_every == 0:
            save_fig(args, x, os.path.join(args.save_dir, 'data_%d.png' % n_epochs))
            save_fig(args, x_fake, os.path.join(args.save_dir, 'buffer_%d.png' % n_epochs))
        if n_epochs % args.eval_every == 0:
            logZ, val_ll, ais_samples = ais.evaluate(ema_model, buffer.init_dist, sampler,
                                                     val_loader, device,
                                                     args.eval_sampling_steps,
                                                     args.batch_size)
            val_metric = eval_metric(val_ll)
            print('eval metric: %.6f' % val_metric)
            if val_ll > best_val_ll:
                best_val_ll = val_ll
                print('saving best valid model')
                d = {}
                d['model'] = model.state_dict()
                d['ema_model'] = ema_model.state_dict()
                d['optimizer'] = optimizer.state_dict()
                torch.save(d, os.path.join(args.save_dir, "best_ckpt.pt"))
