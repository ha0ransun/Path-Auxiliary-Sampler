import torch
import numpy as np
from SIP.debm.data_util.data_loader import process_categorical_data
from SIP.debm.model.ebm import MyOneHotCategorical


class ReplayBuffer(object):
    def __init__(self, args, x_train, device):
        self.buffer_size = args.buffer_size

        self.is_categorical = args.input_type == 'gray'
        eps = 1e-2
        init_mean = x_train.mean(0) * (1. - 2 * eps) + eps

        if self.is_categorical:
            init_batch = process_categorical_data(x_train)
            eps = 1e-2 / 256
            init_mean = init_batch.mean(0) + eps
            init_mean = init_mean / init_mean.sum(-1)[:, None]
            if args.buffer_init == "mean":
                init_dist = MyOneHotCategorical(init_mean)
                buffer = init_dist.sample((args.buffer_size,))
            else:
                raise ValueError("Invalid init")
            self.init_dist = MyOneHotCategorical(init_mean.to(device))
        else:
            if args.buffer_init == "mean":
                if args.input_type == "binary":
                    init_dist = torch.distributions.Bernoulli(probs=init_mean)
                    buffer = init_dist.sample((args.buffer_size,))
                else:
                    buffer = None
                    raise ValueError("Other types of data not yet implemented")

            elif args.buffer_init == "data":
                all_inds = list(range(x_train.size(0)))
                init_inds = np.random.choice(all_inds, args.buffer_size)
                buffer = x_train[init_inds]
            elif args.buffer_init == "uniform":
                buffer = (torch.ones(args.buffer_size, *x_train.size()[1:]) * .5).bernoulli()
            else:
                raise ValueError("Invalid init")
            self.init_dist = torch.distributions.Bernoulli(probs=init_mean.to(device))
            self.reinit_dist = torch.distributions.Bernoulli(probs=torch.tensor(args.reinit_freq).to(device))                
        self.init_mean = init_mean
        self.buffer = buffer
        self.all_inds = list(range(args.buffer_size))
        self.device = device

    def sample(self, batch_size):
        buffer_inds = sorted(np.random.choice(self.all_inds, batch_size, replace=False))
        x_buffer = self.buffer[buffer_inds].to(self.device)
        if self.is_categorical:
            return x_buffer, buffer_inds
        else:
            reinit = self.reinit_dist.sample((batch_size,))
            x_reinit = self.init_dist.sample((batch_size,))
            x_fake = x_reinit * reinit[:, None] + x_buffer * (1. - reinit[:, None])
            return x_fake, buffer_inds
    
    def update(self, x, buffer_inds):
        self.buffer[buffer_inds] = x
