import torchvision
import torch
import numpy as np
from functools import partial
from SIP.debm.experiment.train import main_loop
from SIP.common.config import cmd_args


def save_fig_categorical(args, x, save_path):
    ar = torch.arange(x.size(-1)).to(x.device)
    x_int = (x * ar[None, None, :]).sum(-1)
    x_int = x_int.view(x.size(0), args.input_size[0], args.input_size[1], args.input_size[2])
    torchvision.utils.save_image(x_int, save_path, normalize=True, nrow=int(x.size(0) ** .5))


def bits_per_dim(ll, args):
    nll = -ll
    num_pixels = np.prod(args.input_size)
    return (nll / num_pixels) / np.log(2)


if __name__ == '__main__':
    main_loop(cmd_args, save_fig_categorical, eval_metric=partial(bits_per_dim, args=cmd_args))
