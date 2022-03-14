from SIP.debm.experiment.train import main_loop
import torchvision
from SIP.common.config import cmd_args


def save_fig_binary(args, x, save_path):
    torchvision.utils.save_image(x.view(x.size(0),
                                 args.input_size[0], args.input_size[1], args.input_size[2]),
                                 save_path, normalize=True, nrow=int(x.size(0) ** .5))


if __name__ == '__main__':
    main_loop(cmd_args, save_fig_binary, eval_metric=lambda x: x)
