import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as tr
import pickle as cp

def identity(x):
    return x

class MyTensorDataset(Dataset):

    def __init__(self, data, transform) -> None:
        super(MyTensorDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self):
        return self.data.size(0)


def get_binary_data_loader(x, args, phase):    
    is_train = phase == 'train'
    if args.dynamic_binarization:
        fn_transform = torch.bernoulli
    else:
        fn_transform = identity
    d = MyTensorDataset(x, fn_transform)
    loader = DataLoader(d, batch_size=args.batch_size,
                        shuffle=is_train, drop_last=is_train)
    return loader


def process_categorical_data(data):
    x_int = (data * 256).long()
    x_oh = torch.nn.functional.one_hot(x_int, 256).float()
    return x_oh    


def get_categorical_data_loader(x, args, phase):
    is_train = phase == 'train'
    d = MyTensorDataset(x, process_categorical_data)
    loader = DataLoader(d, batch_size=args.batch_size,
                        shuffle=is_train, drop_last=is_train)
    return loader

def get_data_loader(x, args, phase):
    if args.input_type == 'binary':
        return get_binary_data_loader(x, args, phase)
    else:
        return get_categorical_data_loader(x, args, phase)


def get_from_file(args):
    with open(args.data_file, 'rb') as f:
        x = cp.load(f)
    x = torch.tensor(x).float()
    train_data = TensorDataset(x)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
    test_loader = train_loader
    viz = None
    if args.model == "lattice_ising" or args.model == "lattice_ising_2d":
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
                                                            p, normalize=False, nrow=int(x.size(0) ** .5))
    elif args.model == "lattice_potts":
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), args.dim, args.dim, 3).transpose(3, 1),
                                                            p, normalize=False, nrow=int(x.size(0) ** .5))
    else:
        plot = lambda p, x: None
    return train_loader, test_loader, plot, viz


def load_mnist(args):
    transform = tr.Compose([tr.Resize(args.img_size), tr.ToTensor(), lambda x: (x > .5).float().view(-1)])
    train_data = torchvision.datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root=args.data_dir, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=True, drop_last=True)
    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.img_size, args.img_size),
                                                        p, normalize=True, nrow=sqrt(x.size(0)))
    encoder = None
    viz = None
    return train_loader, test_loader, plot, viz