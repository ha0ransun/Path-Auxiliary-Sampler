import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np
import torch.nn.functional as F
import igraph as ig
from tqdm import tqdm


class MyOneHotCategorical:
    def __init__(self, mean):
        self.dist = torch.distributions.OneHotCategorical(probs=mean)

    def sample(self, x):
        return self.dist.sample(x)

    def log_prob(self, x):
        logits = self.dist.logits
        lp = torch.log_softmax(logits, -1)
        return (x * lp[None]).sum(-1)


class EBM(nn.Module):
    def __init__(self, net, mean=None, is_binary=True):
        super().__init__()
        self.net = net
        self.is_binary = is_binary
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            if self.is_binary:
                base_dist = torch.distributions.Bernoulli(probs=self.mean)
            else:
                base_dist = MyOneHotCategorical(self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return logp + bd


class BernoulliRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, data_mean=None):
        super().__init__()
        linear = nn.Linear(n_visible, n_hidden)
        self.W = nn.Parameter(linear.weight.data)
        self.b_h = nn.Parameter(torch.zeros(n_hidden,))
        self.b_v = nn.Parameter(torch.zeros(n_visible,))
        if data_mean is not None:
            init_val = (data_mean / (1. - data_mean)).log()
            self.b_v.data = init_val
            self.init_dist = dists.Bernoulli(probs=data_mean)
        else:
            self.init_dist = dists.Bernoulli(probs=torch.ones((n_visible,)) * .5)
        self.data_dim = n_visible

    def p_v_given_h(self, h):
        logits = h @ self.W + self.b_v[None]
        return dists.Bernoulli(logits=logits)

    def p_h_given_v(self, v):
        logits = v @ self.W.t() + self.b_h[None]
        return dists.Bernoulli(logits=logits)

    def logp_v_unnorm(self, v):
        sp = torch.nn.Softplus()(v @ self.W.t() + self.b_h[None]).sum(-1)
        vt = (v * self.b_v[None]).sum(-1)
        return sp + vt

    def logp_v_unnorm_beta(self, v, beta):
        if len(beta.size()) > 0:
            beta = beta[:, None]
        vW = v @ self.W.t() * beta
        sp = torch.nn.Softplus()(vW + self.b_h[None]).sum(-1) - torch.nn.Softplus()(self.b_h[None]).sum(-1)
        #vt = (v * self.b_v[None]).sum(-1)
        ref_dist = torch.distributions.Bernoulli(logits=self.b_v)
        vt = ref_dist.log_prob(v).sum(-1)
        return sp + vt

    def forward(self, x):
        return self.logp_v_unnorm(x)

    def _gibbs_step(self, v):
        h = self.p_h_given_v(v).sample()
        v = self.p_v_given_h(h).sample()
        return v

    def gibbs_sample(self, v=None, n_steps=2000, n_samples=None, plot=False):
        if v is None:
            assert n_samples is not None
            v = self.init_dist.sample((n_samples,)).to(self.W.device)
        if plot:
           for i in tqdm(range(n_steps)):
               v = self._gibbs_step(v)
        else:
            for i in range(n_steps):
                v = self._gibbs_step(v)
        return v


class LatticePottsModel(nn.Module):
    def __init__(self, dim, n_out=3, init_sigma=.15, init_bias=0., learn_G=False, learn_sigma=False, learn_bias=False):
        super().__init__()
        g = ig.Graph.Lattice(dim=[dim, dim], circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        self.G = nn.Parameter(torch.tensor(A).float(), requires_grad=learn_G)
        self.sigma = nn.Parameter(torch.tensor(init_sigma).float(), requires_grad=learn_sigma)
        self.bias = nn.Parameter(torch.ones((dim ** 2, n_out)).float() * init_bias, requires_grad=learn_bias)
        self.init_dist = dists.OneHotCategorical(logits=self.bias)
        self.dim = dim
        self.n_out = n_out
        self.data_dim = dim ** 2

    @property
    def mix(self):
        off_diag = -(torch.ones((self.n_out, self.n_out)) - torch.eye(self.n_out)).to(self.G) * self.sigma
        diag = torch.eye(self.n_out).to(self.G) * self.sigma
        return off_diag + diag

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    def forward2(self, x):
        assert list(x.size()[1:]) == [self.dim ** 2, self.n_out]

        xr = x.view(-1, self.n_out)
        xr_mix = (xr @ self.mix).view(x.size(0), -1, self.n_out)

        xr_mix_xr = (xr_mix[:, :, None, :] * x[:, None, :, :]).sum(-1)

        pairwise = (xr_mix_xr * self.G[None]).sum(-1).sum(-1)
        indep = (x * self.bias[None]).sum(-1).sum(-1)

        return pairwise + indep


    def forward(self, x):
        assert list(x.size()[1:]) == [self.dim ** 2, self.n_out]
        xr = x.view(-1, self.n_out)
        xr_mix = (xr @ self.mix).view(x.size(0), -1, self.n_out)

        TEST = torch.einsum("aik,ij->ajk", xr_mix, self.G)
        TEST2 = torch.einsum("aik,aik->a", TEST, x)

        indep = (x * self.bias[None]).sum(-1).sum(-1)

        # return pairwise + indep
        return TEST2 + indep


class ERIsingModel(nn.Module):
    def __init__(self, n_node, avg_degree=2, init_bias=0., learn_G=False, learn_bias=False):
        super().__init__()
        g = ig.Graph.Erdos_Renyi(n_node, float(avg_degree) / float(n_node))
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        A = torch.tensor(A).float()
        weights = torch.randn_like(A) * ((1. / avg_degree) ** .5)
        weights = weights * (1 - torch.tril(torch.ones_like(weights)))
        weights = weights + weights.t()

        self.G = nn.Parameter(A * weights, requires_grad=learn_G)
        self.bias = nn.Parameter(torch.ones((n_node,)).float() * init_bias, requires_grad=learn_bias)
        self.init_dist = dists.Bernoulli(logits=2 * self.bias)
        self.data_dim = n_node

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return self.G

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)

        x = (2 * x) - 1

        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        return xgx + b


class LatticeIsingModel(nn.Module):
    def __init__(self, dim, init_sigma=.15, init_bias=0., learn_G=False, learn_sigma=False, learn_bias=False,
                 lattice_dim=2):
        super().__init__()
        g = ig.Graph.Lattice(dim=[dim] * lattice_dim, circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        self.G = nn.Parameter(torch.tensor(A).float(), requires_grad=learn_G)
        self.sigma = nn.Parameter(torch.tensor(init_sigma).float(), requires_grad=learn_sigma)
        self.bias = nn.Parameter(torch.ones((dim ** lattice_dim,)).float() * init_bias, requires_grad=learn_bias)
        self.init_dist = dists.Bernoulli(logits=2 * self.bias)
        self.data_dim = dim ** lattice_dim

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return self.G * self.sigma

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)

        x = (2 * x) - 1

        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        return xgx + b
