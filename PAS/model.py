import torch
import numpy as np
import networkx as nx
from torch import Tensor, DoubleTensor, LongTensor
import torch.distributions as dists
from torch.types import Tuple
from torch_scatter import scatter_sum
from torch.nn import Module
from tqdm import tqdm
import torch.nn as nn
import random


class Ising(Module):
    def __init__(self, p=100, mu=2.0, sigma=3.0, lamda=1.0, seed=0, device=torch.device("cpu")):
        super().__init__()
        self.p = p
        self.device = device
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.n_w = (2 * torch.rand(p, p, device=device) - 1) * sigma
        for i in range(p):
            for j in range(p):
                self.n_w[i, j] += self._weight((i, j), p, mu)
        self.e_w_h = - lamda * torch.ones(p, p - 1, device=device)
        self.e_w_v = - lamda * torch.ones(p - 1, p, device=device)
        self.init_dist = dists.Bernoulli(probs=torch.ones((p ** 2,)) * .5)
        self.x0 = self.init_dist.sample((1,)).to(self.device)

    def _weight(self, n, p, mu):
        if (n[0] / p - 0.5) ** 2 + (n[1] / p - 0.5) ** 2 < 0.5 / np.pi:
            return - mu
        else:
            return mu

    def forward(self, x):
        x = x.view(-1, self.p, self.p)
        message = self.aggr(x)
        message = message / 2 + self.n_w
        return - ((2 * x - 1) * message).sum(dim=[1, 2])

    def trace(self, x):
        return (x - self.x0).abs().sum(dim=1)

    def change(self, x):
        x = x.view(-1, self.p, self.p)
        message = self.aggr(x)
        message += self.n_w
        return - ((2 - 4 * x) * message).view(-1, self.p ** 2)

    def aggr(self, x):
        message = torch.zeros_like(x)
        message[:, :-1, :] += (2 * x[:, 1:, :] - 1) * self.e_w_v
        message[:, 1:, :] += (2 * x[:, :-1, :] - 1) * self.e_w_v
        message[:, :, :-1] += (2 * x[:, :, 1:] - 1) * self.e_w_h
        message[:, :, 1:] += (2 * x[:, :, :-1] - 1) * self.e_w_h
        return message


class FHMM(Module):
    def __init__(self, L=1000, K=10, sigma=0.5, alpha=0.1, beta=0.9, seed=0, device=torch.device("cpu")) -> None:
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.L = L
        self.K = K
        self.sigma = sigma
        self.device = device
        self.alpha = torch.FloatTensor([alpha]).to(device)
        self.beta = torch.FloatTensor([beta]).to(device)
        self.W = torch.randn((K, 1)).to(device)
        self.b = torch.randn((1, 1)).to(device)
        self.X = self.sample_X(seed)
        self.Y = self.sample_Y(self.X, seed)
        self.info = f"fhmm_L-{L}_K-{K}"
        self.init_dist = dists.Bernoulli(probs=torch.ones((L * K,)) * .5)
        self.P_X0 = torch.distributions.Bernoulli(probs=self.alpha)
        self.P_XC = torch.distributions.Bernoulli(logits=1 - self.beta)
        self.x0 = self.init_dist.sample((1,)).to(device)

    def sample_X(self, seed):
        torch.manual_seed(seed)
        X = torch.ones((self.L, self.K)).to(self.device)
        X[0] = torch.bernoulli(X[0] * self.alpha)
        for l in range(1, self.L):
            p = self.beta * X[l - 1] + (1 - self.beta) * (1 - X[l - 1])
            X[l] = torch.bernoulli(p)
        return X

    def sample_Y(self, X, seed):
        torch.manual_seed(seed)
        return torch.randn((self.L, 1)).to(self.device) * self.sigma + X @ self.W + self.b

    def forward(self, x):
        x = x.view(-1, self.L, self.K)
        x_0 = x[:, 0, :]
        x_cur = x[:, :-1, :]
        x_next = x[:, 1:, :]
        x_c = x_cur * (1 - x_next) + (1 - x_cur) * x_next
        logp_x = - self.P_X0.log_prob(x_0).sum(-1) - self.P_XC.log_prob(x_c).sum(dim=[1, 2])
        logp_y = - (self.Y - x @ self.W - self.b).square().sum(dim=[1, 2]) / (2 * self.sigma ** 2)
        return logp_x + logp_y

    def error(self, x):
        x = x.view(-1, self.L, self.K)
        logp_y = - (self.Y - x @ self.W - self.b).square().sum(dim=[1, 2]) / (2 * self.sigma ** 2)
        return - logp_y

    def trace(self, x):
        return (x - self.x0).abs().sum(dim=1)

    def change(self, x):
        x = x.view(-1, self.L, self.K)
        x_0 = x[:, 0, :]
        x_cur = x[:, :-1, :]
        x_next = x[:, 1:, :]
        x_c = x_cur * (1 - x_next) + (1 - x_cur) * x_next

        change_x = torch.zeros_like(x)
        change_y = torch.zeros_like(x)

        change_x[:, 0, :] += 2 * self.P_X0.log_prob(x_0) - 1
        change_x[:, :-1, :] += 2 * self.P_XC.log_prob(x_c) - 1
        change_x[:, 1:, :] += 2 * self.P_XC.log_prob(x_c) - 1

        Y = self.Y - x @ self.W - self.b
        Y_change = - (1 - 2 * x) * self.W.T
        change_y = - (Y + Y_change).square() + Y.square()

        change = (change_x + change_y / (2 * self.sigma ** 2)).view(-1, self.L * self.K)
        return change


class Permutation(object):
    def __init__(self, p=100, sigma=1.0, seed=0, device=torch.device("cpu")):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = device
        self.w = sigma * torch.randn((p, p)).to(device)
        self._num_nodes = p
        self.info = f"permutation_p-{p}_sigma-{sigma}"
        self.i1 = LongTensor([j for i in range(p - 1) for j in range(i + 1)]).to(device)
        self.i2 = LongTensor([i for i in range(1, p) for _ in range(1, i + 1)]).to(device)

    def energy(self, x):
        if len(x.shape) == 1:
            idx1 = x
            idx2 = torch.cat([x[1:], x[-1:]], dim=0)
            res = self.w[idx1, idx2].sum()
        else:
            Res = []
            for j in range(x.shape[0]):
                idx1 = x[j]
                idx2 = torch.cat([x[j, 1:], x[j, -1:]], dim=0)
                Res.append(self.w[idx1, idx2].sum())
            res = torch.stack(Res, dim=0)
        return res

    def change(self, x):
        i, j = x[self.i1], x[self.i2]
        i_m, j_m = x[(self.i1 - 1) % self._num_nodes], x[(self.i2 - 1) % self._num_nodes]
        i_p, j_p = x[(self.i1 + 1) % self._num_nodes], x[(self.i2 + 1) % self._num_nodes]
        res = self.w[i_m, j] + self.w[j, i_p] + self.w[j_m, i] + self.w[i, j_p] \
            - self.w[i_m, i] - self.w[i, i_p] - self.w[j_m, j] - self.w[j, j_p]
        return res

    def flip_state(self, x, idx):
        z = x.clone()
        i, j = self.i1[idx], self.i2[idx]
        z[i], z[j] = z[j], z[i]
        # for id in idx:
        #     i, j = self.i1[id], self.i2[id]
        #     z[i], z[j] = z[j], z[i]
        return z

    def init_state(self):
        return torch.arange(self._num_nodes, dtype=torch.int64).to(self.device)

    def trace(self, x):
        return 0

    @property
    def num_nodes(self):
        return int(self._num_nodes * (self._num_nodes - 1) / 2)


class BMM(object):
    def __init__(self, p=100, m=10, seed=0, device=torch.device("cpu")):
        """
        theta: (p, m) Tensor
        normlizer: (1, m) Tensor
        """
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.device = device
        self.info = f"bmm_p-{p}_m-{m}_seed-{seed}_{device}"
        theta = torch.ones((p, m))
        self.k = p // m
        for i in range(m):
            theta[i * self.k: (i+1)*self.k, i] *= -1
        theta += torch.randn_like(theta) * 0.1
        self.theta = theta.to(self.device)
        self.normalizer = torch.log(1 + torch.exp(-theta)).sum(dim=0, keepdim=True).to(self.device)
        self._num_nodes = p
        # self.debug = None
        # self.count = 0

    def init_state(self):
        return torch.ones(self.num_nodes).to(self.device)

    def energy(self, x : Tensor):
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        elif len(x.shape) == 2:
            x = x.T
        logits = - self.theta.T @ x - self.normalizer.T
        res = - torch.logsumexp(logits, dim=0)
        # if self.debug is None:
        #     self.debug = torch.argmax(logits)
        #     # print(self.debug)
        # elif self.debug != torch.argmax(logits):
        #     self.debug = torch.argmax(logits)
        #     self.count += 1
            # print(self.debug, self.count)
        return res

    def grad(self, x : Tensor):
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        elif len(x.shape) == 2:
            x = x.T
        logits = - self.theta.T @ x - self.normalizer.T
        res = self.theta @ torch.softmax(logits, dim=0)
        return res.squeeze()

    def energy_grad(self, x : Tensor):
        return self.energy(x), self.grad(x)

    def change(self, x : Tensor):
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        logits = - self.theta.T @ x - self.normalizer.T
        logits_change = - self.theta.T * (1 - 2 * x.T)
        new_energy = - torch.logsumexp(logits + logits_change, dim=0)
        energy = - torch.logsumexp(logits, dim=0)
        res = new_energy - energy
        return res


    def flip_state(self, x : Tensor, idx : Tuple):
        z = x.clone()
        for i in idx:
            z[i] = 1 - z[i]
        return z

    def delta(self, x : Tensor):
        return 1 - 2 * x

    def trace(self, x : Tensor):
        # return x[:self.k].sum()
        assert len(x.shape) == 1
        x = x.unsqueeze(-1)
        logits = - self.theta.T @ x - self.normalizer.T
        return torch.argmax(logits).item()
        # return (logits[0] - torch.logsumexp(logits, dim=0)).item()
        # assert len(x.shape) == 1
        # x = x.unsqueeze(-1)
        # logits = - self.theta.T @ x - self.normalizer.T
        # prob = torch.exp(logits)
        # return (prob * torch.log(prob)).sum().item()
        # return (logits[0] - torch.logsumexp(logits, dim=0)).item()


    @property
    def num_nodes(self):
        return self._num_nodes


class Parity(object):
    def __init__(self, p=100, U=1, seed=0, device=torch.device("cpu")):
        self.rng = np.random.default_rng(seed)
        self._num_nodes = p
        self.U = U
        self.device = device
        self.info = f"parity_p-{p}_U-{U}"

    def init_state(self):
        return torch.ones(self.num_nodes).to(self.device)

    def energy(self, x : Tensor):
        res = x.sum(dim=-1) % 2
        return res * self.U

    def change(self, x : Tensor):
        return torch.ones_like(x)

    def flip_state(self, x : Tensor, idx : Tuple):
        z = x.clone()
        for i in idx:
            z[i] = 1 - z[i]
        return z

    def trace(self, x : Tensor):
        return x

    @property
    def num_nodes(self):
        return self._num_nodes


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

    def change(self, x):
        with torch.no_grad():
            weight = x @ self.W.T + self.b_h
            weight = torch.exp(weight - torch.logaddexp(weight, torch.zeros_like(weight)))
            return (weight @ self.W + self.b_v) * (1 - 2 * x)

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



if __name__ == '__main__':
    model = BMM(p=1000, m=10)
    x = model.init_state()
    energy = model.energy(x)
    grad = model.grad(x)
    change = model.change(x)
    print('123')











