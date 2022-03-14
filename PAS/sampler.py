import torch
import numpy as np
import time
from SIP.args import cmd_args
from itertools import combinations
from tqdm import tqdm


### Single Chain version of samplers. Parallel version can be found at PAS/sampling/ ###

class BaseSampler(object):
    def __init__(self, seed=cmd_args.seed):
        self.rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self):
        self.logp = []
        self.trace = []
        self.elapse = 0
        self.succ = 0

    def _step(self, model, t, x, *args, **kwargs):
        raise NotImplementedError

    def sample(self, model, T=1000, method='Gibbs', *args, **kwargs):
        self._reset()
        self.x = model.init_state()
        self.energy = model.energy(self.x)

        progress_bar = tqdm(range(T))
        progress_bar.set_description(f"[{method}, {model.info}]")
        t_begin = time.time()
        for t in progress_bar:
            self._step(model, t, *args, **kwargs)
            self.logp.append([self.energy.item()])
            self.trace.append([model.trace(self.x)])
        t_end = time.time()
        self.elapse += t_end - t_begin

        return self.logp, self.trace, self.elapse, self.succ


class RWSampler(BaseSampler):
    def __init__(self, R=1, seed=cmd_args.seed):
        super().__init__(seed=seed)
        self.R = R

    def _step(self, model, t, R=1, *args, **kwargs):
        r = self.rng.integers(1, self.R+1)
        idx = self.rng.choice(model.num_nodes, r, replace=False)
        new_state = model.flip_state(self.x, idx)
        new_energy = model.energy(new_state)
        if self.rng.random() < torch.exp(- new_energy + self.energy):
            self.x = new_state
            self.energy = new_energy
            self.succ += 1


class GibbsSampler(BaseSampler):
    def __init__(self, R=1, seed=cmd_args.seed):
        super().__init__(seed=seed)
        self.R = R

    def _step(self, model, t, *args, **kwargs):
        coordinates = [(t * self.R + i) % model.num_nodes for i in range(self.R)]
        new_state = []
        for r in range(self.R + 1):
            for idx in combinations(coordinates, r):
                new_state.append(model.flip_state(self.x, idx))

        new_state = torch.stack(new_state, dim=0)
        new_energy = model.energy(new_state)
        prob = torch.exp(- new_energy + self.energy)
        try:
            idx = torch.multinomial(prob, 1, replacement=True).item()
        except:
            idx = 0

        if idx != 0:
            self.succ += 1
            self.x = new_state[idx]
            self.energy = new_energy[idx]

class LBSampler(BaseSampler):
    def __init__(self, R=1, seed=cmd_args.seed):
        super().__init__(seed)
        assert R == 1
        self.R = R

    def _step(self, model, t, *args, **kwargs):
        change_x = model.change(self.x)
        prob_x = torch.softmax(- change_x / 2, dim=0)
        idx = torch.multinomial(prob_x, self.R, replacement=True)
        y = model.flip_state(self.x, idx)
        energy_y = model.energy(y)

        change_y = model.change(y)
        prob_y = torch.softmax(- change_y / 2, dim=0)

        if self.rng.random() < torch.exp(- energy_y + self.energy) * prob_y[idx] / prob_x[idx]:
            self.x = y
            self.energy = energy_y
            self.prob = prob_y
            self.succ += 1


class MTSampler(BaseSampler):
    def __init__(self, K=20, R=1, seed=cmd_args.seed):
        super().__init__(seed=seed)
        self.R = R
        self.K = K

    def _step(self, model, t, *args, **kwargs):
        new_state = []
        R = self.rng.integers(1, 2 * self.R, self.K)
        for k in range(self.K):
            idx = self.rng.choice(model.num_nodes, R[k], replace=False)
            new_state.append(model.flip_state(self.x, idx))
        new_state = torch.stack(new_state, dim=0)
        new_energy = model.energy(new_state)
        prob = torch.exp((- new_energy + self.energy) / 2)
        Zx = prob.sum()
        idx = int(torch.multinomial(prob, 1, replacement=True))
        y = new_state[idx]
        energy_y = new_energy[idx]

        new_state = [self.x]
        for k in range(self.K - 1):
            r = R[k]
            idx = self.rng.choice(model.num_nodes, r, replace=False)
            new_state.append(model.flip_state(y, idx))
        new_state = torch.stack(new_state, dim=0)
        new_energy = model.energy(new_state)
        Zy = torch.exp((- new_energy + energy_y) / 2).sum()

        if self.rng.random() < Zx / Zy:
            self.x = y
            self.energy = energy_y
            self.succ += 1


class HBSampler(BaseSampler):
    def __init__(self, block_size=10, hamming_dist=1, seed=cmd_args.seed):
        super().__init__(seed=seed)
        self.block_size = block_size
        self.hamming_dis = hamming_dist
        self._inds = []

    def _step(self, model, t, *args, **kwargs):
        if len(self._inds) == 0:
            self._inds = self.rng.permutation(model.num_nodes)

        inds = self._inds[:self.block_size]
        self._inds = self._inds[self.block_size:]


    @staticmethod
    def hamming_ball(n, k):
        ball = [np.zeros((n,))]
        for i in range(k + 1)[1:]:
            it = combinations(range(n), i)
            for tup in it:
                vec = np.zeros((n,))
                for ind in tup:
                    vec[ind] = 1.
                ball.append(vec)
        return ball




class GWGSampler(BaseSampler):
    def __init__(self, R=1, seed=cmd_args.seed):
        super().__init__(seed=seed)
        self.R = R

    def _step(self, model, t, *args, **kwargs):
        R = int(self.rng.integers(1, 2 * self.R, 1))
        grad = model.grad(self.x)
        delta_x = model.delta(self.x)
        energy_change = delta_x * grad
        prob_x = torch.softmax(- energy_change / 2, dim=-1)
        log_x = torch.log_softmax(- energy_change / 2, dim=-1)
        if prob_x.sum() != 1:
            prob_x /= prob_x.sum()
        idx = torch.multinomial(prob_x, R, replacement=True)
        y = model.flip_state(self.x, idx)

        energy_y = model.energy(y)
        grad_y = model.grad(y)
        delta_y = model.delta(y)
        new_energy_change = delta_y * grad_y
        log_y = torch.log_softmax(- new_energy_change / 2, dim=-1)

        Qx = 0
        Qy = 0
        for id in idx:
            Qx += log_x[id]
            Qy += log_y[id]

        if self.rng.random() < torch.exp(- energy_y + self.energy + Qy - Qx):
            self.x = y
            self.energy = energy_y
            self.succ += 1


class MSFSampler(BaseSampler):
    def __init__(self, R=1, seed=cmd_args.seed):
        super().__init__(seed)
        self.R = R
        self.grad = None

    def _step(self, model, t, *args, **kwargs):
        R = int(self.rng.integers(1, 2 * self.R, 1))
        log_ratio = 0
        Delta = []
        Idx = []

        if self.grad is None:
            grad_x = model.grad(self.x)
        else:
            grad_x = self.grad
        delta = model.delta(self.x)
        Delta.append(delta)
        energy_change = delta * grad_x
        prob = torch.softmax(- energy_change / 2, dim=-1)
        idx = torch.multinomial(prob, 1, replacement=True)
        Idx.append(idx)
        x = model.flip_state(self.x, idx)

        # intermediate steps
        for _ in range(1, R):
            delta = model.delta(x)
            Delta.append(delta)
            energy_change = delta * grad_x
            prob = torch.softmax(- energy_change / 2, dim=-1)
            idx = torch.multinomial(prob, 1, replacement=True)
            Idx.append(idx)
            x = model.flip_state(x, idx)

        # last step
        delta = model.delta(x)
        Delta.append(delta)
        grad_y = model.grad(x)
        energy_y = model.energy(x)
        delta_xy =  torch.stack(Delta[:-1], dim=0) * grad_x
        delta_yx = torch.stack(Delta[1:], dim=0) * grad_y
        log_xy = torch.log_softmax(- delta_xy / 2, dim=-1)
        log_yx = torch.log_softmax(- delta_yx / 2, dim=-1)
        # log_xy = torch.log_softmax(- delta_xy * temp[:, None], dim=-1)
        # log_yx = torch.log_softmax(- delta_yx * temp.flip(0)[:, None], dim=-1)
        for i, idx in enumerate(Idx):
            log_ratio += log_yx[i][idx] - log_xy[i][idx]

        if self.rng.random() < torch.exp(- energy_y + self.energy + log_ratio):
            self.x = x
            self.energy = energy_y
            self.succ += 1


class MSASampler(BaseSampler):
    """
    Multi Step Accurate Sampler
    """
    def __init__(self, R=1, seed=cmd_args.seed):
        super().__init__(seed=seed)
        self.R = R
        self.energy_change = None
        self.Z = None

    def _step(self, model, t, *args, **kwargs):
        R = int(self.rng.integers(1, 2 * self.R, 1))
        x = self.x
        indices = []
        for t in range(R):
            if t == 0:
                if self.Z is None:
                    energy_change = model.change(x)
                    Zx = torch.logsumexp(- energy_change / 2, dim=-1)
                else:
                    energy_change = self.energy_change
                    Zx = self.Z
            else:
                energy_change = model.change(x)
            prob = torch.exp(-energy_change / 2)
            idx = torch.multinomial(prob, 1, replacement=True)
            indices.append(idx)
            x = model.flip_state(x, idx)

        energy_change_y = model.change(x)
        Zy = torch.logsumexp(- energy_change_y / 2, dim=-1)


        if self.rng.random() < torch.exp(Zx - Zy):
            self.x = x
            self.energy = model.energy(self.x)
            self.succ += 1
            self.energy_change = energy_change_y
            self.Z = Zy

class NBSampler(BaseSampler):
    """
    Multi Step Accurate Sampler
    """
    def __init__(self, R=1, seed=cmd_args.seed):
        super().__init__(seed=seed)
        self.R = R

    def _step(self, model, t, *args, **kwargs):
        R = int(self.rng.integers(1, 2 * self.R, 1))
        log_ratio = 0
        Change = []
        Idx = []

        # First Step
        energy_change = model.change(self.x)
        prob = torch.exp(-energy_change / 2)
        idx = torch.multinomial(prob, 1, replacement=True)
        Idx.append(idx)
        Change.append(energy_change)
        x = model.flip_state(self.x, idx)

        # Intermediate Steps
        for t in range(1, R):
            energy_change = model.change(x)
            prob = torch.exp(-energy_change / 2)
            for i in Idx:
                prob[i] = 0
            idx = torch.multinomial(prob, 1, replacement=True)
            Idx.append(idx)
            Change.append(energy_change)
            x = model.flip_state(x, idx)

        # Last Step
        energy_change = model.change(x)
        energy_y = model.energy(x)
        Change.append(energy_change)
        Change_x2y = torch.stack(Change[:-1], dim=0)
        Change_y2x = torch.stack(Change[1:], dim=0)
        for i, idx in enumerate(Idx):
            Change_x2y[i+1:, idx] = float('inf')
            Change_y2x[:i, idx] = float('inf')
        log_x2y = torch.log_softmax(-Change_x2y / 2, dim=-1)
        log_y2x = torch.log_softmax(-Change_y2x / 2, dim=-1)
        for i, idx in enumerate(Idx):
            log_ratio += log_y2x[i][idx] - log_x2y[i][idx]

        if self.rng.random() < torch.exp(- energy_y + self.energy + log_ratio):
            self.x = x
            self.energy = model.energy(self.x)
            self.succ += 1