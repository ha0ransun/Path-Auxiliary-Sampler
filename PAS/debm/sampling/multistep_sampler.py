import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np


class MSFastSampler(nn.Module):
    def __init__(self, R):
        super(MSFastSampler, self).__init__()
        self.R = R

    def step(self, x, model):
        bsize = x.shape[0]
        x_rank = len(x.shape) - 1
        radius = torch.randint(1, self.R * 2, size=(bsize, 1))
        max_r = torch.max(radius).item()
        r_mask = torch.arange(max_r).expand(bsize, max_r) < radius
        r_mask = r_mask.float().to(x.device)

        x = x.requires_grad_()
        score_x = model(x)
        grad = torch.autograd.grad(score_x.sum(), x)[0].detach()
        with torch.no_grad():
            delta_x = -(2.0 * x - 1.0)
            score_change_x = delta_x * grad / 2.0
            prob_x_local = torch.softmax(score_change_x, dim=-1)

            indices = torch.multinomial(prob_x_local, max_r, replacement=False)
            cur_bits = x[torch.arange(bsize).unsqueeze(1).to(x.device), indices]
            new_bits = 1.0 - cur_bits
            y = x.clone()
            y[torch.arange(bsize).unsqueeze(1).to(x.device), indices] = r_mask * new_bits + (1 - r_mask) * cur_bits            

            delta_y = -(2.0 * y - 1.0)
            score_y = model(y)
            score_change_y = delta_y * grad / 2.0

            log_pypx = score_y - score_x
            log_tilde_pxpy = -torch.sum(grad * (y - x), dim=-1)
            log_zxzy = torch.logsumexp(score_change_x, dim=-1) - torch.logsumexp(score_change_y, dim=-1)
            log_acc = torch.clip(log_pypx + log_tilde_pxpy + log_zxzy, max=0)
            
            accepted = (log_acc.exp() > torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
        return new_x


class MSPathSampler(nn.Module):
    def __init__(self, R):
        super(MSPathSampler, self).__init__()
        self.R = R

    def step(self, x, model):
        bsize = x.shape[0]
        x_rank = len(x.shape) - 1
        radius = torch.randint(1, self.R * 2, size=(bsize, 1))
        max_r = torch.max(radius).item()
        r_mask = torch.arange(max_r).expand(bsize, max_r) < radius
        r_mask = r_mask.float().to(x.device)
        
        x = x.requires_grad_()
        score_x = model(x)
        grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()

        b_idx = torch.arange(bsize).to(x.device)
        with torch.no_grad():
            cur_x = x.clone()
            for step in range(max_r):
                delta_x = -(2.0 * cur_x - 1.0)
                score_change_x = delta_x * grad_x / 2.0
                if step == 0:
                    log_Zx = torch.logsumexp(score_change_x, dim=-1)
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                cur_bits = cur_x[b_idx, index]
                new_bits = 1.0 - cur_bits
                cur_r_mask = r_mask[:, step]
                cur_x[b_idx, index] = cur_r_mask * new_bits + (1.0 - cur_r_mask) * cur_bits
            y = cur_x
            delta_y = -(2.0 * y - 1.0)
            score_y = model(y)
            score_change_y = delta_y * grad_x / 2.0

            log_pypx = score_y - score_x
            log_tilde_pxpy = -torch.sum(grad_x * (y - x), dim=-1)
            log_zxzy = log_Zx - torch.logsumexp(score_change_y, dim=-1)
            log_acc = torch.clip(log_pypx + log_tilde_pxpy + log_zxzy, max=0)
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
            return new_x


class MSPathCorrectSampler(nn.Module):
    def __init__(self, R):
        print('our binary sampler')
        super(MSPathCorrectSampler, self).__init__()
        self._hops = R
        self.R_list = []
        self.R = R
        self.count = 0
        self.succ = 0

    def step(self, x, model):
        bsize = x.shape[0]
        x_rank = len(x.shape) - 1
        radius = torch.randint(1, self.R * 2, size=(bsize, 1))
        self.R_list.append(radius)
        max_r = torch.max(radius).item()
        r_mask = torch.arange(max_r).expand(bsize, max_r) < radius
        r_mask = r_mask.float().to(x.device)

        x = x.requires_grad_()
        score_x = model(x)
        grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()

        b_idx = torch.arange(bsize).to(x.device)
        delta_list = []
        with torch.no_grad():
            log_fwd = score_x.view(-1)
            cur_x = x.clone()
            idx_list = []
            for step in range(max_r):
                delta_x = -(2.0 * cur_x - 1.0)
                delta_list.append(delta_x)
                score_change_x = delta_x * grad_x / 2.0
                prob_x_local = torch.softmax(score_change_x, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                idx_list.append(index.view(-1, 1))
                cur_bits = cur_x[b_idx, index]
                new_bits = 1.0 - cur_bits
                cur_r_mask = r_mask[:, step]
                cur_x[b_idx, index] = cur_r_mask * new_bits + (1.0 - cur_r_mask) * cur_bits
            y = cur_x
        y = y.requires_grad_()
        score_y = model(y)
        grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()

        with torch.no_grad():
            r_idx = torch.arange(max_r).to(x.device).view(1, -1)
            b_idx = b_idx.view(-1, 1)
            idx_list = torch.cat(idx_list, dim=1)  # bsize x max_r

            # fwd from x -> y
            traj = torch.stack(delta_list, dim=1) # bsize x max_r x dim
            log_fwd = torch.log_softmax(traj * grad_x.unsqueeze(1) / 2.0, dim=-1)
            log_fwd = torch.sum(log_fwd[b_idx, r_idx, idx_list] * r_mask, dim=-1) + score_x.view(-1)

            # backwd from y -> x
            delta_y = -(2.0 * y - 1.0)
            delta_list.append(delta_y)
            traj = torch.stack(delta_list[1:], dim=1) # bsize x max_r x dim
            log_backwd = torch.log_softmax(traj * grad_y.unsqueeze(1) / 2.0, dim=-1)
            log_backwd = torch.sum(log_backwd[b_idx, r_idx, idx_list] * r_mask, dim=-1) + score_y.view(-1)

            log_acc = log_backwd - log_fwd
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
            self.count += bsize
            self.succ += accepted.sum()
            return new_x

    @property
    def avgR(self):
        return torch.stack(self.R_list, dim=-1).float().mean().item()


class MSPathCatSampler(nn.Module):
    def __init__(self, R):
        print('our categorical sampler')
        super(MSPathCatSampler, self).__init__()
        self.R = R

    def step(self, x, model):
        bsize = x.shape[0]
        x_rank = len(x.shape) - 1
        assert x_rank == 2
        radius = torch.randint(1, self.R * 2, size=(bsize, 1))
        max_r = torch.max(radius).item()
        r_mask = torch.arange(max_r).expand(bsize, max_r) < radius
        r_mask = r_mask.float().to(x.device)

        x = x.requires_grad_()
        score_x = model(x)
        grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()

        b_idx = torch.arange(bsize).to(x.device)
        traj_list = []
        idx_list = []
        with torch.no_grad():
            cur_x = x.clone()
            for step in range(max_r):
                score_change = grad_x - (grad_x * cur_x).sum(-1).unsqueeze(-1)
                traj_list.append(cur_x)
                prob_x_local = torch.softmax(score_change.view(bsize, -1) / 2.0, dim=-1)
                index = torch.multinomial(prob_x_local, 1).view(-1)
                idx_list.append(index.view(-1, 1))
                onehot_index = torch.nn.functional.one_hot(index, prob_x_local.shape[1]).float().view(cur_x.shape)
                data_dim_change = onehot_index.sum(-1).unsqueeze(-1)
                new_x = cur_x * (1.0 - data_dim_change) + onehot_index
                cur_r_mask = r_mask[:, step].unsqueeze(-1).unsqueeze(-1)
                cur_x = cur_r_mask * new_x + (1 - cur_r_mask) * cur_x
            y = cur_x
        y = y.requires_grad_()
        score_y = model(y)
        grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()

        with torch.no_grad():
            r_idx = torch.arange(max_r).to(x.device).view(1, -1)
            b_idx = b_idx.view(-1, 1)
            idx_list = torch.cat(idx_list, dim=1)  # bsize x max_r

            # fwd from x -> y
            traj = torch.stack(traj_list, dim=1) # bsize x max_r x dim x 256
            score_change = grad_x.unsqueeze(1) - (grad_x.unsqueeze(1) * traj).sum(-1).unsqueeze(-1)
            score_change = score_change.view(bsize, max_r, -1) / 2.0 # bsize x max_r x (dim * 256)
            log_local = torch.log_softmax(score_change, dim=-1)
            log_fwd = torch.sum(log_local[b_idx, r_idx, idx_list] * r_mask, dim=-1) + score_x.view(-1)

            # backwd from y -> x
            traj_list.append(y)
            traj = torch.stack(traj_list[1:], dim=1)
            score_change = grad_y.unsqueeze(1) - (grad_y.unsqueeze(1) * traj).sum(-1).unsqueeze(-1)
            score_change = score_change.view(bsize, max_r, -1) / 2.0  # bsize x max_r x (dim * 256)
            log_local = torch.log_softmax(score_change, dim=-1)
            log_backwd = torch.sum(log_local[b_idx, r_idx, idx_list], dim=-1) + score_y.view(-1)
            log_acc = log_backwd - log_fwd
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
            return new_x

