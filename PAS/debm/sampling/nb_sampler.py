import torch
import torch.nn as nn

class NBSampler(nn.Module):
    def __init__(self, R):
        print('our binary sampler')
        super().__init__()
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
        with torch.no_grad():
            cur_x = x.clone()
            idx_list = []
            delta_x = -(2.0 * cur_x - 1.0)
            score_change_x = delta_x * grad_x / 2.0
            prob_x = torch.softmax(score_change_x, dim=-1)
            for step in range(max_r):
                index = torch.multinomial(prob_x, 1).view(-1)
                cur_bits = cur_x[b_idx, index]
                new_bits = 1.0 - cur_bits
                cur_r_mask = r_mask[:, step]
                cur_x[b_idx, index] = cur_r_mask * new_bits + (1.0 - cur_r_mask) * cur_bits
                prob_x[b_idx, index] = 0
                idx_list.append(index)
            y = cur_x
        y = y.requires_grad_()
        score_y = model(y)
        grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()

        with torch.no_grad():
            r_idx = torch.arange(max_r).to(x.device).view(1, -1)
            idx_all = torch.stack(idx_list, dim=1)  # bsize x max_r

            # fwd from x -> y
            change_fwd = score_change_x.unsqueeze(1).repeat(1, max_r, 1)
            for i, idx in enumerate(idx_list):
                for j in range(i + 1, max_r):
                    change_fwd[b_idx, torch.LongTensor([j] * bsize).to(x.device), idx] = -float('inf')
            log_fwd = torch.log_softmax(change_fwd, dim=-1)
            log_fwd = torch.sum(log_fwd[b_idx.view(-1, 1), r_idx, idx_all] * r_mask, dim=-1) + score_x.view(-1)

            # backwd from y -> x
            delta_y = -(2.0 * y - 1.0)
            score_change_y = delta_y * grad_y / 2.0
            change_bwd = score_change_y.unsqueeze(1).repeat(1, max_r, 1)
            for i, idx in enumerate(idx_list):
                for j in range(i):
                    change_bwd[b_idx, torch.LongTensor([j] * bsize).to(x.device), idx] = -float('inf')
            log_bwd = torch.log_softmax(change_bwd, dim=-1)
            log_bwd = torch.sum(log_bwd[b_idx.view(-1,1), r_idx, idx_all] * r_mask, dim=-1) + score_y.view(-1)

            log_acc = log_bwd - log_fwd
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
            self.count += bsize
            self.succ += accepted.sum()
            return new_x

    @property
    def avgR(self):
        return torch.stack(self.R_list, dim=-1).float().mean().item()


class NBASampler(nn.Module):
    def __init__(self):
        print('our binary sampler')
        super().__init__()
        self.R_list = []
        self.count = 0
        self.succ = 0

    def step(self, x, model):
        bsize = x.shape[0]
        x_rank = len(x.shape) - 1

        x = x.requires_grad_()
        score_x = model(x)
        grad_x = torch.autograd.grad(score_x.sum(), x)[0].detach()

        b_idx = torch.arange(bsize).to(x.device)
        with torch.no_grad():
            cur_x = x.clone()
            idx_list = []
            delta_x = -(2.0 * cur_x - 1.0)
            score_change_x = delta_x * grad_x / 2.0
            prob_x = torch.softmax(score_change_x, dim=-1)
            radius = (prob_x > 0.02).sum(dim=1, keepdim=True) + 1
            self.R_list.append(radius)
            max_r = torch.max(radius).item()
            r_mask = torch.arange(max_r).to(x.device).expand(bsize, max_r) < radius
            r_mask = r_mask.float().to(x.device)
            for step in range(max_r):
                index = torch.multinomial(prob_x, 1).view(-1)
                cur_bits = cur_x[b_idx, index]
                new_bits = 1.0 - cur_bits
                cur_r_mask = r_mask[:, step]
                cur_x[b_idx, index] = cur_r_mask * new_bits + (1.0 - cur_r_mask) * cur_bits
                prob_x[b_idx, index] = 0
                idx_list.append(index)
            y = cur_x
        y = y.requires_grad_()
        score_y = model(y)
        grad_y = torch.autograd.grad(score_y.sum(), y)[0].detach()

        with torch.no_grad():
            r_idx = torch.arange(max_r).to(x.device).view(1, -1)
            idx_all = torch.stack(idx_list, dim=1)  # bsize x max_r

            # fwd from x -> y
            change_fwd = score_change_x.unsqueeze(1).repeat(1, max_r, 1)
            for i, idx in enumerate(idx_list):
                for j in range(i + 1, max_r):
                    change_fwd[b_idx, torch.LongTensor([j] * bsize).to(x.device), idx] = -float('inf')
            log_fwd = torch.log_softmax(change_fwd, dim=-1)
            log_fwd = torch.sum(log_fwd[b_idx.view(-1, 1), r_idx, idx_all] * r_mask, dim=-1) + score_x.view(-1)

            # backwd from y -> x
            delta_y = -(2.0 * y - 1.0)
            score_change_y = delta_y * grad_y / 2.0
            change_bwd = score_change_y.unsqueeze(1).repeat(1, max_r, 1)
            for i, idx in enumerate(idx_list):
                for j in range(i):
                    change_bwd[b_idx, torch.LongTensor([j] * bsize).to(x.device), idx] = -float('inf')
            log_bwd = torch.log_softmax(change_bwd, dim=-1)
            log_bwd = torch.sum(log_bwd[b_idx.view(-1,1), r_idx, idx_all] * r_mask, dim=-1) + score_y.view(-1)

            log_acc = log_bwd - log_fwd
            accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
            new_x = y * accepted + (1.0 - accepted) * x
            self.count += bsize
            self.succ += accepted.sum()
            return new_x

    @property
    def avgR(self):
        return torch.stack(self.R_list, dim=-1).float().mean().item()
