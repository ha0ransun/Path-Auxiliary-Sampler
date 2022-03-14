import os, pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from PAS.model import Parity
from PAS.sampler import LBSampler, MSASampler
from PAS.args import *
from statsmodels.tsa.stattools import acf

class MainExp(object):
    def __init__(self):
        self.model = None
        self.color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:grey', 'tab:olive', 'tab:cyan']

    def init_parity(self, p=100, U=1, seed=0, device=torch.device("cpu")):
        self.model = Parity(p=p, U=U, seed=seed, device=device)
        self.model_info = f"parity/p-{p}_U-{U}"

    def _run(self, method='LB_R-1', T=1000):
        config = method.split(sep='_')
        if config[0] == 'LB':
            sampler = LBSampler(R=1)
            return sampler.sample(self.model, T=T, method=method)
        elif config[0] == 'PAS':
            sampler = MSASampler(R=2)
            return sampler.sample(self.model, T=T, method=method)
        else:
            raise NotImplementedError

    def ess(self, auto_cor):
        rho = 0
        for i in range(len(auto_cor)):
            if auto_cor[i] < 0:
                break
            rho += auto_cor[i]
        return len(auto_cor) / (1 + 2 * rho)

    def trace_process(self, trace):
        trace = np.stack([t[0].cpu().numpy() for t in trace], axis=0)
        for i in range(1, len(trace)):
            trace[i] = trace[i-1] * i / (i + 1) + trace[i] / (i + 1)
        trace = np.linalg.norm(trace - 0.5, axis=-1)
        return trace

    def eva(self, device=torch.device("cpu"), N=5):
        fig, ax = plt.subplots(1, 3, figsize=(24, 4))
        U = [1, 3, 5]
        for i, u in enumerate(U):
            self.init_parity(p=100, U=u, device=device)
            # run lb
            Trace = []
            for n in range(N):
                logp, trace, elapse, succ = self._run('LB', T=20000)
                trace = self.trace_process(trace)
                Trace.append(trace)
            Trace = np.stack(Trace, axis=0)
            ax[i].plot(np.arange(20000) * 2, np.mean(Trace, axis=0), label='LB', color=self.color[0])
            ax[i].fill_between(np.arange(20000) * 2, np.mean(Trace, axis=0) - np.std(Trace, axis=0),
                               np.mean(Trace, axis=0) + np.std(Trace, axis=0),
                               color=self.color[0], alpha=0.3)
            # run pas
            Trace = []
            for n in range(N):
                logp, trace, elapse, succ = self._run('PAS', T=10000)
                trace = self.trace_process(trace)
                Trace.append(trace)
            Trace = np.stack(Trace, axis=0)
            ax[i].plot(np.arange(10000)*4, np.mean(Trace, axis=0), label='PAS-2', color=self.color[1])
            ax[i].fill_between(np.arange(10000)*4, np.mean(Trace, axis=0) - np.std(Trace, axis=0),
                               np.mean(Trace, axis=0) + np.std(Trace, axis=0),
                               color=self.color[1], alpha=0.3)
            ax[i].legend(fontsize=18)
            ax[i].tick_params(labelsize=12)
            ax[i].grid()
            ax[i].set_title(f"p = 100, U = {u}", fontsize=18)
            ax[i].set_xlabel("Energy Function Evaluations", fontsize=14)
        plt.savefig("figs/parity.pdf", bbox_inches="tight")
        # plt.show()

if __name__ == '__main__':
    device = torch.device(f"cuda:{cmd_args.device}" if torch.cuda.is_available() else "cpu")
    myExp = MainExp()
    names = ['LB', 'PAS-2']
    myExp.eva(device=device, N=5)

