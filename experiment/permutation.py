import os, pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from PAS.model import Permutation
from PAS.sampler import LBSampler, MSASampler, GibbsSampler, RWSampler
from PAS.args import *
from statsmodels.tsa.stattools import acf

class MainExp(object):
    def __init__(self):
        self.model = None
        self.color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:grey', 'tab:olive', 'tab:cyan']

    def init_tsp(self, p=100, sigma=5, seed=0, device=torch.device("cpu")):
        self.model = Permutation(p=p, sigma=sigma, seed=seed, device=device)
        self.model_info = f"parity/p-{p}_sigma-{sigma}"

    def _run(self, method='LB_R-1', T=1000):
        config = method.split(sep='-')
        if config[0] == 'LB':
            sampler = LBSampler(R=1)
            return sampler.sample(self.model, T=T, method=method)
        elif config[0] == 'PAS':
            sampler = MSASampler(R=3)
            return sampler.sample(self.model, T=T, method=method)
        elif config[0] == 'RW':
            sampler = RWSampler(R=int(config[1]))
            return sampler.sample(self.model, T=T, method=method)
        elif config[0] == 'Gibbs':
            sampler = GibbsSampler(R=int(config[1].split(sep='-')[1]))
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

    def eva(self, methods, T=1000, device=torch.device("cpu"), N=5):
        Sigma = [3, 5, 10]
        fig, ax = plt.subplots(1, len(Sigma), figsize=(len(Sigma) * 8, 4))
        for i, sigma in enumerate(Sigma):
            self.init_tsp(p=100, sigma=sigma, device=device)
            for j, method in enumerate(methods[:-1]):
                LogP = []
                for n in range(N):
                    logp, trace, elapse, succ = self._run(method, T=T)
                    LogP.append(logp)
                LogP = np.array(LogP).squeeze()
                ax[i].plot(np.arange(T) * 2, np.mean(LogP, axis=0), label=method, color=self.color[j])
                ax[i].fill_between(np.arange(T) * 2, np.mean(LogP, axis=0) - np.std(LogP, axis=0),
                                   np.mean(LogP, axis=0) + np.std(LogP, axis=0),
                                   color=self.color[j], alpha=0.3)
            LogP = []
            for n in range(N):
                logp, trace, elapse, succ = self._run('PAS-3', T=1666)
                LogP.append(logp)
            LogP = np.array(LogP).squeeze()
            ax[i].plot(np.arange(1666) * 6, np.mean(LogP, axis=0), label='PAS-3', color=self.color[3])
            ax[i].fill_between(np.arange(1666) * 6, np.mean(LogP, axis=0) - np.std(LogP, axis=0),
                               np.mean(LogP, axis=0) + np.std(LogP, axis=0),
                               color=self.color[3], alpha=0.3)
            ax[i].legend(fontsize=18)
            ax[i].tick_params(labelsize=12)
            ax[i].grid()
            ax[i].set_title(f"p = 100, Sigma = {sigma}", fontsize=18)
            ax[i].set_xlabel("Energy Function Evaluations", fontsize=14)
        plt.savefig("figs/permutation.pdf", bbox_inches="tight")
        # plt.show()

if __name__ == '__main__':
    device = torch.device(f"cuda:{cmd_args.device}" if torch.cuda.is_available() else "cpu")
    myExp = MainExp()
    methods = ['RW-1', 'RW-3', 'LB-1', 'PAS-3']
    myExp.eva(methods, T=5000,  device=device, N=5)

