"""Full-matrix reconstruction objective with learnable sigmoid coefficients."""
import math
import torch
from torch import nn


class Objective(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(0.8 / 0.2)))
        self.beta_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.eps = eps

    def forward(self, x, reconstruction, mu, logvar):
        mse = (x - reconstruction).square().mean()
        a, b = x.flatten(1), reconstruction.flatten(1)
        a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
        corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(self.eps)
        pearson = 1 - corr.mean()
        kl = 0.5 * (mu.square() + logvar.exp() - logvar - 1).sum(-1).mean()
        alpha, beta = self.alpha_logit.sigmoid(), self.beta_logit.sigmoid()
        total = mse + alpha * pearson + beta * kl
        terms = dict(total=total, mse=mse, pearson=pearson, kl=kl, alpha=alpha, beta=beta, weighted_pearson=alpha * pearson, weighted_kl=beta * kl)
        return total, terms
