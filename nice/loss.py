"""
Implementation of NICE log-likelihood loss.
"""
import torch
import torch.nn as nn
import numpy as np

# ===== ===== Loss Function Implementations ===== =====
"""
We assume that we final output of the network are components of a multivariate distribution that
factorizes, i.e. the output is (y1,y2,...,yK) ~ p(Y) s.t. p(Y) = p_1(Y1) * p_2(Y2) * ... * p_K(YK),
with each individual component's prior distribution coming from a standardized family of
distributions, i.e. p_i == Gaussian(mu,sigma) for all i in 1..K, or p_i == Logistic(mu,scale).
"""

# wrap above loss functions in Modules:
class GaussianPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(GaussianPriorNICELoss, self).__init__()
        self.size_average = size_average
        self.factor = torch.log(torch.tensor(2 * np.pi)).cuda()
    def forward(self, fx, diag):
        if self.size_average:
            return torch.mean(-(torch.sum(diag) - (
                0.5 * torch.sum(torch.pow(fx, 2), dim=1) + fx.size(1) * 0.5 * self.factor)))
        else:
            return torch.sum(-(torch.sum(diag) - (
                0.5 * torch.sum(torch.pow(fx, 2), dim=1) + fx.size(1) * 0.5 * self.factor)))

