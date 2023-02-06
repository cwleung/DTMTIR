import torch
import numpy as np

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.priors import NormalPrior
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False, nu=2.5):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim).to(device)
        q_u = CholeskyVariationalDistribution(n_inducing)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        X_prior_mean = torch.zeros(n, latent_dim).to(device)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean).to(device))
        X_init = torch.nn.Parameter(torch.randn(n, latent_dim))
        # LatentVariable (c)
        self.X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        super().__init__(self.X, q_f)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        init_lengthscale = 0.1
        self.covar_module.base_kernel.lengthscale = init_lengthscale

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def get_state(self):
        X = self.X.q_mu.detach().numpy()
        std = self.X.q_log_sigma.detach().numpy()
        return X, std

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)
