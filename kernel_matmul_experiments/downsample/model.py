from typing import Self

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, SpectralMixtureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from kernel_matmul.util import find_periodogram_peaks
from torch import Tensor


class SimpleGP(ExactGP):
    def __init__(
        self, train_x: Tensor, train_y: Tensor, likelihood: GaussianLikelihood, covar_module: Kernel
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = covar_module

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(
    model: SimpleGP, max_epochs: int = 200, lr: float = 0.01, convergence_atol: float = 1e-2
) -> None:
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = None
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets).sum()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if last_loss is not None and abs(last_loss - loss) < convergence_atol:
            break
        last_loss = loss


def init_naive_covar_module(peak_freqs: Tensor, peak_mags: Tensor, lengthscale: float) -> Kernel:
    kernel = SpectralMixtureKernel(num_mixtures=len(peak_freqs))
    kernel.mixture_scales = torch.full((len(peak_freqs),), 1 / (2 * torch.pi * lengthscale))[
        ..., None, None
    ]
    kernel.raw_mixture_scales.requires_grad = False
    kernel.mixture_means = peak_freqs[..., None, None]
    kernel.mixture_weights = peak_mags
    return kernel


class LogNormalPrior(gpytorch.priors.LogNormalPrior):
    def __init__(self, loc_, scale_, validate_args=None, transform=None):
        loc = torch.log(loc_**2 / torch.sqrt(scale_**2 + loc_**2))
        scale = torch.sqrt(torch.log(1 + scale_**2 / loc_**2)).clamp_min(1e-6)
        super().__init__(loc, scale, validate_args, transform)

    def to(self, *args, **kwargs) -> Self:
        self.base_dist.loc = self.loc.to(*args, **kwargs)
        self.base_dist.scale = self.scale.to(*args, **kwargs)


def make_gp(
    train_x: Tensor,
    train_y: Tensor,
    lengthscale: float,
    num_components: int,
    max_frequency: float,
    peak_distance: int,
    peak_oversample: int,
    noise: float,
) -> SimpleGP:
    likelihood = GaussianLikelihood()
    likelihood.noise = noise
    likelihood.raw_noise.requires_grad = False
    peak_freqs, peak_mags = find_periodogram_peaks(
        train_x,
        train_y,
        num_components,
        max_frequency,
        peak_distance,
        peak_oversample,
    )
    kernel = init_naive_covar_module(peak_freqs, peak_mags, lengthscale)
    model = SimpleGP(train_x, train_y, likelihood, kernel)
    return model
