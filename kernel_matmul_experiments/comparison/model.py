import contextlib
import math
import warnings
from typing import Any, Self

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import GridInterpolationKernel, Kernel, SpectralMixtureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    MeanFieldVariationalDistribution,
    NNVariationalStrategy,
)
from kernel_matmul.gpytorch import SpectralKernelMatmulKernel, SumKernel
from kernel_matmul.util import find_periodogram_peaks
from linear_operator.utils.warnings import NumericalWarning
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


class VNNGP(ApproximateGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: GaussianLikelihood,
        covar_module: Kernel,
        k: int = 64,
        training_batch_size: int = 4096,
    ):
        inducing_points = train_x.unsqueeze(-1).detach().clone()
        variational_strategy = FixedNNVariationalStrategy(
            self,
            inducing_points,
            MeanFieldVariationalDistribution(inducing_points.shape[-2]),
            k=k,
            training_batch_size=training_batch_size,
        )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = covar_module
        self.likelihood = likelihood
        self.train_y = train_y

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x: Tensor | None, prior: bool = False, **kwargs):
        return self.variational_strategy(x=x, prior=prior, **kwargs)


class FixedNNVariationalStrategy(NNVariationalStrategy):
    def _set_training_iterator(self):
        super()._set_training_iterator()
        self._training_indices_iterator = [x.cpu() for x in self._training_indices_iterator]


class MonitorCG(contextlib.AbstractContextManager):
    def __enter__(self):
        self._context = warnings.catch_warnings(record=True, category=NumericalWarning)
        self._catched = self._context.__enter__()
        return self

    def __exit__(self, *args):
        self._context.__exit__(*args)

    @property
    def converged(self):
        return len(self._catched) == 0


def train_gp(
    model: SimpleGP | VNNGP, max_epochs: int = 200, lr: float = 0.01, convergence_atol: float = 1e-2
) -> None:
    model.train()
    if isinstance(model, VNNGP):
        mll = VariationalELBO(model.likelihood, model, model.train_y.shape[-1])
    else:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    max_steps = max_epochs
    if isinstance(model, VNNGP):
        batch_size = model.variational_strategy.training_batch_size
        train_size = model.variational_strategy.M
        max_steps *= math.ceil(train_size / batch_size)

    last_loss = None
    for _ in range(max_steps):
        with contextlib.ExitStack() as stack:
            monitor = stack.enter_context(MonitorCG())
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, VNNGP):
                stack.enter_context(gpytorch.settings.cholesky_jitter(1e-4))
                output = model(x=None)
                batch = model.variational_strategy.current_training_indices
                loss = -mll(output, model.train_y[batch].to(output.mean.device)).sum()
            else:
                output = model(*model.train_inputs)
                loss = -mll(output, model.train_targets).sum()
            loss.backward()
            optimizer.step()
        if not monitor.converged:
            break
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


def init_ski_covar_module(
    peak_freqs: Tensor, peak_mags: Tensor, lengthscale: float, grid_size: int = 10000
) -> Kernel:
    kernel = GridInterpolationKernel(
        init_naive_covar_module(peak_freqs, peak_mags, lengthscale),
        grid_size=grid_size,
        num_dims=1,
    )
    return kernel


def init_kernel_matmul_covar_module(
    peak_freqs: Tensor, peak_mags: Tensor, lengthscale: float
) -> Kernel:
    kernel = SpectralKernelMatmulKernel(epsilon=1e-5, batch_shape=peak_freqs.shape)
    kernel.lengthscale = torch.tensor(lengthscale)
    kernel.raw_lengthscale.requires_grad = False
    kernel.frequency = peak_freqs
    kernel.outputscale = peak_mags
    return SumKernel(kernel)


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
    method: str,
    **kwargs: Any,
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
    if method == "naive":
        kernel = init_naive_covar_module(peak_freqs, peak_mags, lengthscale)
        return SimpleGP(train_x, train_y, likelihood, kernel, **kwargs)
    elif method == "kernel-matmul":
        kernel = init_kernel_matmul_covar_module(peak_freqs, peak_mags, lengthscale)
        return SimpleGP(train_x, train_y, likelihood, kernel, **kwargs)
    elif method == "ski":
        kernel = init_ski_covar_module(peak_freqs, peak_mags, lengthscale)
        return SimpleGP(train_x, train_y, likelihood, kernel, **kwargs)
    elif method == "ski-exact":
        kernel = init_ski_covar_module(peak_freqs, peak_mags, lengthscale, grid_size=len(train_x))
        return SimpleGP(train_x, train_y, likelihood, kernel, **kwargs)
    elif method == "vnngp":
        kernel = init_naive_covar_module(peak_freqs, peak_mags, lengthscale)
        return VNNGP(train_x, train_y, likelihood, kernel, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
