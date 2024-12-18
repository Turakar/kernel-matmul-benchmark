from typing import Any

import torch
from gpytorch.kernels.keops.keops_kernel import KeOpsKernel
from kernel_matmul.gpytorch.spectral import SpectralKernelMatmulKernel
from linear_operator.operators import KernelLinearOperator
from pykeops.torch import LazyTensor as KEOLazyTensor
from torch import Tensor


def _covar_func(x1, x2, lengthscale, frequency, outputscale):
    # symbolic array of shape ..., ndatax1_ x 1 x 1
    x1_ = KEOLazyTensor(x1[..., :, None, :])
    # symbolic array of shape ..., 1 x ndatax2_ x 1
    x2_ = KEOLazyTensor(x2[..., None, :, :])
    lengthscale = lengthscale[..., None, None, None]  # 1 x 1 x 1
    frequency = frequency[..., None, None, None]  # 1 x 1 x 1
    outputscale = outputscale[..., None, None, None]  # 1 x 1 x 1
    tau = x1_ - x2_
    rbf = (-0.5 * (tau**2).sum(-1) / lengthscale).exp()
    cos = (2 * torch.pi * frequency * tau).cos()
    return outputscale * rbf * cos


class KeOpsSpectralKernel(KeOpsKernel, SpectralKernelMatmulKernel):
    def _nonkeops_forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs: Any):
        if diag:
            tau = (x1 - x2).squeeze(-1)
            rbf = torch.exp(-0.5 * (x1 - x2) ** 2 / self.lengthscale[..., None])
            cos = torch.cos(2 * torch.pi * self.frequency[..., None] * tau)
            return self.outputscale[..., None] * rbf * cos
        raise NotImplementedError("Non-diagonal non-keops forward not implemented")

    def _keops_forward(self, x1: Tensor, x2: Tensor, **kwargs: Any):
        return KernelLinearOperator(
            x1,
            x2,
            num_nonbatch_dimensions={"lengthscale": 0, "frequency": 0, "outputscale": 0},
            covar_func=_covar_func,
            lengthscale=self.lengthscale,
            frequency=self.frequency,
            outputscale=self.outputscale,
        )

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs: Any):
        if diag:
            return self._nonkeops_forward(x1, x2, diag=True, **kwargs)
        else:
            return self._keops_forward(x1, x2, **kwargs)
