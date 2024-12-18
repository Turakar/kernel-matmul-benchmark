import abc
import multiprocessing.pool

import torch
from gpytorch.kernels.kernel import sq_dist
from kernel_matmul.configurations import MatmulAutotuneConfiguration
from kernel_matmul.native_function import NativeFunction
from kernel_matmul.ranges import make_ranges
from pykeops.torch import LazyTensor
from pykeops.torch.cluster import from_matrix
from torch import Tensor


class MatmulBase(abc.ABC):
    @abc.abstractmethod
    def prepare_train(self, x: Tensor, kernel_type: str) -> None:
        ...

    @abc.abstractmethod
    def prepare_epoch(self, params: Tensor) -> None:
        ...

    @abc.abstractmethod
    def __call__(self, rhs: Tensor) -> Tensor:
        ...


class KernelMatmul(MatmulBase):
    def __init__(
        self, cutoff: float | None = None, compile_pool: multiprocessing.pool.Pool | None = None
    ):
        super().__init__()
        self.cutoff = cutoff
        self.x = None
        self.params = None
        self.kernel_type = None
        self.start = None
        self.end = None
        self.compile_pool = compile_pool

    def prepare_train(self, x: Tensor, kernel_type: str) -> None:
        start, end = make_ranges(self.cutoff, x)
        self.x = x
        self.kernel_type = kernel_type
        self.start = start
        self.end = end
        self.kernel_matmul = NativeFunction(
            "matmul", MatmulAutotuneConfiguration(kernel_type), compile_pool=self.compile_pool
        )

    def prepare_epoch(self, params: Tensor) -> None:
        self.params = params

    def __call__(self, rhs: Tensor) -> Tensor:
        x = self.x.unsqueeze(0)
        params = self.params
        start = self.start.unsqueeze(0)
        end = self.end.unsqueeze(0)
        shape = torch.broadcast_shapes(
            x.shape[:-1], rhs.shape[:-2], params.shape[:-1], start.shape[:-1], end.shape[:-1]
        )
        x = x.expand(*shape, *x.shape[-1:])
        rhs = rhs.expand(*shape, *rhs.shape[-2:])
        params = params.expand(*shape, *params.shape[-1:])
        start = start.expand(*shape, *start.shape[-1:])
        end = end.expand(*shape, *end.shape[-1:])
        return self.kernel_matmul(x, x, rhs, params, start, end)


class NaiveMatmul(MatmulBase):
    def __init__(self):
        super().__init__()
        self.x = None
        self.kernel = None

    def prepare_train(self, x: Tensor, kernel_type: str) -> None:
        assert kernel_type == "rbf"
        self.x = x

    def prepare_epoch(self, params: Tensor) -> None:
        tau_sq = sq_dist(self.x.unsqueeze(-1), self.x.unsqueeze(-1), x1_eq_x2=True)[None, :, :]
        lengthscale = params[:, 0, None, None]
        outputscale = params[:, 1, None, None]
        self.kernel = outputscale * torch.exp(-0.5 * tau_sq / lengthscale**2)

    def __call__(self, rhs: Tensor) -> Tensor:
        return self.kernel @ rhs.unsqueeze(0)


class SparseBsrMatmul(MatmulBase):
    def __init__(self, block_size: int, cutoff: float) -> None:
        super().__init__()
        self.block_size = block_size
        self.cutoff = cutoff
        self.kernels = []

    def prepare_train(self, x: Tensor, kernel_type: str) -> None:
        assert kernel_type == "rbf"
        assert len(x) % self.block_size == 0
        self.size = len(x)
        # crow, cols, values = _sparse_bsr_prepare(x, self.block_size, self.cutoff)

        start, end = make_ranges(self.cutoff, x, block_size=self.block_size, align=True)
        crow = [0]
        cols = []
        values = []
        for i in range(len(start)):
            dist = sq_dist(
                x[i * self.block_size : (i + 1) * self.block_size, None], x[start[i] : end[i], None]
            )
            for j in range(start[i].item() // self.block_size, end[i].item() // self.block_size):
                cols.append(j)
                values.append(
                    dist[
                        :, (j * self.block_size) - start[i] : ((j + 1) * self.block_size) - start[i]
                    ]
                )
            crow.append(len(cols))
        crow = torch.tensor(crow, dtype=torch.int32, device=x.device)
        cols = torch.tensor(cols, dtype=torch.int32, device=x.device)
        values = torch.stack(values, dim=0)

        self.crow = crow
        self.cols = cols
        self.values = values

    def prepare_epoch(self, params: Tensor) -> None:
        lengthscale = params[:, 0, None, None, None]
        outputscale = params[:, 1, None, None, None]
        kernel_values = outputscale * torch.exp(-0.5 * self.values[None, :, :, :] / lengthscale**2)
        self.kernels = [
            torch.sparse_bsr_tensor(
                self.crow,
                self.cols,
                kernel_values[i],
                size=torch.Size([self.size, self.size]),
            )
            for i in range(params.shape[0])
        ]

    def __call__(self, rhs: Tensor) -> Tensor:
        results = [(kernel @ rhs[i]).to_dense() for i, kernel in enumerate(self.kernels)]
        return torch.stack(results, dim=0)


class KeOpsMatmul(MatmulBase):
    def __init__(self, cutoff: float | None = None, block_size: int | None = None):
        super().__init__()
        if cutoff is None:
            block_size = None
        self.cutoff = cutoff
        self.block_size = block_size
        self.x = None
        self.outputscale = None
        self.ranges = None
        self.batch_ranges = None

    def prepare_train(self, x: Tensor, kernel_type: str) -> None:
        assert kernel_type == "rbf"
        self.x = x

        if self.cutoff is not None:
            assert (
                len(x) % self.block_size == 0
            ), f"len(x)={len(x)} is not divisible by block_size={self.block_size}"
            num_blocks = len(x) // self.block_size
            block_index = torch.stack(
                [
                    torch.arange(num_blocks, device=x.device, dtype=torch.int32) * self.block_size,
                    torch.arange(num_blocks, device=x.device, dtype=torch.int32) * self.block_size
                    + self.block_size,
                ],
                dim=-1,
            )
            mask = torch.zeros(num_blocks, num_blocks, device=x.device, dtype=torch.bool)
            start, end = make_ranges(self.cutoff, x, block_size=self.block_size, align=True)
            for i in range(len(start)):
                mask[i, start[i] // self.block_size : end[i] // self.block_size] = True
            self.ranges = from_matrix(block_index, block_index, mask)

    def prepare_epoch(self, params: Tensor) -> None:
        self.outputscale = params[:, 1, None, None]

        if self.cutoff is None:
            # This is almost equal to the GPyTorch implementation.
            lengthscale = params[:, 0, None]
            x = self.x[None, :] / lengthscale
            x1 = LazyTensor(x[..., :, None, None])
            x2 = LazyTensor(x[..., None, :, None])
            self.kernel = (-((x1 - x2) ** 2).sum(-1) * 0.5).exp()

        else:
            tkwargs = dict(device=params.device, dtype=torch.int32)
            b = params.shape[0]

            def repeat_batch(tensor: Tensor) -> Tensor:
                return (
                    (torch.arange(b, **tkwargs)[:, None, None] * len(self.x) + tensor[None, :, :])
                    .reshape(-1, tensor.shape[-1])
                    .to(**tkwargs)
                )

            def repeat_slices(slices: Tensor) -> Tensor:
                sizes = torch.cat([slices[0:1], slices[1:] - slices[:-1]])
                batch_sizes = sizes.repeat(b)
                return torch.cumsum(batch_sizes, dim=0).to(**tkwargs)

            ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i = self.ranges
            self.batch_ranges = (
                repeat_batch(ranges_i),
                repeat_slices(slices_i),
                repeat_batch(redranges_j),
                repeat_batch(ranges_j),
                repeat_slices(slices_j),
                repeat_batch(redranges_i),
            )

            lengthscale = params[:, 0, None]
            x = (self.x[None, :] / lengthscale).reshape(-1)
            x1 = LazyTensor(x[:, None, None])
            x2 = LazyTensor(x[None, :, None])
            self.kernel = (-((x1 - x2) ** 2).sum(-1) * 0.5).exp()

    def __call__(self, rhs: Tensor) -> Tensor:
        rhs = rhs.contiguous()
        if self.cutoff is None:
            result = self.kernel.__matmul__(rhs, use_fast_math=False)
            return self.outputscale * result
        else:
            self.kernel: LazyTensor
            result = self.kernel.__matmul__(
                rhs.reshape(-1, rhs.shape[-1]), ranges=self.batch_ranges, use_fast_math=False
            )
            result = self.outputscale * result.reshape(self.outputscale.shape[0], -1, rhs.shape[-1])
            return result


class BlockedNaiveMatmul(MatmulBase):
    def __init__(self, block_size: int, cutoff: float) -> None:
        super().__init__()
        self.block_size = block_size
        self.cutoff = cutoff

    def prepare_train(self, x: Tensor, kernel_type: str) -> None:
        assert kernel_type == "rbf"
        start, end = make_ranges(self.cutoff, x, block_size=self.block_size, align=True)
        self.x = x
        self.start = start
        self.end = end

    def prepare_epoch(self, params: Tensor) -> None:
        lengthscale = params[:, 0, None, None]
        outputscale = params[:, 1, None, None]
        self.kernels = []
        for i, (start, end) in enumerate(zip(self.start, self.end)):
            dist = sq_dist(
                self.x[i * self.block_size : (i + 1) * self.block_size, None],
                self.x[start:end, None],
            )[None, :, :]
            kernel = outputscale * (-0.5 * dist**2 / lengthscale**2).exp()
            self.kernels.append(kernel)

    def __call__(self, rhs: Tensor) -> Tensor:
        results = []
        for kernel, start, end in zip(self.kernels, self.start, self.end):
            results.append(kernel @ rhs[:, start:end])
        return torch.cat(results, dim=-2)
