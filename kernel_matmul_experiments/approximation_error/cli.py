import multiprocessing
import warnings
import click
import gpytorch
from kernel_matmul.linear_operator import KernelMatmulLinearOperator
from kernel_matmul.ranges import make_ranges
import torch
import math
from tqdm import tqdm
from torch import Tensor
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from linear_operator.operators import LinearOperator, AddedDiagLinearOperator, DiagLinearOperator
import scipy.special
from linear_operator import settings
import plotly.express as px
from linear_operator.utils.warnings import NumericalWarning


@click.command()
@torch.no_grad()
def cli():
    num_x = 10000
    noise = 1e-1
    iterations = 10
    lengthscales = [0.5, 1.0, 5.0, 10.0, 20.0]
    epsilons = [0] + [float(e) for e in torch.logspace(-8, -3, 5 * 2 + 1)]
    low_tolerance = 1e-4

    try:
        results = torch.load("data/approximation_error.pt")

    except FileNotFoundError:
        with multiprocessing.get_context("spawn").Pool(processes=10) as pool:
            device = torch.device("cuda:0")
            x = torch.sort(torch.rand(iterations, num_x, device=device) * 100, dim=-1).values
            y = torch.sin(2 * torch.pi * x)
            y = y + torch.randn_like(y) * noise
            y = y / y.norm(dim=-1, keepdim=True)
            results = torch.zeros(len(lengthscales), len(epsilons), iterations)
            for i, lengthscale in enumerate(tqdm(lengthscales)):
                for j, epsilon in enumerate(tqdm(epsilons, leave=False)):
                    for iteration in range(iterations):
                        full_covar = make_covar(
                            x[iteration],
                            lengthscale,
                            noise,
                            0,
                            pool,
                        )
                        covar = make_covar(
                            x[iteration],
                            lengthscale,
                            noise,
                            epsilon,
                            pool,
                        )
                        solution, converged = cg_solve(covar, y[iteration], low_tolerance)
                        if epsilon == 0 and not converged:
                            print(f"Warning: λ = {lengthscale} did not converge for ϵ = 0")
                        residual = float(calc_residual(full_covar, y[iteration], solution))
                        results[i, j, iteration] = residual
        torch.save(results, "data/approximation_error.pt")

    results = results.nanmean(dim=-1)

    fig = make_subplots()
    for i in range(len(lengthscales)):
        fig.add_trace(
            go.Scatter(
                x=epsilons[1:],
                y=results[i, 1:],
                mode="lines+markers",
                line_color=px.colors.qualitative.T10[i],
                name=f"λ = {lengthscales[i]}",
            )
        )
        fig.add_hline(y=results[i, 0], line_dash="dash", line_color=px.colors.qualitative.T10[i])
    fig.add_hline(
        y=gpytorch.settings.eval_cg_tolerance.value(), line_dash="dash", line_color="black"
    )
    fig.update_xaxes(type="log", title="ϵ")
    fig.update_yaxes(
        type="log",
        title="Residual without cutoff",
        range=[math.log10(90e-6) * 1.01, math.log10(0.013)],
    )
    fig.write_html("data/approximation_error.html")
    fig.update_layout(width=650, height=200, margin=dict(t=0, l=0, r=0, b=0))
    fig.write_image("data/approximation_error.svg")


def make_covar(
    x: Tensor,
    lengthscale: float,
    noise: float,
    epsilon: float,
    pool: multiprocessing.Pool,
) -> LinearOperator:
    num_x = x.shape[-1]

    if epsilon > 0:
        cutoff = math.sqrt(2) * lengthscale * scipy.special.erfinv(1 - epsilon)
    else:
        cutoff = None
    start, end = make_ranges(cutoff, x)
    params = torch.tensor([lengthscale, 1.0, 1.0], device=x.device, dtype=x.dtype)

    covar = KernelMatmulLinearOperator(
        x.unsqueeze(-1),
        x.unsqueeze(-1),
        params,
        start,
        end,
        kernel_type="spectral",
        compile_pool=pool,
    )
    noise = DiagLinearOperator(torch.full((num_x,), noise**2, device=x.device))
    covar = AddedDiagLinearOperator(covar, noise)
    return covar


def cg_solve(operator: LinearOperator, y: Tensor, tolerance: float) -> tuple[Tensor, bool]:
    with (
        settings.cg_tolerance(tolerance),
        warnings.catch_warnings(record=True) as w,
        settings.max_cg_iterations(1000),
    ):
        solution = operator.solve(y)
    converged = not any(issubclass(warning.category, NumericalWarning) for warning in w)
    return solution, converged


def calc_residual(covar: LinearOperator | Tensor, y: Tensor, solution: Tensor) -> Tensor:
    assert len(y.shape) == len(solution.shape) == 1
    return (y - covar @ solution).norm()


if __name__ == "__main__":
    cli()
