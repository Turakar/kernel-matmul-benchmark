import math
import multiprocessing
import warnings

import click
import gpytorch
import plotly.express as px
import plotly.graph_objects as go
import scipy.special
import torch
from kernel_matmul.linear_operator import KernelMatmulLinearOperator
from kernel_matmul.ranges import make_ranges
from linear_operator import settings
from linear_operator.operators import AddedDiagLinearOperator, DiagLinearOperator, LinearOperator
from linear_operator.utils.warnings import NumericalWarning
from plotly.subplots import make_subplots
from torch import Tensor
from tqdm import tqdm


@click.command()
@torch.no_grad()
def cli():
    num_x = 10000
    num_val = 1000
    noise = 1e-1
    iterations = 10
    lengthscales = [0.5, 1.0, 5.0, 10.0, 20.0]
    epsilons = [0] + [float(e) for e in torch.logspace(-8, -3, 5 * 2 + 1)]
    low_tolerance = 1e-4

    try:
        results = torch.load("data/approximation_error.pt")
        if "test_mse" not in results and "val_mse" in results:
            print("Warning: Renaming 'val_mse' to 'test_mse' in data/approximation_error.pt")
            results["test_mse"] = results.pop("val_mse")
        torch.save(results, "data/approximation_error.pt")

    except FileNotFoundError:
        with multiprocessing.get_context("spawn").Pool(processes=10) as pool:
            device = torch.device("cuda:0")
            x = torch.sort(
                torch.rand(iterations, num_x + num_val, device=device) * 100, dim=-1
            ).values
            y = torch.sin(2 * torch.pi * x)
            x, x_val = x[:, :num_x], x[:, num_x:]
            y, y_val = y[:, :num_x], y[:, num_x:]
            y = y + torch.randn_like(y) * noise
            norm = y.norm(dim=-1, keepdim=True)
            y = y / norm
            y_val = y_val / norm
            results = {
                "residual": torch.zeros(len(lengthscales), len(epsilons), iterations),
                "test_mse": torch.zeros(len(lengthscales), len(epsilons), iterations),
            }
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
                        results["residual"][i, j, iteration] = residual

                        solution_sloppy, _ = cg_solve(
                            covar, y[iteration], gpytorch.settings.eval_cg_tolerance.value()
                        )
                        cross_covar = make_covar(
                            x[iteration],
                            lengthscale,
                            noise,
                            epsilon,
                            pool,
                            x_val=x_val[iteration],
                        )
                        prediction = cross_covar @ solution_sloppy
                        mse = float(((prediction - y_val[iteration]) ** 2).mean()) * norm[iteration]
                        results["test_mse"][i, j, iteration] = mse
        torch.save(results, "data/approximation_error.pt")

    results["residual"] = results["residual"].nanmean(dim=-1)
    results["test_mse"] = results["test_mse"].nanmean(dim=-1)

    fig = make_subplots(rows=1, cols=2)
    colors = px.colors.qualitative.T10
    for i in range(len(lengthscales)):
        fig.add_trace(
            go.Scatter(
                x=epsilons[1:],
                y=results["residual"][i, 1:],
                mode="lines+markers",
                line_color=colors[i],
                line_width=1.5,
                marker_size=5,
                name=f"λ = {lengthscales[i]}",
                legendgroup=f"λ = {lengthscales[i]}",
            )
        )
        fig.add_hline(
            y=results["residual"][i, 0], line_dash="dash", line_color=colors[i], row=1, col=1
        )
    fig.add_hline(
        y=gpytorch.settings.eval_cg_tolerance.value(),
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        type="log",
        title=dict(
            text="Residual without cutoff",
            standoff=10,
        ),
        range=[math.log10(90e-6) * 1.01, math.log10(0.013)],
        tickmode="array",
        tickvals=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
        ticktext=["1e-4", "2e-4", "5e-4", "1e-3", "2e-3", "5e-3", "1e-2"],
        row=1,
        col=1,
    )
    for i in range(len(lengthscales)):
        fig.add_trace(
            go.Scatter(
                x=epsilons,
                y=results["test_mse"][i],
                mode="lines+markers",
                line_color=colors[i],
                line_width=1.5,
                marker_size=5,
                name=f"λ = {lengthscales[i]}",
                legendgroup=f"λ = {lengthscales[i]}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(
        title=dict(
            text="Test MSE",
            standoff=2,
        ),
        type="log",
        tickmode="array",
        tickvals=[1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6],
        ticktext=["1e-4", "1e-2", "1e0", "1e2", "1e4", "1e6"],
        row=1,
        col=2,
    )
    fig.update_xaxes(
        type="log",
        title=dict(
            text="ϵ",
            standoff=10,
        ),
        tickmode="array",
        tickvals=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        ticktext=["1e-8", "1e-7", "1e-6", "1e-5", "1e-4", "1e-3"],
    )
    fig.update_layout(legend_tracegroupgap=0)
    fig.write_html("data/approximation_error.html")
    fig.update_layout(width=650, height=200, margin=dict(t=0, l=0, r=0, b=0))
    font_size = 9
    fig.update_layout(font_size=font_size, legend_font_size=font_size)
    fig.update_annotations(font_size=font_size)
    tickfont_size = int(0.8 * font_size)
    fig.update_xaxes(tickfont_size=tickfont_size, title_font_size=font_size)
    fig.update_yaxes(tickfont_size=tickfont_size, title_font_size=font_size)
    fig.update_coloraxes(colorbar_tickfont_size=font_size)
    fig.write_image("data/approximation_error.svg")


def make_covar(
    x: Tensor,
    lengthscale: float,
    noise: float,
    epsilon: float,
    pool: multiprocessing.Pool,
    x_val: Tensor | None = None,
) -> LinearOperator:
    num_x = x.shape[-1]

    if epsilon > 0:
        cutoff = math.sqrt(2) * lengthscale * scipy.special.erfinv(1 - epsilon)
    else:
        cutoff = None
    if x_val is None:
        start, end = make_ranges(cutoff, x)
    else:
        start, end = make_ranges(cutoff, x_val, x)
    params = torch.tensor([lengthscale, 1.0, 1.0], device=x.device, dtype=x.dtype)

    covar = KernelMatmulLinearOperator(
        x.unsqueeze(-1) if x_val is None else x_val.unsqueeze(-1),
        x.unsqueeze(-1),
        params,
        start,
        end,
        kernel_type="spectral",
        compile_pool=pool,
    )
    if x_val is None:
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
