import copy
import functools
import multiprocessing
import os
import tomllib
import click
import torch
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from rich.live import Live
from rich.table import Table
from rich.status import Status
from rich.text import Text
import operator
import implementations
from implementations import MatmulBase
from torch import Tensor
from kernel_matmul_experiments.benchmark.store import HierarchicalStore

from kernel_matmul_experiments.benchmark.util import is_oom_exception, timeit
from kernel_matmul_experiments.benchmark import plot


@click.group()
def cli():
    pass


@cli.command()
@click.option("--path", type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option("--iterations", type=int, default=50)
def run(path: str, iterations: int) -> None:
    # Load files
    params_path = os.path.join(path, "params.toml")
    out_path = os.path.join(path, "output.pt")
    with open(params_path, "rb") as fd:
        params = tomllib.load(fd)
    try:
        store = HierarchicalStore.from_dict(torch.load(out_path))
    except FileNotFoundError:
        store = HierarchicalStore()

    # Setup
    tkwargs = dict(device="cuda", dtype=torch.float32)

    # Setup progress bar
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} of {task.total}"),
        TimeElapsedColumn(),
    )
    status = Status(Text("Preparing..."))
    table = Table.grid()
    table.add_row(progress)
    table.add_row(status)

    with (
        # Progress bar renderer
        Live(table),
        # No gradients for this benchmark
        torch.no_grad(),
        # Passed to NativeFunction() for parallel compilation
        multiprocessing.get_context("spawn").Pool(processes=10) as compile_pool,
    ):
        num_tasks = 0
        for method in params["methods"].keys():
            if params["methods"][method].get("sparse", False):
                num_tasks += functools.reduce(
                    operator.mul,
                    (
                        len(params["params"][key]["values"])
                        for key in ["cutoff", "num_samples", "batch_size", "rhs_columns"]
                    ),
                    1,
                )
            else:
                num_tasks += functools.reduce(
                    operator.mul,
                    (
                        len(params["params"][key]["values"])
                        for key in ["num_samples", "batch_size", "rhs_columns"]
                    ),
                    1,
                )
        task = progress.add_task("Benchmarking", total=num_tasks)

        # Iterate through all combinations of parameters with lazy initialization
        for method in params["methods"].keys():
            sparse = params["methods"][method].get("sparse", False)
            for cutoff in params["params"]["cutoff"]["values"] if sparse else [None]:
                impl: MatmulBase | None = None
                for num_samples in params["params"]["num_samples"]["values"]:
                    x: Tensor | None = None
                    for batch_size in params["params"]["batch_size"]["values"]:
                        parameters: Tensor | None = None
                        for rhs_columns in params["params"]["rhs_columns"]["values"]:
                            # Check if already done
                            if store[method][cutoff][num_samples][batch_size][
                                rhs_columns
                            ].has_value:
                                progress.advance(task)
                                continue

                            # Prepare implementation
                            status.update(Text(f"{method} cutoff={cutoff}"))
                            if impl is None:
                                try:
                                    method_params = copy.deepcopy(params["methods"][method])
                                    if "max_size" in method_params:
                                        del method_params["max_size"]
                                    if "sparse" in method_params:
                                        if method_params["sparse"]:
                                            method_params["cutoff"] = cutoff
                                        del method_params["sparse"]
                                    clz = getattr(implementations, method_params["class"])
                                    if clz == implementations.KernelMatmul:
                                        method_params["compile_pool"] = compile_pool
                                    del method_params["class"]
                                    del method_params["name"]
                                    impl = clz(**method_params)
                                except Exception as e:
                                    if is_oom_exception(e):
                                        store[method][cutoff].value = torch.tensor([float("nan")])
                                        torch.save(store.to_dict(), out_path)
                                        progress.advance(task)
                                        continue
                                    else:
                                        raise e

                            # Prepare data
                            status.update(
                                Text(f"{method} cutoff={cutoff} num_samples={num_samples}")
                            )
                            if x is None:
                                try:
                                    if num_samples > params["methods"][method].get(
                                        "max_size", float("inf")
                                    ):
                                        store[method][cutoff][num_samples].value = torch.tensor(
                                            [float("nan")]
                                        )
                                        torch.save(store.to_dict(), out_path)
                                        progress.advance(task)
                                        continue
                                    dt = 1 / 24
                                    torch.manual_seed(0)
                                    x = torch.sort(
                                        torch.rand(num_samples, **tkwargs) * dt * num_samples
                                    )[0]
                                    impl.prepare_train(x, "rbf")
                                except Exception as e:
                                    if is_oom_exception(e):
                                        store[method][cutoff][num_samples].value = torch.tensor(
                                            [float("nan")]
                                        )
                                        for x in params["params"]["num_samples"]["values"]:
                                            if x > num_samples:
                                                store[method][cutoff][x].value = torch.tensor(
                                                    [float("nan")]
                                                )
                                        torch.save(store.to_dict(), out_path)
                                        progress.advance(task)
                                        continue
                                    else:
                                        raise e

                            # Prepare parameters
                            status.update(
                                Text(
                                    f"{method} cutoff={cutoff} num_samples={num_samples} batch_size={batch_size}"
                                )
                            )
                            if parameters is None:
                                try:
                                    torch.manual_seed(0)
                                    parameters = (
                                        torch.ones(batch_size, 2, **tkwargs)
                                        + torch.randn(batch_size, 2, **tkwargs) * 0.01
                                    )
                                    impl.prepare_epoch(parameters)
                                except Exception as e:
                                    if is_oom_exception(e):
                                        store[method][cutoff][num_samples][
                                            batch_size
                                        ].value = torch.tensor([float("nan")])
                                        torch.save(store.to_dict(), out_path)
                                        progress.advance(task)
                                        continue
                                    else:
                                        raise e

                            # Prepare RHS
                            status.update(
                                Text(
                                    f"{method} cutoff={cutoff} num_samples={num_samples} batch_size={batch_size} rhs_columns={rhs_columns}"
                                )
                            )
                            try:
                                # Construct a RHS similar to how it would like for
                                # CG inference with an LMC.
                                torch.manual_seed(0)
                                rhs = torch.randn(1, num_samples, rhs_columns, **tkwargs)
                                rhs = rhs / rhs.norm(dim=1, keepdim=True)
                                rhs = rhs.expand(batch_size, num_samples, rhs_columns)

                                # Benchmark
                                timings = timeit(impl, rhs, iterations)
                                store[method][cutoff][num_samples][batch_size][
                                    rhs_columns
                                ].value = timings

                            # Handle OOM
                            except Exception as e:
                                if is_oom_exception(e):
                                    store[method][cutoff][num_samples][batch_size][
                                        rhs_columns
                                    ].value = torch.tensor([float("nan")])
                                else:
                                    raise e

                            # Save
                            torch.save(store.to_dict(), out_path)
                            progress.advance(task)


@cli.command()
@click.option("--path", type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
def make_plots(path: str) -> None:
    store = HierarchicalStore.from_dict(torch.load(os.path.join(path, "output.pt")))
    with open(os.path.join(path, "params.toml"), "rb") as fd:
        params = tomllib.load(fd)
    fig = plot.plot_ranking(store, params)
    fig.write_html(os.path.join(path, "ranking.html"))
    fig.write_image(os.path.join(path, "ranking.pdf"))
    fig.show()

    fig = plot.plot_ranking(store, params, only_dense=True)
    fig.show()


if __name__ == "__main__":
    cli()
