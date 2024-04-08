import collections
import json
import os

import click
import plotly.graph_objects as go
import smac
import torch
from ConfigSpace.read_and_write import json as cs_json
from plotly.subplots import make_subplots

from kernel_matmul_experiments.comparison.dataset_helper import (
    TimeSeries,
    downsample,
    get_hpo_subset,
    load_monash,
    remove_train_from_val,
)
from kernel_matmul_experiments.comparison.hpo import do_hpo


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--base-path", type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True
)
@click.option("--name", type=str, required=True)
@click.option("--method", type=click.Choice(["naive", "kernel-matmul", "ski"]), required=True)
@click.option("--hpo-subset-index", type=int, required=True)
@click.option("--hpo-subset-size", type=int, default=10)
@click.option("--results-path", type=str, required=True)
def run(
    base_path: str,
    name: str,
    method: str,
    hpo_subset_index: int,
    hpo_subset_size: int,
    results_path: str,
) -> None:
    train, val, test = load_data()
    hpo_subset = get_hpo_subset(len(train), hpo_subset_index, hpo_subset_size)
    result = do_hpo(name, base_path, train, val, test, hpo_subset, method=method)
    torch.save(result, results_path)


@cli.command()
@click.option(
    "--template", type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True
)
@click.option("--base-path", type=str, required=True)
def submit() -> None:
    pass


@cli.command()
@click.option("--smac-path", required=True, type=str)
def slice_plot(base_path: str) -> None:
    with open(os.path.join(base_path, "configspace.json")) as fd:
        config_space = cs_json.read(fd.read())
    runhistory = smac.RunHistory()
    runhistory.load(os.path.join(base_path, "runhistory.json"), config_space)
    print(f"{len(runhistory)} trials | {len(runhistory.get_configs())} configs")

    def average_costs():
        config_instances = {}
        for key in runhistory.keys():
            if key.config_id not in config_instances:
                config_instances[key.config_id] = set()
            config_instances[key.config_id].add(key.instance)
        results = []
        max_size = max(len(v) for v in config_instances.values())
        for k, v in config_instances.items():
            if len(v) == max_size:
                config = runhistory.ids_config[k]
                results.append((config, runhistory.average_cost(config)))
        results.sort(key=lambda x: x[1])
        return results

    average_costs = average_costs()

    fig = make_subplots(rows=1, cols=len(config_space), shared_yaxes="all")
    for i, param in enumerate(config_space.keys()):
        fig.add_trace(
            go.Scatter(
                x=[c.get(param, "None") for c, _ in average_costs],
                y=[v for _, v in average_costs],
                text=[repr(dict(c)) for c, _ in average_costs],
                mode="markers",
                marker_color=[runhistory.config_ids[c] for c, _ in average_costs],
                marker_colorscale="Blues",
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(title=param, row=1, col=i + 1)
        if param in ["noise", "lr"]:
            fig.update_xaxes(type="log", row=1, col=i + 1)
    fig.update_yaxes(type="log")
    fig.show(renderer="browser")


@cli.command()
@click.argument("base-path", type=str)
def analyze(base_path: str) -> None:
    jobs_path = os.path.join(base_path, "jobs")
    results_path = os.path.join(base_path, "results")
    values = collections.defaultdict(list)
    for filename in os.listdir(results_path):
        job_id = int(filename[: -len(".pt")])
        result = torch.load(os.path.join(results_path, filename))
        with open(os.path.join(jobs_path, f"{job_id}.json")) as fd:
            job = json.load(fd)
        values[job["sweep"]["method"]].append(result["results"].nanmean().item())
    print(values)


def load_data() -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries]]:
    train = load_monash(split="train")
    val = load_monash(split="validation")
    test = load_monash(split="test")
    test = [remove_train_from_val(val[i], test[i]) for i in range(len(val))]
    val = [remove_train_from_val(train[i], val[i]) for i in range(len(train))]
    train = [downsample(series, 10000, 10000, "last") for series in train]
    return train, val, test


if __name__ == "__main__":
    cli()
