import collections
import json
import os

import click
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
import smac
import torch
from ConfigSpace.read_and_write import json as cs_json
from plotly.subplots import make_subplots
from tqdm import tqdm

from kernel_matmul_experiments.comparison.dataset_helper import (
    TimeSeries,
    downsample,
    get_hpo_subset,
    load_monash,
    remove_nan,
    remove_train_from_val,
)
from kernel_matmul_experiments.comparison.hpo import do_hpo

PLOT_COLORS = px.colors.qualitative.T10


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--base-path", type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True
)
@click.option("--name", type=str, required=True)
@click.option(
    "--method",
    type=click.Choice(["naive", "kernel-matmul", "ski", "ski-exact", "vnngp"]),
    required=True,
)
@click.option("--hpo-subset-index", type=int, required=True)
@click.option("--hpo-subset-size", type=int, default=10)
@click.option("--results-path", type=str, required=True)
@click.option(
    "--timeout", type=float, default=4 * 60 * 60, help="Timeout for optimization in seconds."
)
@click.option("--dataset", type=str, default="traffic_hourly")
def run(
    base_path: str,
    name: str,
    method: str,
    hpo_subset_index: int,
    hpo_subset_size: int,
    results_path: str,
    timeout: float,
    dataset: str,
) -> None:
    train, val, test = load_data(dataset)
    hpo_subset = get_hpo_subset(len(train), hpo_subset_index, hpo_subset_size)
    result = do_hpo(name, base_path, train, val, test, hpo_subset, method=method, timeout=timeout)
    torch.save(result, results_path)


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
@click.option("--refresh", is_flag=True, help="Recompute aggregate results")
def analyze(base_path: str, refresh: bool) -> None:
    found = False
    analysis_path = os.path.join(base_path, "analysis.pt")
    if not refresh:
        try:
            values = torch.load(analysis_path)
            found = True
        except FileNotFoundError:
            pass

    if not found or refresh:
        values = collections.defaultdict(lambda: collections.defaultdict(list))
        jobs_path = os.path.join(base_path, "jobs")
        results_path = os.path.join(base_path, "results")
        for filename in tqdm(os.listdir(results_path), desc="Loading"):
            job_id = int(filename[: -len(".pt")])

            result = torch.load(os.path.join(results_path, filename))
            with open(os.path.join(jobs_path, f"{job_id}.json")) as fd:
                job = json.load(fd)

            smac_dir = os.path.join(base_path, "smac", str(job_id), "0")
            with open(os.path.join(smac_dir, "configspace.json")) as fd:
                config_space = cs_json.read(fd.read())
            run_history = smac.RunHistory()
            run_history.load(os.path.join(smac_dir, "runhistory.json"), config_space)

            values[job["sweep"]["dataset"]][job["sweep"]["method"]].append(
                {
                    "job": job,
                    "result": result["results"],
                    "config_space": config_space,
                    "run_history": run_history,
                }
            )
        values = {k: dict(v) for k, v in values.items()}
        torch.save(values, analysis_path)

    human_readable_methods = {
        "vnngp": "VNNGP",
        "ski": "SKI",
        "ski-exact": "SKI",
        "kernel-matmul": "KernelMatmul",
    }
    human_readable_datasets = {
        "traffic_hourly": "Traffic",
        "solar_10_minutes": "Solar",
        "electricity_hourly": "Electricity",
        "london_smart_meters": "London Smart Meters",
    }
    dataset_keys = sorted(list(values.keys()))
    # method_keys = sorted(list(values[dataset_keys[0]].keys()))
    method_keys = ["kernel-matmul", "ski-exact", "vnngp"]

    for dataset in dataset_keys:
        print(f"Dataset: {human_readable_datasets[dataset]}")
        for method in method_keys:
            print(f"  Method: {human_readable_methods[method]}")
            for entry in values[dataset][method]:
                print(
                    f"    {entry['job']['sweep']['hpo_subset_index']:2} "
                    f"{entry['result'].nanmean().item():.3f} {entry['run_history'].finished}"
                )

    print("Aggregate results:")
    for dataset in dataset_keys:
        print(f"Dataset: {human_readable_datasets[dataset]}")
        for method in method_keys:
            results = ", ".join(
                f"{entry['result'].nanmean().item():.3f}" for entry in values[dataset][method]
            )
            print(f"  {human_readable_methods[method]:13}: {results}")

    pvalues = []
    for dataset in dataset_keys:
        kernel_matmul = np.array(
            [entry["result"].nanmean().item() for entry in values[dataset]["kernel-matmul"]]
        )
        for method in method_keys:
            if method == "kernel-matmul":
                continue
            other = np.array(
                [entry["result"].nanmean().item() for entry in values[dataset][method]]
            )
            pvalue = scipy.stats.wilcoxon(kernel_matmul - other, alternative="less").pvalue
            pvalues.append(pvalue)
    pvalues = scipy.stats.false_discovery_control(pvalues).tolist()
    for dataset in dataset_keys:
        print(f"Dataset: {human_readable_datasets[dataset]}")
        for method in method_keys:
            if method == "kernel-matmul":
                continue
            pvalue = pvalues.pop(0)
            print(
                f"  KernelMatmul vs {human_readable_methods[method]}: p={pvalue:.1e} {'*' if pvalue < 0.05 else ''}"
            )

    fig = make_subplots(
        rows=2,
        cols=len(dataset_keys),
        subplot_titles=[human_readable_datasets[dataset] for dataset in dataset_keys],
        vertical_spacing=0.05,
    )
    for i, dataset in enumerate(dataset_keys):
        for j, method in enumerate(method_keys):
            results = [entry["result"].nanmean().item() for entry in values[dataset][method]]
            fig.add_trace(
                go.Box(
                    y=results,
                    name=human_readable_methods[method],
                    legendgroup=method,
                    showlegend=i == 0,
                    marker_color=PLOT_COLORS[j],
                    marker_size=2,
                    line_width=1,
                    boxpoints="all",
                    jitter=0.5,
                    pointpos=-2.0,
                ),
                row=1,
                col=i + 1,
            )
            fig.add_trace(
                go.Box(
                    y=[4 * 60 * 60 / len(x["run_history"]) for x in values[dataset][method]],
                    name=human_readable_methods[method],
                    legendgroup=human_readable_methods[method],
                    showlegend=False,
                    marker_color=PLOT_COLORS[j],
                    marker_size=2,
                    line_width=1,
                    boxpoints="all",
                    jitter=0.5,
                    pointpos=-2.0,
                ),
                row=2,
                col=i + 1,
            )
    fig.update_layout(showlegend=True, boxgap=0.5)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(title_text="Test MASE", row=1, col=1)
    fig.update_yaxes(title_text="Seconds per trial", row=2, col=1)
    fig.write_html("data/comparison.html")

    fig.update_layout(width=500, height=250, margin=dict(t=12, l=0, r=0, b=0))
    font_size = 9
    fig.update_layout(font_size=font_size, legend_font_size=font_size)
    fig.update_annotations(font_size=font_size)
    tickfont_size = int(0.8 * font_size)
    fig.update_xaxes(tickfont_size=tickfont_size, title_font_size=font_size)
    fig.update_yaxes(tickfont_size=tickfont_size, title_font_size=font_size)
    fig.update_coloraxes(colorbar_tickfont_size=font_size)
    fig.write_image("data/comparison.svg")


def load_data(
    dataset: str, min_series_length: int = 10000, max_size: int = 1000
) -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries]]:
    train = load_monash(name=dataset, split="train")
    val = load_monash(name=dataset, split="validation")
    test = load_monash(name=dataset, split="test")

    subset = [i for i in range(len(train)) if len(train[i].x) >= min_series_length]
    if len(subset) > max_size:
        subset = subset[:max_size]
    train = [remove_nan(train[i]) for i in subset]
    val = [remove_nan(val[i]) for i in subset]
    test = [remove_nan(test[i]) for i in subset]

    test = [remove_train_from_val(val[i], test[i]) for i in range(len(val))]
    val = [remove_train_from_val(train[i], val[i]) for i in range(len(train))]
    train = [downsample(series, 10000, 10000, "last") for series in train]

    return train, val, test


if __name__ == "__main__":
    cli()
