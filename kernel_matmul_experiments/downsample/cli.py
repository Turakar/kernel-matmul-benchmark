import click
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kernel_matmul_experiments.downsample.dataset_helper import (
    downsample,
    load_monash,
    remove_train_from_val,
)
from kernel_matmul_experiments.downsample.hpo import do_hpo


@click.group()
def cli():
    pass


@cli.command()
def plot_data():
    dataset = load_monash()
    window = 5000
    fig = make_subplots(rows=len(dataset))
    for i, series in enumerate(dataset):
        series = series[-window:]
        fig.add_trace(go.Scatter(x=series.datetimes, y=series.y, mode="lines"), row=i + 1, col=1)
    fig.show()


@cli.command()
def foo():
    train = load_monash(split="train")
    val = load_monash(split="val")
    val = [remove_train_from_val(train[i], val[i]) for i in range(len(train))]
    train_downsampled = [downsample(series, 1000, 1000, "last") for series in train]

    config = do_hpo("foo", train_downsampled, val)
    print(config)


if __name__ == "__main__":
    cli()
