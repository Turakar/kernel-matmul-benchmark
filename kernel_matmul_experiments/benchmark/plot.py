import math
from typing import Callable
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots

from kernel_matmul_experiments.benchmark.store import HierarchicalStore


CATEGORICAL_COLORS = ["#fcde9c", "#ffa552", "#63b995", "#ced01f", "#7392b7", "#c4769e", "#ffffff"]


def plot_ranking(store: HierarchicalStore, params: dict, only_dense: bool = False) -> go.Figure:
    method_keys = params["methods"].keys()
    cutoff_values = params["params"]["cutoff"]["values"]

    if only_dense:
        method_keys = [k for k in method_keys if not params["methods"][k].get("sparse", False)]
        cutoff_values = [None]

    batch_size_values = params["params"]["batch_size"]["values"]
    num_samples_values = params["params"]["num_samples"]["values"]
    rhs_columns_values = params["params"]["rhs_columns"]["values"]
    fig = make_subplots(
        rows=len(cutoff_values),
        cols=len(batch_size_values),
        row_titles=[f"Cutoff: {cutoff}" for cutoff in cutoff_values]
        if len(cutoff_values) > 1
        else None,
        column_titles=[f"Batch size: {batch_size}" for batch_size in batch_size_values],
    )

    for i, cutoff in enumerate(cutoff_values):
        for j, batch_size in enumerate(batch_size_values):
            timings = {}
            for method in method_keys:
                name = params["methods"][method]["name"]
                sparse = params["methods"][method].get("sparse", False)
                timings[name] = [
                    [
                        store[method][cutoff if sparse else None][num_samples][batch_size][
                            rhs_columns
                        ]
                        .value.median()
                        .item()
                        for rhs_columns in rhs_columns_values
                    ]
                    for num_samples in num_samples_values
                ]
            fig = plot_ranking_patch(
                fig,
                timings,
                num_samples_values,
                "Kernel size",
                rhs_columns_values,
                "RHS columns",
                row=i + 1,
                col=j + 1,
            )

    return fig


def plot_ranking_patch(
    fig: go.Figure,
    timings: dict[str, list[list[float]]],
    x_values: list[float | int],
    x_title: str,
    y_values: list[float],
    y_title: str,
    x_axis_type: str = "log",
    y_axis_type: str = "log",
    row: int = 1,
    col: int = 1,
) -> go.Figure:
    # Helper functions
    def min_or_oom(values: list[str], key_fn: Callable[[str], float]) -> str:
        """Return the minimum value by key_fn, returning OOM if none present."""
        if len(values) == 0:
            return "OOM"
        else:
            return min(values, key=key_fn)

    def bf(text: str, condition: bool) -> str:
        """Return bold text if condition is true."""
        if condition:
            return "<b>" + text + "</b>"
        else:
            return text

    # Define an order for the different keys.
    order = list(timings.keys())
    order_oom = order + ["OOM"]

    # For each parameter combination, find the best method (OOM if none succeded).
    # Store the index of the best method (or OOM) in `leaderboard`.
    leaderboard = [
        [
            order_oom.index(
                min_or_oom(
                    [k for k in order if not math.isnan(timings[k][i][j])],
                    key_fn=lambda k: timings[k][i][j],
                )
            )
            for j in range(len(y_values))
        ]
        for i in range(len(x_values))
    ]

    # For each parameter combination, create a text with the timings of all methods.
    text = [
        [
            f"{x_title}: {x_values[i]}<br>{y_title}: {y_values[j]}<br><br>"
            + "<br>".join(
                bf(
                    f"{k}: {timings[k][i][j]:.3f} ms",
                    leaderboard[i][j] < len(order) and order[leaderboard[i][j]] == k,
                )
                for k in order
            )
            for j in range(len(y_values))
        ]
        for i in range(len(x_values))
    ]

    draw_grid(
        fig,
        row,
        col,
        x_values,
        x_axis_type,
        y_values,
        y_axis_type,
        [[CATEGORICAL_COLORS[i] for i in row] for row in leaderboard],
        text,
        [[order_oom[i] for i in row] for row in leaderboard],
    )
    fig.update_xaxes(title_text=x_title, row=row, col=col)
    fig.update_yaxes(title_text=y_title, row=row, col=col)

    # These traces will be used for the legend.
    if row == col == 1:
        for name, color in zip(order, CATEGORICAL_COLORS):
            fig.add_trace(
                go.Scatter(
                    x=[1e6],
                    y=[1e6],
                    name=name,
                    mode="lines",
                    fill="toself",
                    fillcolor=color,
                    line_color=color,
                    showlegend=True,
                    legendgroup=name,
                ),
                row=row,
                col=col,
            )

    return fig


def draw_grid(
    fig: go.Figure,
    row: int,
    col: int,
    x_values: list[float],
    x_axis_type: str,
    y_values: list[float],
    y_axis_type: str,
    colors: list[list[str]],
    text: list[list[str]],
    legendgroup: list[list[str]] | None = None,
) -> None:
    # Define transformations from data to plot coordinates
    x_transform = np.log10 if x_axis_type == "log" else lambda x: x
    x_inv_transform = lambda x: 10**x  # noqa: E731
    y_transform = np.log10 if y_axis_type == "log" else lambda x: x
    y_inv_transform = lambda x: 10**x  # noqa: E731

    # Draw the 2D surface with one scatter plot per cell
    borders, (x_min, x_max, y_min, y_max) = compute_grid_borders(
        x_values,
        x_transform,
        x_inv_transform,
        y_values,
        y_transform,
        y_inv_transform,
    )
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            # Draw the cell
            x_left, x_right, y_bottom, y_top = borders[i][j]
            color = colors[i][j]
            fig.add_trace(
                go.Scatter(
                    fill="toself",
                    mode="lines",
                    line=dict(width=2, color=color),
                    fillcolor=color,
                    text=text[i][j],
                    name="",
                    x=[x_left, x_left, x_right, x_right, x_left],
                    y=[y_bottom, y_top, y_top, y_bottom, y_bottom],
                    showlegend=False,
                    legendgroup=legendgroup[i][j] if legendgroup is not None else None,
                ),
                row=row,
                col=col,
            )

    # Axes layout
    fig.update_xaxes(
        type=x_axis_type,
        range=[x_transform(x_min), x_transform(x_max)],
        row=row,
        col=col,
    )
    fig.update_yaxes(
        type=y_axis_type,
        range=[y_transform(y_min), y_transform(y_max)],
        row=row,
        col=col,
    )


def compute_grid_borders(
    x_values: list[float],
    x_transform: Callable[[float], float],
    x_inv_transform: Callable[[float], float],
    y_values: list[float],
    y_transform: Callable[[float], float],
    y_inv_transform: Callable[[float], float],
) -> tuple[list[list[tuple[float, float, float, float]]], tuple[float, float, float, float]]:
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    positions = []
    for i in range(len(x_values)):
        positions.append([])
        for j in range(len(y_values)):
            # Compute border locations
            if i == 0:
                next_distance = x_transform(x_values[1]) - x_transform(x_values[0])
                x_left = x_inv_transform(x_transform(x_values[0]) - next_distance / 2)
                x_min = x_left
            else:
                x_left = x_inv_transform(
                    (x_transform(x_values[i - 1]) + x_transform(x_values[i])) / 2
                )
            if i == len(x_values) - 1:
                prior_distance = x_transform(x_values[-1]) - x_transform(x_values[-2])
                x_right = x_inv_transform(x_transform(x_values[-1]) + prior_distance / 2)
                x_max = x_right
            else:
                x_right = x_inv_transform(
                    (x_transform(x_values[i]) + x_transform(x_values[i + 1])) / 2
                )
            if j == 0:
                next_distance = y_transform(y_values[1]) - y_transform(y_values[0])
                y_bottom = y_inv_transform(x_transform(y_values[0]) - next_distance / 2)
                y_min = y_bottom
            else:
                y_bottom = y_inv_transform(
                    (y_transform(y_values[j - 1]) + y_transform(y_values[j])) / 2
                )
            if j == len(y_values) - 1:
                prior_distance = y_transform(y_values[-1]) - y_transform(y_values[-2])
                y_top = y_inv_transform(y_transform(y_values[-1]) + prior_distance / 2)
                y_max = y_top
            else:
                y_top = y_inv_transform(
                    (y_transform(y_values[j]) + y_transform(y_values[j + 1])) / 2
                )
            positions[-1].append((x_left, x_right, y_bottom, y_top))
    return positions, (x_min, x_max, y_min, y_max)
