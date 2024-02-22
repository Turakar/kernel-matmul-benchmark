import collections
import math
from typing import Callable

import numpy as np
import plotly.graph_objs as go
import scipy.stats
import torch
from plotly.colors import sample_colorscale
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
        column_titles=[f"Batch size: {batch_size}" for batch_size in batch_size_values]
        if len(batch_size_values) > 1
        else None,
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
            plot_ranking_patch(
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


def plot_paper_figure(store: HierarchicalStore, params: dict) -> go.Figure:
    batch_size = params["params"]["batch_size"]["values"][0]
    cutoff_values = params["params"]["cutoff"]["values"]
    rhs_columns_values = params["params"]["rhs_columns"]["values"]

    fig = make_subplots(
        rows=2,
        cols=len(cutoff_values) + 1,
        column_titles=[f"Cutoff: {cutoff}" for cutoff in cutoff_values] + ["Dense"],
        vertical_spacing=0.08,
    )
    fig.update_layout(legend_tracegroupgap=0)

    # Collect data
    max_log_speedup = 0
    all_timings = []  # timings for each method participating
    all_improvements = []  # speedup over SOTA
    all_colors = []  # colors for the methods in timings
    all_deltas = []  # difference between SOTA and new method (for tests)
    num_samples_values_cutoff = []  # differs between dense and sparse
    for i, cutoff in enumerate(cutoff_values + [None]):
        num_samples_values = params["params"]["num_samples"]["values"]
        if cutoff is None:
            num_samples_values = [
                x
                for x in num_samples_values
                if x <= params["params"]["num_samples"]["max_dense_size"]
            ]
        num_samples_values_cutoff.append(num_samples_values)
        make_grid = lambda: [[None for _ in rhs_columns_values] for _ in num_samples_values]  # noqa: E731
        timings = collections.defaultdict(make_grid)
        all_timings.append(timings)
        improvement = make_grid()
        all_improvements.append(improvement)
        deltas = make_grid()
        all_deltas.append(deltas)
        for j, num_samples in enumerate(num_samples_values):
            for k, rhs_columns in enumerate(rhs_columns_values):
                baseline = float("inf")
                values_baseline = None
                new_timing = float("inf")
                values_new = None
                for method in params["methods"].keys():
                    name = params["methods"][method]["name"]
                    sparse = params["methods"][method].get("sparse", False)
                    if sparse and cutoff is None:
                        continue
                    type_ = params["methods"][method]["type"]
                    cutoff_ = cutoff if sparse else None
                    values = store[method][cutoff_][num_samples][batch_size][rhs_columns].value
                    value = values.median().item()
                    timings[name][j][k] = value
                    if type_ == "new" and not math.isnan(value):
                        new_timing = min(new_timing, value)
                        values_baseline = values
                    if type_ == "baseline" and not math.isnan(value):
                        baseline = min(baseline, value)
                        values_new = values
                speedup = baseline / new_timing
                improvement[j][k] = speedup
                max_log_speedup = max(max_log_speedup, abs(math.log10(speedup)))
                deltas[j][k] = values_baseline - values_new

        colors = []
        all_colors.append(colors)
        for name in timings.keys():
            for j, (method, method_params) in enumerate(params["methods"].items()):
                if method_params["name"] == name:
                    colors.append(CATEGORICAL_COLORS[j])
                    break
        colors.append(CATEGORICAL_COLORS[-1])

    # Perform tests
    p_values = []
    for i, cutoff in enumerate(cutoff_values + [None]):
        num_samples_values = params["params"]["num_samples"]["values"]
        if cutoff is None:
            num_samples_values = [
                x
                for x in num_samples_values
                if x <= params["params"]["num_samples"]["max_dense_size"]
            ]
        for j, num_samples in enumerate(num_samples_values_cutoff[i]):
            for k, rhs_columns in enumerate(rhs_columns_values):
                if not torch.any(~torch.isfinite(all_deltas[i][j][k])):
                    p_values.append(
                        scipy.stats.wilcoxon(
                            all_deltas[i][j][k].numpy(),
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
    p_values = scipy.stats.false_discovery_control(p_values).tolist()
    for i, cutoff in enumerate(cutoff_values + [None]):
        for j, num_samples in enumerate(num_samples_values_cutoff[i]):
            for k, rhs_columns in enumerate(rhs_columns_values):
                if not torch.any(~torch.isfinite(all_deltas[i][j][k])):
                    p_value = p_values.pop(0)
                    if p_value >= 0.05:
                        print(
                            f"Warning: p-value for {cutoff=}, {num_samples=}, {rhs_columns=} is >= 0.05 ({p_value})"
                        )
                        all_improvements[i][j][k] = 1.0
    assert len(p_values) == 0

    # Plot
    for i, (cutoff, timings, improvement, colors) in enumerate(
        zip(cutoff_values + [None], all_timings, all_improvements, all_colors)
    ):
        plot_ranking_patch(
            fig,
            timings,
            num_samples_values_cutoff[i],
            "Kernel size",
            rhs_columns_values,
            "RHS columns",
            row=1,
            col=i + 1,
            ghosttrace=i == 0,
            colors=colors,
        )
        plot_relative_improvement(
            fig,
            improvement,
            num_samples_values_cutoff[i],
            "Kernel size",
            rhs_columns_values,
            "RHS columns",
            row=2,
            col=i + 1,
        )

    # Update layout
    for i in range(2):
        fig.update_yaxes(title_text="RHS columns", row=i + 1, col=1)
    for i in range(len(cutoff_values) + 1):
        fig.update_xaxes(title_text="Kernel size", row=2, col=i + 1)
    fig.update_xaxes(
        tickmode="array",
        tickvals=[2048, 16384, 131072],
        ticktext=["2¹¹", "2¹⁴", "2¹⁷"],
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=[1, 4, 16, 64, 256],
        ticktext=["2⁰", "2²", "2⁴", "2⁶", "2⁸"],
    )
    fig.update_layout(
        legend=dict(
            xanchor="left",
            x=1.02,
            yanchor="top",
            y=1.05,
        )
    )
    fig.update_coloraxes(
        colorscale="RdBu",
        cmid=0,
        cmin=-max_log_speedup,
        cmax=max_log_speedup,
        colorbar=dict(
            xanchor="left",
            x=1.01,
            yanchor="top",
            y=0.5,
            len=0.535,
            tickmode="array",
            tickvals=[math.log10(x) for x in [0.2, 0.5, 1.0, 2.0, 5.0]],
            ticktext=["0.2x", "0.5x", "1x", "2x", "5x"],
        ),
    )

    return fig


def plot_relative_improvement(
    fig: go.Figure,
    improvement: list[list[float]],
    x_values: list[float | int],
    x_title: str,
    y_values: list[float],
    y_title: str,
    row: int = 1,
    col: int = 1,
    x_axis_type: str = "log",
    y_axis_type: str = "log",
    z_axis_type: str = "log",
) -> None:
    z_transform, z_inv_transform = create_transform_functions(z_axis_type)
    improvement_transformed = [
        [z_transform(improvement[i][j]) for j in range(len(y_values))] for i in range(len(x_values))
    ]
    min_improvement_transformed = min(min(row) for row in improvement_transformed)
    max_improvement_transformed = max(max(row) for row in improvement_transformed)
    transformed_range = max(-min_improvement_transformed, max_improvement_transformed)
    text = [
        [
            f"{x_title}: {x_values[i]}<br>{y_title}: {y_values[j]}<br>Speedup: {improvement[i][j]:.3f}x"
            for j in range(len(y_values))
        ]
        for i in range(len(x_values))
    ]
    colors = [
        [
            sample_colorscale(
                "RdBu",
                (improvement_transformed[i][j] + transformed_range) / (transformed_range * 2),
            )[0]
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
        colors,
        text,
    )
    fig.add_trace(
        go.Heatmap(
            z=improvement_transformed,
            x=[1e6 + i for i in range(len(x_values))],
            y=[1e6 + i for i in range(len(y_values))],
            showscale=True,
            coloraxis="coloraxis",
        ),
        row=row,
        col=col,
    )


def plot_ranking_patch(
    fig: go.Figure,
    timings: dict[str, list[list[float]]],
    x_values: list[float | int],
    x_title: str | None,
    y_values: list[float],
    y_title: str | None,
    x_axis_type: str = "log",
    y_axis_type: str = "log",
    row: int = 1,
    col: int = 1,
    ghosttrace: bool = False,
    colors: list[str] | None = None,
):
    if colors is None:
        colors = CATEGORICAL_COLORS

    # Helper function
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
                min(
                    [k for k in order if not math.isnan(timings[k][i][j])],
                    key=lambda k: timings[k][i][j],
                    default="OOM",
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
        [[colors[i] for i in row] for row in leaderboard],
        text,
        legendgroup=[[order_oom[i] for i in row] for row in leaderboard],
    )

    # These traces will be used for the legend.
    if ghosttrace:
        for name, color in zip(order, colors):
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
    x_transform, x_inv_transform = create_transform_functions(x_axis_type)
    y_transform, y_inv_transform = create_transform_functions(y_axis_type)

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


def create_transform_functions(
    axis_type: str,
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    if axis_type == "log":
        transform = np.log10
        inv_transform = lambda x: 10**x  # noqa: E731
    else:
        transform = lambda x: x  # noqa: E731
        inv_transform = lambda x: x  # noqa: E731

    return transform, inv_transform


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
