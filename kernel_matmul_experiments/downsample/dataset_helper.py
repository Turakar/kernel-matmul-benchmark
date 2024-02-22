import datetime
from dataclasses import dataclass

import torch
import torch_scatter
from datasets import load_dataset
from torch import Tensor


@dataclass
class TimeSeries:
    start: datetime.datetime
    x: Tensor
    y: Tensor

    @property
    def datetimes(self):
        return [self.start + datetime.timedelta(days=float(x)) for x in self.x]

    def __getitem__(self, item):
        return TimeSeries(
            start=self.start,
            x=self.x[item],
            y=self.y[item],
        )


def load_monash(
    name: str = "australian_electricity_demand", split: str = "train"
) -> list[TimeSeries]:
    dt = {
        "australian_electricity_demand": 0.5 / 24,  # half-hourly data
    }[name]
    dataset = load_dataset("monash_tsf", name=name, split=split)
    return [
        TimeSeries(
            start=series["start"],
            x=torch.arange(0, len(series["target"]), dtype=torch.float32) * dt,
            y=torch.tensor(series["target"], dtype=torch.float32),
        )
        for series in dataset
    ]


def remove_train_from_val(train: TimeSeries, val: TimeSeries) -> TimeSeries:
    assert torch.allclose(train.y, val.y[: len(train)])
    return val.y[len(train) :]


def downsample(series: TimeSeries, from_size: int, to_size: int, method: str) -> TimeSeries:
    assert to_size <= from_size <= len(series.y)
    if method == "last":
        return series[-to_size:]
    elif method == "uniform":
        from_window = torch.linspace(len(series) - from_size, len(series))
        selection = torch.sort(torch.randperm(from_window)[:to_size]).values
        return series[selection]
    elif method == "average":
        bins = torch.linspace(series.x[-from_size], series.x[-1], to_size + 1)
        indices = torch.bucketize(series.x, bins)
        new_x = (bins[:-1] + bins[1:]) / 2
        new_y = torch_scatter.scatter(series.y, indices, reduce="mean")
        return TimeSeries(start=series.start, x=new_x, y=new_y)
    else:
        raise ValueError(f"Unknown method: {method}")
