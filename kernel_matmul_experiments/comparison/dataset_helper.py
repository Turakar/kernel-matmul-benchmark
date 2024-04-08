import csv
import datetime
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch import Tensor
from typing_extensions import Self


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

    def __len__(self):
        return len(self.x)

    def __add__(self: Self, other: Self) -> Self:
        assert self.start == other.start
        new_series = TimeSeries(
            start=self.start,
            x=torch.cat([self.x, other.x]),
            y=torch.cat([self.y, other.y]),
        )
        assert torch.argsort(new_series.x).equal(torch.arange(len(new_series)))
        return new_series


def load_monash(name: str = "traffic_hourly", split: str = "train") -> list[TimeSeries]:
    dt = {
        "australian_electricity_demand": 0.5 / 24,  # half-hourly data
        "traffic_hourly": 1 / 24,  # hourly data
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
    assert torch.allclose(train.y, val.y[: len(train)]) and torch.allclose(
        train.x, val.x[: len(train)]
    )
    return val[len(train) :]


def downsample(series: TimeSeries, from_size: int, to_size: int, method: str) -> TimeSeries:
    assert to_size <= from_size <= len(series.y)
    if method == "last":
        return series[-to_size:]
    elif method == "uniform":
        from_window = torch.linspace(len(series) - from_size, len(series))
        selection = torch.sort(torch.randperm(from_window)[:to_size]).values
        return series[selection]
    elif method == "average":
        import torch_scatter
        bins = torch.linspace(series.x[-from_size], series.x[-1], to_size + 1)
        indices = torch.bucketize(series.x, bins)
        new_x = (bins[:-1] + bins[1:]) / 2
        new_y = torch_scatter.scatter(series.y, indices, reduce="mean")
        return TimeSeries(start=series.start, x=new_x, y=new_y)
    else:
        raise ValueError(f"Unknown method: {method}")


def load_etth1():
    with open("data/ETT-small/ETTh1.csv") as f:
        reader = csv.DictReader(f)
        start = None
        x = []
        y = [[] for _ in reader.fieldnames[1:]]
        for row in reader:
            date = datetime.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
            if start is None:
                start = date
            x.append((date - start).total_seconds() / 3600 / 24)
            for i, key in enumerate(reader.fieldnames[1:]):
                y[i].append(float(row[key]))
    return TimeSeries(
        start=start,
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.stack([torch.tensor(yi, dtype=torch.float32) for yi in y], dim=-1),
    )


def get_hpo_subset(num_instances: int, index: int, size: int, seed: int = 0) -> list[int]:
    rng = torch.Generator().manual_seed(seed)
    for _ in range(index + 1):
        split = torch.randperm(num_instances, generator=rng)[:size].tolist()
    return split
