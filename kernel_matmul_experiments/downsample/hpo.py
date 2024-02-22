import math

import gpytorch
import torch
from ConfigSpace import Configuration, ConfigurationSpace
from smac import MultiFidelityFacade, Scenario

from kernel_matmul_experiments.downsample.dataset_helper import TimeSeries
from kernel_matmul_experiments.downsample.model import make_gp, train_gp


class SmacModel:
    def __init__(self, train: list[TimeSeries], val: list[TimeSeries]):
        self.train = train
        self.val = val

        # Calculate lengthscale s.t. k(tau) >= 0.9 for an RBF kernel at max tau
        max_timespan = max((series.x[-1] - series.x[0]).item() for series in val)
        self.lengthscale = math.sqrt(-0.5 * max_timespan**2 / math.log(0.9))

    @property
    def configspace(self) -> ConfigurationSpace:
        from ConfigSpace import Categorical, EqualsCondition, Float, Integer

        cs = ConfigurationSpace()
        lr = Float("lr", 1e-3, 1e0, log=True, default=1e-2)
        noise = Float("noise", 1e-3, 1e1, log=True, default=1e-2)
        num_components = Integer("num_components", 1, 100, default=10)
        train = Categorical("train", [True, False], default=True)
        peak_distance = Integer("peak_distance", 1, 50, default=5)
        peak_oversampling = Integer("peak_oversampling", 1, 20, default=1)
        cs.add_hyperparameters([lr, noise, num_components, train, peak_distance, peak_oversampling])
        cs.add_condition(EqualsCondition(child=lr, parent=train, value=True))
        return cs

    def train(self, config: Configuration, instance: str) -> float:
        instance = int(instance)
        model = make_gp(
            self.train[instance].x,
            self.train[instance].y,
            lengthscale=self.lengthscale,
            num_components=config["num_components"],
            max_frequency=48.0,
            peak_distance=config["peak_distance"],
            peak_oversample=config["peak_oversampling"],
            noise=config["noise"],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if config["train"]:
            train_gp(model, lr=config["lr"])
        model.eval()
        with torch.no_grad(), gpytorch.settings.skip_posterior_variances():
            prediction = model(self.val[instance].x.to(device)).mean.cpu()
        return float(torch.mean((prediction - self.val[instance].y) ** 2))


def do_hpo(
    name: str, train: list[TimeSeries], val: list[TimeSeries], n_trials: int = 100
) -> Configuration:
    model = SmacModel(train, val)
    scenario = Scenario(
        model.configspace,
        n_trials=n_trials,
        min_budget=1,
        max_budget=len(train),
        instances=[str(i) for i in range(len(train))],
        name=name,
        use_default_config=True,
    )
    smac = MultiFidelityFacade(
        scenario,
        model.train,
    )
    incumbent = smac.optimize()
    return incumbent
