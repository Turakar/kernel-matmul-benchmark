import math
import warnings
from pathlib import Path

import gpytorch
import linear_operator.utils.warnings
import torch
from ConfigSpace import Configuration, ConfigurationSpace
from kernel_matmul.compile import compile_pool
from smac import Callback, MultiFidelityFacade, Scenario
from smac.main.smbo import SMBO
from smac.runhistory import TrialInfo, TrialValue
from tqdm import tqdm

from kernel_matmul_experiments.comparison.dataset_helper import TimeSeries
from kernel_matmul_experiments.comparison.model import make_gp, train_gp


class SmacModel:
    def __init__(self, train: list[TimeSeries], val: list[TimeSeries], method: str):
        self.train_data = train
        self.val_data = val
        self.method = method
        # Calculate lengthscale s.t. k(tau) >= 0.9 for an RBF kernel at max tau
        max_timespan = max((series.x[-1] - series.x[0]).item() for series in val)
        self.lengthscale = math.sqrt(-0.5 * max_timespan**2 / math.log(0.9))

    @property
    def configspace(self) -> ConfigurationSpace:
        from ConfigSpace import Categorical, EqualsCondition, Float, Integer

        cs = ConfigurationSpace()
        lr = Float("lr", (1e-3, 1e0), log=True, default=1e-2)
        noise = Float("noise", (1e-3, 1e1), log=True, default=1e-2)
        num_components = Integer("num_components", (1, 100), default=10)
        train = Categorical("train", [True, False], default=True)
        peak_distance = Integer("peak_distance", (1, 100), default=5)
        peak_oversampling = Integer("peak_oversampling", (1, 20), default=1)
        max_frequency = Float("max_frequency", (1, 100), log=True)
        cs.add_hyperparameters(
            [lr, noise, num_components, train, peak_distance, peak_oversampling, max_frequency]
        )
        cs.add_condition(EqualsCondition(child=lr, parent=train, value=True))
        return cs

    def train(self, config: Configuration, instance: str, seed: int) -> float:
        instance = int(instance)
        return self.evaluate(
            config,
            self.train_data[instance],
            self.val_data[instance],
            seed,
        )

    def evaluate(
        self, config: Configuration, train: TimeSeries, test: TimeSeries, seed: int
    ) -> float:
        torch.manual_seed(seed)
        with gpytorch.settings.max_preconditioner_size(0):
            mean, std = train.y.mean(), train.y.std()
            train_y_norm = (train.y - mean) / std
            model = make_gp(
                train.x,
                train_y_norm,
                lengthscale=self.lengthscale,
                num_components=config["num_components"],
                max_frequency=config["max_frequency"],
                peak_distance=config["peak_distance"],
                peak_oversample=config["peak_oversampling"],
                noise=config["noise"],
                method=self.method,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            if config["train"]:
                train_gp(model, lr=config["lr"])
            with (
                torch.no_grad(),
                gpytorch.settings.skip_posterior_variances(),
                gpytorch.settings.max_cg_iterations(10000),
                warnings.catch_warnings(),
            ):
                warnings.filterwarnings(
                    "ignore", category=linear_operator.utils.warnings.NumericalWarning
                )
                model.eval()
                prediction = model(test.x.to(device)).mean.cpu() * std + mean
            return mean_absolute_scaled_error(test.y, prediction, train.y)


class NumMaxBudgetConfigs(Callback):
    def __init__(self, max_n_configs: int | None):
        super().__init__()
        self.max_n_configs = max_n_configs
        self.last_print = 0

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        config_instances = {}
        for key in smbo.runhistory.keys():
            if key.config_id not in config_instances:
                config_instances[key.config_id] = set()
            config_instances[key.config_id].add(key.instance)
        current = sum(
            1
            for instances in config_instances.values()
            if len(instances) == smbo._scenario.max_budget
        )
        if current - self.last_print > 0:
            print(f"Current number of configurations at maximum budget: {current}")
            self.last_print = current
        if self.max_n_configs is not None and current >= self.max_n_configs:
            return False
        return None


def mean_absolute_scaled_error(
    y_true: torch.Tensor, y_pred: torch.Tensor, y_train: torch.Tensor
) -> torch.Tensor:
    mae = torch.abs(y_true - y_pred).nanmean()
    scale = torch.abs(y_train[1:] - y_train[:-1]).nanmean()
    return mae / scale


def do_hpo(
    name: str,
    out: str,
    train: list[TimeSeries],
    val: list[TimeSeries],
    test: list[TimeSeries],
    hpo_subset: list[int],
    n_configs: int = 100,
    method: str = "kernel-matmul",
) -> dict:
    model = SmacModel(
        [train[i] for i in hpo_subset],
        [val[i] for i in hpo_subset],
        method,
    )
    scenario = Scenario(
        model.configspace,
        n_trials=int(1e12),
        min_budget=1,
        max_budget=len(hpo_subset),
        instances=[str(i) for i in range(len(hpo_subset))],
        instance_features={str(i): [i] for i in range(len(hpo_subset))},
        name=name,
        output_directory=Path(out) / "smac",
        use_default_config=True,
        walltime_limit=60 * 60 * 2,  # 2 hours
    )
    smac = MultiFidelityFacade(
        scenario,
        model.train,
        overwrite=False,
        callbacks=[NumMaxBudgetConfigs(n_configs)],
    )
    with compile_pool(10):
        incumbent = smac.optimize()
        results = []
        for i in tqdm(range(len(test)), desc="Testing"):
            mse = model.evaluate(incumbent, train[i] + val[i], test[i], seed=0)
            results.append(mse)
        results = torch.tensor(results)
        print(
            f"Test loss: {results.nanmean().item()} (NaN: {torch.count_nonzero(results.isnan()).item()}/{len(results)})"
        )

    return {
        "name": name,
        "hpo_subset": hpo_subset,
        "n_configs": n_configs,
        "incumbent": incumbent,
        "results": results,
    }
