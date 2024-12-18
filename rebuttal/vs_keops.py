import time

import click
import torch
from tqdm import tqdm

from kernel_matmul_experiments.comparison.cli import load_data
from kernel_matmul_experiments.comparison.hpo import SmacModel


@click.command()
@click.option("--dataset", type=str)
@click.option("--hpo-subset-index", type=int)
@click.option("--results-path", type=str)
@click.option("--method", type=str)
def main(dataset: str, hpo_subset_index: int, results_path: str, method: str):
    print(f"Dataset: {dataset}")
    print(f"HPO subset index: {hpo_subset_index}")
    print(f"Method: {method}")
    print(f"Results path: {results_path}")

    analysis = torch.load("data/comparison/v3/analysis.pt")
    train, val, test = load_data(dataset)

    run_history = analysis[dataset]["kernel-matmul"][hpo_subset_index]["run_history"]
    incumbent = min(run_history.get_configs(), key=lambda x: run_history.get_min_cost(x))
    print(incumbent)

    smac_model = SmacModel(train, val, method)
    results = {"time": [], "mase": []}
    for i in tqdm(range(len(test)), desc="Testing"):
        prior_time = time.time()
        mase = smac_model.evaluate(incumbent, train[i] + val[i], test[i], seed=0)
        results["time"].append(time.time() - prior_time)
        results["mase"].append(mase.item())
    results["time"] = torch.tensor(results["time"])
    results["mase"] = torch.tensor(results["mase"])
    torch.save(results, results_path)
    print(
        f"Test loss: {results['mase'].nanmean().item()} (NaN: {torch.count_nonzero(results['mase'].isnan()).item()}/{len(results['mase'])})"
    )
    print(
        f"Time: {results['time'].median().item()} (min: {results['time'].min().item()}, max: {results['time'].max().item()})"
    )


if __name__ == "__main__":
    main()
