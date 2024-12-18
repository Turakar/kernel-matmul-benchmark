import click
import smac
import torch
from tqdm import tqdm

from kernel_matmul_experiments.comparison.cli import load_data
from kernel_matmul_experiments.comparison.dataset_helper import downsample
from kernel_matmul_experiments.comparison.hpo import SmacModel


@click.command()
@click.option("--dataset", type=str)
@click.option("--hpo-subset-index", type=int)
@click.option("--results-path", type=str)
def main(dataset: str, hpo_subset_index: int, results_path: str):
    analysis = torch.load("data/comparison/v3/analysis.pt")

    run_history: smac.RunHistory = analysis[dataset]["kernel-matmul"][hpo_subset_index][
        "run_history"
    ]
    incumbent = min(run_history.get_configs(), key=lambda x: run_history.get_min_cost(x))
    print(incumbent)

    train, val, test = load_data(dataset)

    smac_model = SmacModel(train, val, "naive")
    results = []
    for i in tqdm(range(len(test)), desc="Testing"):
        train_series = downsample(train[i] + val[i], 10000, 5000, "last")
        # train_series = train[i] + val[i]
        mse = smac_model.evaluate(incumbent, train_series, test[i], seed=0)
        results.append(mse)
    results = torch.tensor(results)
    torch.save(results, results_path)
    print(
        f"Test loss: {results.nanmean().item()} (NaN: {torch.count_nonzero(results.isnan()).item()}/{len(results)})"
    )


if __name__ == "__main__":
    main()
