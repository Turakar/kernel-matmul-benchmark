import collections
import glob
import re

import torch


def main():
    results = collections.defaultdict(list)
    for file in glob.glob("data/rebuttal/vs_keops/results/*.pt"):
        match = re.match(r".*/(.*)_\d_(.*)_\d+\.pt", file)
        if match is None:
            continue
        dataset, method = match.groups()
        results[(dataset, method)].append(torch.load(file))

    for (dataset, method), result in results.items():
        print(f"{dataset} {method}")
        time = torch.stack([r["time"] for r in result])
        mase = torch.stack([r["mase"] for r in result])
        print(f"{time.median().item():.2f}s, MASE {mase.mean().item():.4f}")


if __name__ == "__main__":
    main()
