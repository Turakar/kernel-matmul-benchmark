# KernelMatmul Experiments

This is the experiments repository for the "KernelMatmul: Scaling Gaussian Processes to Large Time Series" paper.
The actual implementation lives in the `kernel-matmul` submodule.
Please refer to the [README](kernel-matmul/README.md) __first__, as most of it applies to the experiments repository as well.
For reproducibilty, the package versions are fixed in the `poetry.lock` file and the `.devcontainer` configuration directory.

## Experiments
You can find several artifacts of (intermediate) results of our runs in `data/`.

### Performance Comparison
Code is in `kernel_matmul_experiments/benchmark/`.

- `cli.py`: Implements a CLI interface for running the benchmark and contains the main code.
- `implementations.py`: Contains the kernel matrix multiplication implementations.
- `plot.py`: Responsible for plotting the results.
- `store.py`: Defines a special data structure used for storing the results (see docstring).
- `util.py`: Utilities for timing and checking for out-of-memory exceptions.

The benchmark can be run using the following commands:

```bash
python kernel_matmul_experiments/benchmark/cli.py run --path mypath
python kernel_matmul_experiments/benchmark/cli.py make_plots --path mypath
```

This takes some time, roughly a few hours.

### Sparsity approximation error
Code is in `kernel_matmul_experiments/approximation_error/`.
There is only a single file `cli.py` which runs the experiment and plots the results.
This experiment is quick to execute.

### Accuracy comparison
Code is in `comparison/`.

- `cli.py`: Contains the code for running a single instance in the experiment (i.e., one of the HPOs) and for plotting the aggregate results.
- `dataset_helper.py`: Loads the dataset and aids in splitting and pre-processing the dataset.
- `hpo.py`: Implements the HPO using SMAC.
- `model.py`: Contains the model initialization and training.

The comparison must be run on a cluster due to the high computational cost of the HPOs.
To aid in parallelization, each HPO is run in a separate process.
For SLURM, you can build a custom batch script using the template provided in `sbatchx/`, which uses a tool named not released, yet.
However, customization to your cluster is necessary anyway and should have low complexity.
