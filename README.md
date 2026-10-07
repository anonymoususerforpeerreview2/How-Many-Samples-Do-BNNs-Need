# How Many Samples Do BNNs Need? Adaptive Inference Time with Statistical Guarantees

This repository contains the code to run the experiments from the paper.

To use the tracker library in your own code, see the
[library usage guide](docs/README.md).

## Python setup

Use Python 3.10 or 3.11. Run the commands below from the repository root.

### Install the Python packages

Prebuilt wheels are included in `wheels/` for x86-64 Windows and Linux, Intel macOS 14+, and Apple silicon macOS 15+. They install without C++ build tools. Choose the commands for your operating system:

#### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
# Install the tracker wheel from wheels/.
python -m pip install --find-links ./wheels adaptive-cs==0.1.0
# Install the experiment package in editable mode.
python -m pip install -e code "matplotlib>=3.8,<3.9" "scipy>=1.11,<1.14"
```

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install the tracker wheel from wheels/.
python -m pip install --find-links ./wheels adaptive-cs==0.1.0
# Install the experiment package in editable mode.
python -m pip install -e code "matplotlib>=3.8,<3.9" "scipy>=1.11,<1.14"
```

### Build the tracker from source (optional)

If you want to compile the tracker locally, follow the
[source-build instructions](library/README.md) and use
`python -m pip install ./library` in place of the tracker wheel command above.

## Toy simulation

No image datasets or GPU are needed. Generate the synthetic figures; each
command runs independently:

```bash
# Introductory example: fixed sample budgets for easy and ambiguous inputs.
python figures/intro_fig/generate.py

# Examples of the three stopping rules on synthetic prediction streams.
python figures/toy_examples/generate.py

# One example from each of the four simulated prediction settings.
python figures/toy_settings/generate.py

# Sample efficiency of adaptive rules versus fixed budgets and baselines.
python figures/toy_sample_efficiency/generate.py

# Samples saved by safe early abstention on inconclusive toy inputs.
python figures/early_stopping_evaluation/generate.py

# Illustration of the bound used for safe early abstention.
python figures/early_stopping_bound/generate.py

# Illustration of aleatoric and epistemic uncertainty.
python figures/epistemic_uncertainty/generate.py

# Local CPU runtime comparison of bounding-box implementations.
python figures/bbox_speedup/generate.py

# TOMS 748 root-finding breakpoint analysis.
python figures/toms_break_point_analysis/generate.py
```

These scripts use the [shared toy configuration](code/experiments/toy/config.yaml) and write PDFs and numerical summaries beside themselves. The bounding-box command measures eight local CPU workers, so its runtime values depend on the machine.

## Image experiments

The image benchmarks need a CUDA GPU. After the installation above, add their dependencies:

```bash
python -m pip install -r code/requirements.txt
```

Download the public CIFAR data:

```bash
python -m experiments.prepare_datasets
```

This downloads the original CIFAR-10/100 archives, CIFAR-10/100-C, and CIFAR-10/100-P into `code/data/datasets/`. The CIFAR-C files come from the [robustness repo](https://github.com/hendrycks/robustness#readme). Public ResNet-18 checkpoints are automatically downloaded by the experiment code.

For ImageNet, obtain the licensed ImageNet-1K validation and devkit archives and the four [ImageNet-C](https://github.com/hendrycks/robustness#readme) group archives. Place them as follows:

```text
code/data/datasets/imagenet/ILSVRC2012_img_val.tar
code/data/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz
code/data/datasets/imagenet/downloads/blur.tar
code/data/datasets/imagenet/downloads/digital.tar
code/data/datasets/imagenet/downloads/noise.tar
code/data/datasets/imagenet/downloads/weather.tar
```

Then fetch and verify ImageNet-P:

```bash
python -m experiments.imagenet.prepare_data
```

Run all three image benchmarks:

```bash
python -m experiments.run_vision
```

This runs CIFAR-10, CIFAR-100, and ImageNet-1K in that order. To run just one, append `cifar10`, `cifar100`, or `imagenet`. For each dataset it fits the CIFAR Laplace model where applicable, evaluates 1,000 clean/C/P inputs, calibrates inspection times on 100 disjoint inputs, repeats the evaluation, times adaptive and quality-matched fixed budgets, and runs 30 paired accuracy streams. The CIFAR-C severity is 5; the P result uses the final sequence frame. Results go to `code/data/<benchmark>/local/`.

After the image experiments finish, generate their figures and tables:

```bash
# Combined image-benchmark quality, stopping times, latency, and early abstention.
python figures/real_data_experiments/generate_all.py

# Class-selection accuracy and its variation across 30 prediction streams.
python figures/real_data_experiments/generate_accuracy_repeats.py

# CIFAR-100 class-selection sample efficiency and prior calibration.
python figures/real_data_experiments/sample_efficiency/generate.py

# CIFAR-100 distribution-accuracy and threshold-decision efficiency.
python figures/real_data_experiments/task_efficiency/generate.py
```
