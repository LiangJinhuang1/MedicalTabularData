# Medical Tabular Data

Machine learning experiments on medical tabular data.

## Overview

This repository provides a framework for machine learning on clinical tabular data. The objective is to compare predictive performance across multiple architectures and representation-learning approaches for regression tasks on outcomes such as LVEF, LACm and LACa.

## Setup

- Python 3.x
- Dependencies in `requirements.txt` (PyTorch, pandas, scikit-learn, PyYAML, etc.)

```bash
pip install -r requirements.txt
```

## Usage

- Training and hyperparameters are configured via YAML under `configs/`.
- After running the training script, results are written to `output/`, including `best_results.csv` and logs.

## Output

Each run produces a folder `output/<run_id>/` with: best results table, training losses, full config, and logs.
