# Siamese Pore GNN

Code and processed pore data for ceramic strength prediction and attribution.

## Install

Use Python 3.10 and run the following command from the repository root.

```bash
python -m pip install -r requirements-cpu.txt
```

For CUDA 12.1, use `requirements-cu121.txt`.

## Train, predict, and compute Shapley values

```bash
python -m poregnn.train
python -m poregnn.predict
python -m poregnn.shapley
```

[Model settings](configs/siamese_gnn.json) · [Processed data](data/pore_features.xlsx)

## Integrated gradients

```bash
python -m poregnn.integrated_gradients
```

[Model file](models/ig_model.pt) · Figure and tables in `outputs/figure6/`
