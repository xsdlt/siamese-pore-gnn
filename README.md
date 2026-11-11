# Siamese Pore GNN

Code and processed pore data for ceramic strength prediction and attribution.

## Install

Use Python 3.10 and install the pinned CPU dependencies from the repository root:

```bash
python -m pip install -r requirements-cpu.txt
```

Use `requirements-cu121.txt` for CUDA 12.1. Arial is recommended for rendering Figure 6.

## Train and explain E7

```bash
python -m poregnn.train
python -m poregnn.predict
python -m poregnn.shapley
```

Training writes `outputs/e7/model.pt` and predictions for held-out specimens 2,
5, and 13. Add `--loocv` to also produce leave-one-out predictions on the
18 development specimens. `configs/e7.json` gives the fixed architecture,
training parameters, split, and seed. Model and pair-sampling generators are
reset for each fit. CUDA and CPU arithmetic can give slightly different fits.

E7 uses four GraphConv experts with TopK, Attention, Mean, and Max pooling.
The graph builder applies training-fitted MinMax coordinate scaling,
CPU `radius_graph` with radius 100000 and `max_num_neighbors=20`, and
symmetrization. The neighbor parameter is passed to the candidate search
before symmetrization. Node and specimen features use training-fitted
standardization.

SHAP uses the fitted E7 bundle and defaults to 100 whole-pore permutations
and four within-pore permutations per outer context. It explains the mean
sigmoid comparison score against training references with a zero standardized
node-feature baseline. The graph and specimen features stay fixed. The output
contains all grouped pore scores and signed feature scores for the top negative
pores. The E7 entry point reports E7 attributions. The manuscript spatial
panels for specimens 2 and 13 were generated from a separate R96 model.

## Reproduce Figure 6

```bash
python -m poregnn.integrated_gradients
```

This command uses the included `models/figure6_ig.pt` and writes the PNG, PDF,
and two numeric CSVs under `outputs/figure6/`. The bundle contains the frozen
three-expert checkpoint, scalers, and 19 reference graphs used for Figure 6.
The historical IG protocol uses the sum of comparison scores and 40 integration
steps. It retains the historical query-side sum of pore diameters in the
descriptor named `ROI_Vol_Fraction`. Scatter jitter is seeded for repeatable
rendering. E7 checkpoints are generated locally by the training command.

## Data

`data/pore_features.xlsx` contains the processed pores in the model ROI
(`-2500 <= X <= -2000` micrometres), preserving specimen and pore order.
Worksheet names are specimen IDs. Coordinates and pore dimensions use
micrometres. Specimens 1–21 include measured strength in MPa. Worksheet 22
contains geometry for the historical Figure 6 run and is excluded from E7
training and evaluation. The MIT license applies to the source code.
