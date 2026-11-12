# Siamese Pore GNN

Code and processed pore data for ceramic strength prediction and attribution.

## Install

Use Python 3.10 and install the CPU dependencies from the repository root.

```bash
python -m pip install -r requirements-cpu.txt
```

For CUDA 12.1, use `requirements-cu121.txt`. Figure 6 uses Arial.

## Train, predict, and compute Shapley values

```bash
python -m poregnn.train
python -m poregnn.predict
python -m poregnn.shapley
```

Training writes `outputs/siamese_gnn/model.pt` and predictions for held-out
specimens 2, 5, and 13. Add `--loocv` to produce leave-one-out predictions on
the 18 training specimens. [Model settings](configs/siamese_gnn.json) specify
the architecture, training parameters, split, and seed.

The model uses four GraphConv encoders with TopK, Attention, Mean, and Max
pooling. Graph construction uses training-fitted MinMax coordinate scaling,
CPU `radius_graph` with radius 100000 and `max_num_neighbors=20`, and
symmetrization. Node features and specimen descriptors are standardized
using training-set statistics.

The Shapley script explains the mean sigmoid comparison score against the
training reference graphs, using zero standardized node features as the
baseline. It uses 100 whole-pore permutations and four within-pore
permutations per outer context. Outputs include grouped Shapley values for
all pores and signed feature attributions for the top negatively ranked
pores. The manuscript spatial panels for specimens 2 and 13 use a Siamese
model with three GraphConv encoders and 96 hidden channels per encoder.

## Reproduce Figure 6

```bash
python -m poregnn.integrated_gradients
```

The script loads [the Figure 6 model bundle](models/figure6_ig.pt), which
contains the three-encoder model, its scalers, and 19 reference graphs.
Integrated gradients are computed on the sum of sigmoid comparison scores
using 40 integration steps. For each query, the `ROI_Vol_Fraction` input is
the sum of pore diameters. The script writes PNG, PDF, and attribution tables
to `outputs/figure6/`.

## Data

[The workbook](data/pore_features.xlsx) contains pores within the model region
(`-2500 <= X <= -2000` micrometres). Worksheets 1–21 contain model inputs and
measured strengths in MPa. Worksheet 22 supplies an additional query for
Figure 6. Worksheet names are specimen IDs. Pore dimensions and coordinates
are in micrometres. The MIT license applies to the source code.
