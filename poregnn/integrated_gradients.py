"""Compute integrated gradients and render Figure 6."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import AttentionalAggregation, GraphConv, global_max_pool, global_mean_pool, radius_graph
from torch_geometric.utils import to_dense_batch, to_undirected

COORDINATES = ["中间 x [µm]", "中间 y [µm]", "中间 z [µm]"]
DIAMETER = "直径 [µm]"
LABELS = {
    DIAMETER: r"$\mathbfit{D_{pore}}$", "球度": r"$\mathbfit{S_{ph}}$",
    "Angle_with_Load": r"$\mathbfit{\theta_{load}}$",
    "投影尺寸 x [µm]": r"$\mathbfit{L_X}$", "投影尺寸 y [µm]": r"$\mathbfit{L_Y}$",
    "投影尺寸 z [µm]": r"$\mathbfit{L_Z}$", "Ratio_XY": r"$\mathbfit{R_{XY}}$",
    "Griffith_Factor": r"$\mathbfit{D_{pore}^{-1/2}}$",
}


class PoreEncoder(nn.Module):
    def __init__(self, node_dim, hidden, wide_dim, pooling, k):
        super().__init__()
        self.conv1 = GraphConv(node_dim, hidden, aggr="add")
        self.pooling = pooling
        self.k = k
        if pooling == "attention":
            self.attn = AttentionalAggregation(nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, 1),
            ))
        width = hidden * 2 if pooling == "mean_max" else hidden
        self.bn = nn.BatchNorm1d(width)
        self.fc = nn.Sequential(nn.Linear(width + wide_dim, 64), nn.BatchNorm1d(64),
                                nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 32))

    def forward(self, graph, wide):
        x = self.conv1(graph.x, graph.edge_index).relu()
        batch = graph.batch
        if self.pooling == "top_k":
            dense, _ = to_dense_batch(x, batch)
            pooled = dense.topk(min(dense.size(1), self.k), dim=1).values.mean(dim=1)
        elif self.pooling == "attention":
            pooled = self.attn(x, batch)
        else:
            pooled = torch.cat((global_mean_pool(x, batch), global_max_pool(x, batch)), dim=1)
        return self.fc(torch.cat((self.bn(pooled), wide), dim=1))


class Figure6SiameseModel(nn.Module):
    def __init__(self, node_dim, wide_dim, hidden_dim, top_k):
        super().__init__()
        self.experts = nn.ModuleList(
            PoreEncoder(node_dim, hidden_dim, wide_dim, pool, top_k)
            for pool in ("top_k", "attention", "mean_max")
        )
        self.fusion = nn.Sequential(nn.Linear(96, 64), nn.BatchNorm1d(64),
                                    nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 1))

    def encode(self, graph, wide):
        return torch.cat([expert(graph, wide) for expert in self.experts], dim=1)


def query_inputs(frame, scalers):
    roi = frame.loc[frame[COORDINATES[0]].between(-2500, -2000)].copy()
    diameters = roi[DIAMETER].to_numpy()
    roi["Local_Density"] = 0
    roi["Ratio_XY"] = roi["投影尺寸 x [µm]"] / roi["投影尺寸 y [µm]"].replace(0, 1e-6)
    roi["Griffith_Factor"] = 1 / np.sqrt(roi[DIAMETER].replace(0, 1e-6).abs())
    node = scalers["node_scaler"]
    coord = scalers["coord_scaler"]
    wide = scalers["wide_scaler"]
    x = (roi[node["feature_names_in_"]].to_numpy() - node["mean_"]) / node["scale_"]
    pos = roi[COORDINATES].to_numpy() * coord["scale_"] + coord["min_"]
    edges = to_undirected(radius_graph(torch.tensor(pos, dtype=torch.float), r=100000,
                                      loop=False, max_num_neighbors=20))
    # Figure 6 uses the sum of pore diameters for this query input.
    descriptors = {
        "ROI_Rmax": diameters.max(), "ROI_Mean_Diameter": diameters.mean(),
        "ROI_Vol_Fraction": diameters.sum(), "ROI_Mean_Sphericity": roi["球度"].mean(),
    }
    wide_input = np.array([descriptors[name] for name in wide["feature_names_in_"]])
    wide_input = (wide_input - wide["mean_"]) / wide["scale_"]
    return torch.tensor(x, dtype=torch.float), edges, torch.tensor(wide_input, dtype=torch.float)


def specimen_ig(model, refs, x, edges, wide, steps=40):
    total = torch.zeros_like(x)
    wide = wide.expand(len(refs), -1)
    for step in range(1, steps + 1):
        state = (x * (step / steps)).requires_grad_(True)
        query = Batch.from_data_list([Data(x=state, edge_index=edges)] * len(refs))
        score = model.fusion(model.encode(query, wide) - refs).sigmoid().sum()
        gradient, = torch.autograd.grad(score, state)
        total += gradient
    return (x * (total / steps)).sum(dim=0).detach().numpy()


def render(attributions, feature_means, output):
    values = attributions[list(LABELS)]
    order = values.abs().mean().sort_values(ascending=False).index
    importance = values[order].abs().mean()
    colors = feature_means[order]
    colors = (colors - colors.min()) / (colors.max() - colors.min()).replace(0, 1)
    cmap = LinearSegmentedColormap.from_list("feature_value", ["#1E90FF", "#9400D3", "#FF0000"])
    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans"], "font.weight": "bold",
        "axes.labelweight": "bold", "axes.unicode_minus": False,
        "mathtext.fontset": "custom", "mathtext.bfit": "Arial:bold:italic",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(5.9, 11.3))
    bars = fig.add_axes([.15, .602, .83, .392])
    scatter = fig.add_axes([.15, .088, .665, .392])
    color_axis = fig.add_axes([.813, .225, .023, .125])
    rows = np.arange(len(order))
    bars.barh(rows, importance / importance.max(), height=.65, color="#0074D9",
              edgecolor="black", linewidth=1.4)
    bars.set(xlim=(0, 1.05), xticks=[0, .25, .5, .75, 1], xlabel="Mean Absolute IG Value")
    rng = np.random.default_rng(43)
    for i, name in enumerate(order):
        scatter.scatter(values[name], i + rng.normal(0, .12, len(values)),
                        c=cmap(colors[name]), s=52, alpha=.85, edgecolors="none")
    scatter.axvline(0, color="black", linewidth=1.4)
    scatter.set(xlim=(-11.7, 4.3), xticks=[-10, -5, 0], xlabel="IG Value")
    for ax, label, label_x in ((bars, "(a)", .86), (scatter, "(b)", 1.08)):
        ax.set_yticks(rows, [LABELS[name] for name in order])
        ax.set_ylim(len(order) - .27, -.75)
        ax.tick_params(axis="both", labelsize=17)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.label.set_size(18)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_linewidth(1.5)
        ax.text(label_x, .08, label, transform=ax.transAxes, fontsize=21, fontweight="normal")
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                      cax=color_axis, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.ax.tick_params(labelsize=14, length=0)
    cb.set_label("Feature Value", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"figure6_ig.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/figure6_ig.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/pore_features.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figure6"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    scalers = bundle["scalers"]
    names = scalers["node_scaler"]["feature_names_in_"]
    model = Figure6SiameseModel(len(names), len(scalers["wide_scaler"]["feature_names_in_"]),
                        **bundle["architecture"]).eval()
    model.load_state_dict(bundle["model_state"])
    references = bundle["references"]
    with torch.no_grad():
        graphs = Batch.from_data_list([Data(x=r["x"], edge_index=r["edge_index"]) for r in references])
        refs = model.encode(graphs, torch.stack([r["wide"] for r in references]))
    frames = pd.read_excel(args.data, sheet_name=None)
    ids = [str(i) for i in range(1, 23)]
    attributions, feature_means = [], []
    for sid in ids:
        x, edges, wide = query_inputs(frames[sid], scalers)
        attributions.append(specimen_ig(model, refs, x, edges, wide))
        feature_means.append(x.mean(dim=0).numpy())
    index = pd.Index(ids, name="sample_id")
    ig = pd.DataFrame(attributions, index=index, columns=names)
    means = pd.DataFrame(feature_means, index=index, columns=names)
    args.output.mkdir(parents=True, exist_ok=True)
    ig.to_csv(args.output / "ig_signed_specimen.csv")
    means.to_csv(args.output / "ig_feature_means.csv")
    render(ig, means, args.output)


if __name__ == "__main__":
    main()
