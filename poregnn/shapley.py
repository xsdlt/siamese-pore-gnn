"""Estimate whole-pore and within-pore Shapley values."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from .bundle import load_bundle
from .data import load_specimens
from .training import resolve_device


class GraphScore:
    def __init__(self, fitted, query, references, device):
        self.model = fitted.model
        self.edges = query.graph.edge_index.to(device)
        self.wide = query.wide.to(device)
        self.references = references
        self.layouts = {}

    def __call__(self, states):
        scores = []
        for chunk in states.split(16):
            count, nodes, _ = chunk.shape
            if count == 1:
                chunk = chunk.expand(2, -1, -1)
            size = len(chunk)
            if size not in self.layouts:
                edges = torch.cat([self.edges + i * nodes for i in range(size)], dim=1)
                batch = torch.arange(size, device=chunk.device).repeat_interleave(nodes)
                self.layouts[size] = edges, batch
            edges, batch = self.layouts[size]
            graph = Data(x=chunk.flatten(0, 1), edge_index=edges, batch=batch)
            embeddings = self.model.forward_one(graph, self.wide.expand(size, -1))
            differences = embeddings[:, None, :] - self.references[None, :, :]
            logits = self.model.fusion(differences.flatten(0, 1)).reshape(size, -1)
            scores.append(logits.sigmoid().mean(dim=1)[:count])
        return torch.cat(scores).cpu().numpy().astype(np.float64)


def whole_pore_values(score, features, orders):
    n = len(features)
    totals = np.zeros(n)
    for permutation in orders:
        ranks = np.argsort(permutation)
        masks = torch.as_tensor(ranks < np.arange(n + 1)[:, None], device=features.device)
        totals[permutation] += np.diff(score(features * masks.unsqueeze(-1)))
    return totals / len(orders)


def inner_orders(seed, sample_id, outer_count, count, n_features):
    rng = np.random.default_rng(seed * 1000 + int(sample_id))

    def draw(width):
        return np.array([rng.permutation(n_features) for _ in range(outer_count * width * 4)]).reshape(
            outer_count, width, 4, n_features,
        )

    first = draw(5)
    if count > 5:
        return np.concatenate((first, draw(count - 5)), axis=1)
    return first[:, :count]


def within_pore_values(score, features, outer, inner, pores, zero_feature):
    totals = np.zeros((len(pores), features.shape[1]))
    for order, feature_orders in zip(outer, inner):
        ranks = torch.as_tensor(np.argsort(order), device=features.device)
        states, paths = [], []
        for index, pore in enumerate(pores):
            context = features * (ranks < ranks[pore]).unsqueeze(1)
            for permutation in feature_orders[index]:
                path = [feature for feature in permutation if feature != zero_feature]
                state = context.clone()
                states.append(state.clone())
                for feature in path:
                    state[pore, feature] = features[pore, feature]
                    states.append(state.clone())
                paths.append((index, path))
        values = score(torch.stack(states))
        start = 0
        for index, path in paths:
            end = start + len(path) + 1
            totals[index, path] += np.diff(values[start:end])
            start = end
    return totals / (len(outer) * 4)


@torch.inference_mode()
def explain(fitted, specimen, references, output, device, permutations, top, seed):
    query = fitted.processor.transform(specimen, fitted.config["graph"])
    features = query.graph.x.to(device)
    score = GraphScore(fitted, query, references, device)
    generator = torch.Generator().manual_seed(seed)
    orders = [torch.randperm(len(features), generator=generator).numpy() for _ in range(permutations)]
    grouped = whole_pore_values(score, features, orders)

    names = fitted.processor.final_features
    roi = fitted.processor.preprocess(specimen.frame).reset_index(drop=True)
    cols = [*fitted.config["dataset"]["coordinates"], "直径 [µm]"]
    all_pores = roi[cols].copy()
    all_pores.insert(0, "ROI_node_index_0based", np.arange(len(roi)))
    all_pores["Grouped_SHAP"] = grouped
    all_pores["Negative_Importance"] = np.maximum(-grouped, 0)
    ranked = all_pores[all_pores["Grouped_SHAP"] < 0].nsmallest(top, "Grouped_SHAP").copy()
    pores = ranked["ROI_node_index_0based"].to_numpy(dtype=int)
    if len(pores):
        zero = names.index("Local_Density")
        zero_feature = zero if bool(torch.all(features[:, zero] == 0)) else None
        inner = inner_orders(seed, specimen.sample_id, len(orders), len(pores), len(names))
        values = within_pore_values(score, features, orders, inner, pores, zero_feature)
    else:
        values = np.empty((0, len(names)))
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked[[f"SHAP_{name}" for name in names]] = values
    np.testing.assert_allclose(values.sum(1), grouped[pores], atol=2e-6, rtol=0)
    endpoints = score(torch.stack((torch.zeros_like(features), features)))
    np.testing.assert_allclose(grouped.sum(), np.diff(endpoints)[0], atol=5e-6, rtol=0)
    output.mkdir(parents=True, exist_ok=True)
    all_pores.to_csv(output / "all_pores.csv", index=False)
    ranked.to_csv(output / "top_pores.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/siamese_gnn/model.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/pore_features.xlsx"))
    parser.add_argument("--specimen", action="append", help="Worksheet ID. Repeat for multiple specimens.")
    parser.add_argument("--output", type=Path, default=Path("outputs/shapley"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--top", type=int, help="Default: ten pores for specimen 5, five for other specimens.")
    args = parser.parse_args()
    if args.permutations < 1 or (args.top is not None and args.top < 1):
        parser.error("--permutations and --top must be positive.")
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    fitted = load_bundle(args.checkpoint, device)
    if fitted.config["model"]["architecture"] != "experts4":
        parser.error("Supply a four-encoder checkpoint from poregnn.train.")
    dataset = fitted.config["dataset"]
    specimens = load_specimens(args.data, dataset, args.specimen or dataset["test_ids"], require_targets=False)
    with torch.inference_mode():
        graphs = Batch.from_data_list([r.graph for r in fitted.references]).to(device)
        wide = torch.stack([r.wide for r in fitted.references]).to(device)
        refs = fitted.model.forward_one(graphs, wide)
    for specimen in specimens:
        top = args.top or (10 if specimen.sample_id == "5" else 5)
        explain(fitted, specimen, refs, args.output / f"specimen_{specimen.sample_id}",
                device, args.permutations, top, args.seed)


if __name__ == "__main__":
    main()
