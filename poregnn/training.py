"""Ordered specimen pairs and strength-difference-weighted training."""

import random
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from .data import PoreDataProcessor
from .model import SiamesePoreGNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select --device cpu or auto.")
    return device


class OrderedPairs(Dataset):
    def __init__(self, specimens):
        self.pairs = []
        for a, b in combinations(specimens, 2):
            difference = abs(a.strength - b.strength)
            self.pairs.append((a.graph, a.wide, b.graph, b.wide,
                               float(a.strength > b.strength), difference))
            self.pairs.append((b.graph, b.wide, a.graph, a.wide,
                               float(b.strength > a.strength), difference))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def collate_pairs(batch):
    ga, wa, gb, wb, labels, differences = zip(*batch)
    return (Batch.from_data_list(ga), torch.stack(wa), Batch.from_data_list(gb),
            torch.stack(wb), torch.tensor(labels, dtype=torch.float),
            torch.tensor(differences, dtype=torch.float))


def weighted_pair_loss(logits, labels, differences, maximum=5.0):
    # Equal-strength pairs have zero weight, including an all-tied batch.
    denominator = differences.mean().clamp_min(torch.finfo(differences.dtype).eps)
    weights = torch.clamp(differences / denominator, max=maximum)
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean()


@dataclass
class FittedModel:
    model: SiamesePoreGNN
    processor: PoreDataProcessor
    references: list
    config: dict
    history: list


def fit_model(specimens, config, device, seed=None):
    if len(specimens) < 2 or any(s.strength is None for s in specimens):
        raise ValueError("Training requires at least two labeled specimens.")
    if seed is None:
        seed = config["training"]["seed"]
    set_seed(seed)
    processor = PoreDataProcessor(config["dataset"]).fit(specimens)
    references = [processor.transform(s, config["graph"]) for s in specimens]
    pairs = OrderedPairs(references)
    cfg = config["training"]
    if len(pairs) % cfg["batch_size"] == 1:
        raise ValueError("The final pair batch would contain one item. Choose another batch size for BatchNorm.")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(pairs, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=collate_pairs, num_workers=0, generator=generator)
    model = SiamesePoreGNN(references[0].graph.num_node_features,
                           references[0].wide.numel(), config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["cosine_t_max"], eta_min=cfg["min_learning_rate"])
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        seen = 0
        for ga, wa, gb, wb, labels, differences in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(ga.to(device), wa.to(device), gb.to(device), wb.to(device))
            loss = weighted_pair_loss(logits, labels.to(device), differences.to(device), cfg["max_pair_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch + 1}.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            seen += labels.numel()
        history.append({"epoch": epoch + 1, "weighted_bce": total_loss / seen,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step()
    model.eval()
    return FittedModel(model, processor, references, config, history)
