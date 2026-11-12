"""Predict specimen strengths from a trained Siamese graph model."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from .bundle import load_bundle, predict_strengths
from .data import load_specimens
from .training import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/siamese_gnn/model.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/pore_features.xlsx"))
    parser.add_argument("--specimen", action="append", help="Worksheet ID. Repeat for multiple specimens.")
    parser.add_argument("--output", type=Path, default=Path("outputs/siamese_gnn/predictions.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    fitted = load_bundle(args.checkpoint, device)
    dataset = fitted.config["dataset"]
    ids = args.specimen or dataset["test_ids"]
    specimens = load_specimens(args.data, dataset, ids, require_targets=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_id": [s.sample_id for s in specimens],
        "predicted_mpa": predict_strengths(fitted, specimens, device),
    }).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
