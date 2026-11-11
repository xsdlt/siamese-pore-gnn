"""Fit E7, evaluate its held-out specimens, and save a prediction bundle."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .bundle import predict_strengths, save_bundle
from .data import load_specimens, split_specimens
from .training import fit_model, resolve_device


def evaluate(fitted, specimens, device):
    return pd.DataFrame({
        "sample_id": [s.sample_id for s in specimens],
        "measured_mpa": [s.strength for s in specimens],
        "predicted_mpa": predict_strengths(fitted, specimens, device),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/e7.json"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/e7"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--loocv", action="store_true", help="Also evaluate leave-one-out on the development set.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    torch.set_num_threads(config["runtime"]["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = config["dataset"]
    data = load_specimens(args.data or dataset["workbook"], dataset)
    train, test = split_specimens(data, dataset["test_ids"])
    args.output.mkdir(parents=True, exist_ok=True)
    fitted = fit_model(train, config, device)
    save_bundle(fitted, args.output / "model.pt", {"train_ids": [s.sample_id for s in train]})
    evaluate(fitted, test, device).to_csv(args.output / "test_predictions.csv", index=False)

    if args.loocv:
        predictions = []
        for held_out in train:
            fold = fit_model([s for s in train if s is not held_out], config, device)
            predictions.append(evaluate(fold, [held_out], device))
        pd.concat(predictions, ignore_index=True).to_csv(args.output / "loocv_predictions.csv", index=False)


if __name__ == "__main__":
    main()
