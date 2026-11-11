"""Specimen loading, training-only normalization, and graph construction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected


@dataclass
class Specimen:
    sample_id: str
    frame: pd.DataFrame
    wide: dict
    strength: float | None


@dataclass
class GraphSpecimen:
    sample_id: str
    graph: Data
    wide: torch.Tensor
    strength: float | None


def calculate_wide_features(frame, config):
    roi = frame.loc[frame[config["coordinates"][0]].between(
        config["roi_x_min"], config["roi_x_max"])]
    if roi.empty:
        raise ValueError("No pores fall inside the configured ROI.")
    diameters = roi["直径 [µm]"].to_numpy()
    return {
        "ROI_Rmax": float(roi["直径 [µm]"].max()),
        "ROI_Mean_Diameter": float(roi["直径 [µm]"].mean()),
        "ROI_Vol_Fraction": float(np.sum((4 / 3) * np.pi * (diameters / 2) ** 3)),
        "ROI_Mean_Sphericity": float(roi["球度"].mean()),
    }


def load_specimens(workbook, config, sample_ids=None, require_targets=True):
    path = Path(workbook)
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found: {path}. Pass --data PATH.")
    requested = [str(x) for x in (config["sample_ids"] if sample_ids is None else sample_ids)]
    excluded = {str(x) for x in config.get("excluded_ids", [])}
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Requested specimen IDs must be nonempty and unique.")
    if excluded.intersection(requested):
        raise ValueError(f"Excluded specimen requested: {excluded.intersection(requested)}")
    specimens = []
    with pd.ExcelFile(path, engine="openpyxl") as book:
        missing = set(requested) - set(book.sheet_names)
        if missing:
            raise ValueError(f"Workbook lacks sheets: {sorted(missing)}")
        # Worksheet order determines the sequence of specimens and folds.
        for sample_id in book.sheet_names:
            if sample_id not in requested:
                continue
            frame = book.parse(sample_id)
            if frame.empty:
                raise ValueError(f"Specimen {sample_id}: empty worksheet.")
            required = set(config["coordinates"]) | {
                "直径 [µm]", "球度", "Angle_with_Load", "投影尺寸 x [µm]",
                "投影尺寸 y [µm]", "投影尺寸 z [µm]",
            }
            missing_columns = required - set(frame.columns)
            if missing_columns:
                raise ValueError(f"Specimen {sample_id}: missing columns {sorted(missing_columns)}")
            target = config["target_column"]
            strength = None
            if target in frame and pd.notna(frame[target].iloc[0]):
                strength = float(frame[target].iloc[0])
                if not np.isfinite(strength):
                    raise ValueError(f"Specimen {sample_id}: nonfinite strength.")
            if require_targets and strength is None:
                raise ValueError(f"Specimen {sample_id}: missing strength in first row.")
            specimens.append(Specimen(sample_id, frame, calculate_wide_features(frame, config), strength))
    return specimens


def split_specimens(specimens, test_ids):
    test_ids = {str(x) for x in test_ids}
    present = {s.sample_id for s in specimens}
    if not test_ids.issubset(present):
        raise ValueError("Configured test specimens are missing.")
    train = [s for s in specimens if s.sample_id not in test_ids]
    test = [s for s in specimens if s.sample_id in test_ids]
    return train, test


class PoreDataProcessor:
    def __init__(self, config):
        self.config = config
        self.node_scaler = StandardScaler()
        self.coord_scaler = MinMaxScaler((0, 1))
        self.wide_scaler = StandardScaler()
        self.final_features = config["node_features"] + ["Griffith_Factor"]
        self.fitted_ids = []

    def preprocess(self, frame):
        cfg = self.config
        frame = frame.loc[frame[cfg["coordinates"][0]].between(
            cfg["roi_x_min"], cfg["roi_x_max"])].copy()
        if frame.empty:
            raise ValueError("No pores fall inside the configured ROI.")
        for column in cfg["node_features"] + cfg["coordinates"]:
            if column not in frame:
                frame[column] = 0.0
        frame["Ratio_XY"] = frame["投影尺寸 x [µm]"] / frame["投影尺寸 y [µm]"].replace(0, 1e-6)
        frame["Griffith_Factor"] = 1 / np.sqrt(frame["直径 [µm]"].replace(0, 1e-6).abs())
        values = frame[self.final_features + cfg["coordinates"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("ROI features and coordinates must be finite numeric values.")
        return frame

    def fit(self, specimens):
        if not specimens:
            raise ValueError("Cannot fit a processor without training specimens.")
        frames = [self.preprocess(s.frame) for s in specimens]
        self.node_scaler.fit(pd.concat([f[self.final_features] for f in frames]))
        self.coord_scaler.fit(pd.concat([f[self.config["coordinates"]] for f in frames]))
        self.wide_scaler.fit(pd.DataFrame([s.wide for s in specimens])[self.config["wide_features"]])
        self.fitted_ids = [s.sample_id for s in specimens]
        return self

    def transform(self, specimen, graph_config):
        frame = self.preprocess(specimen.frame)
        x = torch.tensor(self.node_scaler.transform(frame[self.final_features]), dtype=torch.float)
        pos = torch.tensor(self.coord_scaler.transform(frame[self.config["coordinates"]]), dtype=torch.float)
        wide_frame = pd.DataFrame([specimen.wide])[self.config["wide_features"]]
        wide = torch.tensor(self.wide_scaler.transform(wide_frame), dtype=torch.float).squeeze(0)
        edge_index = to_undirected(radius_graph(
            pos, r=graph_config["radius"], loop=False,
            max_num_neighbors=graph_config["max_neighbors"],
        ))
        return GraphSpecimen(specimen.sample_id, Data(x=x, pos=pos, edge_index=edge_index),
                             wide, specimen.strength)

    def state_dict(self):
        def encode(scaler):
            state = {}
            for key in ("mean_", "scale_", "var_", "min_", "data_min_", "data_max_",
                        "data_range_", "n_samples_seen_", "n_features_in_", "feature_names_in_"):
                if hasattr(scaler, key):
                    value = getattr(scaler, key)
                    state[key] = value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
            return state
        return {"config": self.config, "fitted_ids": self.fitted_ids,
                "node_scaler": encode(self.node_scaler), "coord_scaler": encode(self.coord_scaler),
                "wide_scaler": encode(self.wide_scaler)}

    @classmethod
    def from_state_dict(cls, state):
        processor = cls(state["config"])
        processor.fitted_ids = list(state["fitted_ids"])
        for name in ("node_scaler", "coord_scaler", "wide_scaler"):
            scaler = getattr(processor, name)
            for key, value in state[name].items():
                if isinstance(value, list):
                    value = np.asarray(value, dtype=object if key == "feature_names_in_" else None)
                setattr(scaler, key, value)
        return processor
