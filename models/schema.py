"""Shared candidate schema and input normalization."""
import numpy as np
import pandas as pd

try:
    from preprocessing.feature_engineering import FEATURE_COLUMNS
except ModuleNotFoundError as error:
    if error.name not in {"preprocessing", "preprocessing.feature_engineering"}:
        raise
    FEATURE_COLUMNS = (
        "distance_m", "heading_diff", "speed_mps", "time_gap_s", "gps_step_m",
        "segment_length_m", "candidate_rank", "segment_fraction",
    )

FEATURES = list(FEATURE_COLUMNS)
ID_COLUMNS = ["trip_id", "gps_index", "arc_id"]
DEFAULT_TRAIN = "results/member1/train_features.csv"
DEFAULT_OUT = "results/member2"


def prepare_frame(frame, require_label=False):
    required = ID_COLUMNS + FEATURES + (["label"] if require_label else [])
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame.loc[:, required].copy()
    ids = result["trip_id"].astype("string")
    if not ids.str.fullmatch(r"[0-9]{8}").fillna(False).all():
        raise ValueError("trip_id must contain eight digits")
    result["trip_id"] = ids
    for name in ID_COLUMNS[1:]:
        values = pd.to_numeric(result[name], errors="coerce")
        if not (np.isfinite(values) & values.ge(0) & values.mod(1).eq(0)).all():
            raise ValueError(f"{name} must contain nonnegative integers")
        result[name] = values.astype("int64")
    if result.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate candidate keys")
    for name in FEATURES:
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
    result[FEATURES] = result[FEATURES].replace([np.inf, -np.inf], np.nan)
    if require_label:
        labels = pd.to_numeric(result["label"], errors="coerce")
        if not labels.isin([0, 1]).all() or set(labels) != {0, 1}:
            raise ValueError("label must contain both binary classes and no missing values")
        result["label"] = labels.astype("int8")
        if result.groupby(ID_COLUMNS[:2])["label"].sum().gt(1).any():
            raise ValueError("An observation has multiple positive candidates")
    return result


def read_training(path):
    return prepare_frame(pd.read_csv(path, dtype={"trip_id": "string"}), True)

