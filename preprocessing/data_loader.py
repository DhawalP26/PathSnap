"""Read and validate the PathSnap whitespace-delimited trip format."""
from dataclasses import dataclass
import os
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray


def _rows(path, columns, dtype):
    values = []
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != columns:
                raise ValueError(f"{path}:{number}: expected {columns} fields")
            try:
                row = tuple(dtype(token) for token in tokens)
                if not all(np.isfinite(float(value)) for value in row):
                    raise ValueError("non-finite value")
                if dtype is int and any(abs(value) > np.iinfo(np.int64).max for value in row):
                    raise ValueError("integer outside int64 range")
                if dtype is float and not (-180 <= row[0] <= 180 and -90 <= row[1] <= 90):
                    raise ValueError("coordinate outside WGS84 bounds")
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            values.append(row)
    if not values:
        raise ValueError(f"{path}: no data rows")
    return np.asarray(values, dtype=dtype)


def partition_path(partition):
    if not re.fullmatch(r"[a-zA-Z0-9_]+", partition):
        raise ValueError("partition must be a single folder name")
    root = os.environ.get("PATHSNAP_DATA_DIR")
    if not root:
        raise ValueError("Set PATHSNAP_DATA_DIR to the dataset directory")
    path = Path(root).expanduser() / partition
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def trip_ids(partition):
    return sorted(path.name for path in partition_path(partition).iterdir()
                  if path.is_dir() and re.fullmatch(r"[0-9]{8}", path.name))


def load_trip(trip_id, partition="student_test", *, trip_dir=None):
    """Read track, nodes and arcs. Node and directed arc IDs are zero-based row indices."""
    if not re.fullmatch(r"[0-9]{8}", trip_id):
        raise ValueError("trip_id must have exactly eight digits")
    folder = Path(trip_dir) if trip_dir is not None else partition_path(partition) / trip_id
    track = _rows(folder / f"{trip_id}.track", 3, float)
    nodes = _rows(folder / f"{trip_id}.nodes", 2, float)
    arcs = _rows(folder / f"{trip_id}.arcs", 2, int)
    if np.any((arcs < 0) | (arcs >= len(nodes))):
        raise ValueError(f"{folder}: arc endpoint outside node table")
    return Trip(trip_id, track[np.argsort(track[:, 2], kind="stable")], nodes, arcs)


def load_reference(trip_id, partition, arc_count):
    """Read ordered directed arc IDs for training or offline evaluation."""
    if not re.fullmatch(r"[0-9]{8}", trip_id):
        raise ValueError("trip_id must have exactly eight digits")
    path = partition_path(partition) / trip_id / f"{trip_id}.route"
    route = _rows(path, 1, int).ravel()
    if np.any((route < 0) | (route >= arc_count)):
        raise ValueError(f"{path}: reference arc outside arc table")
    return route
