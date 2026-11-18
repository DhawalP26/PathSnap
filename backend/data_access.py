"""Read-only access to trip data, without exposing .route reference labels."""
from __future__ import annotations

import os
from pathlib import Path
import re

TRIP_ID_PATTERN = re.compile(r"^[0-9]{8}$")


def dataset_root() -> Path:
    return Path(os.getenv("PATHSNAP_DATA_DIR", "data")).expanduser().resolve()


def partition_name() -> str:
    partition = os.getenv("PATHSNAP_PARTITION", "student_test")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", partition):
        raise ValueError("PATHSNAP_PARTITION must be a single safe folder name")
    return partition


def trip_folder(trip_id: str) -> Path:
    if not TRIP_ID_PATTERN.fullmatch(trip_id):
        raise ValueError("Trip ID must be exactly eight digits")
    return dataset_root() / partition_name() / trip_id


def has_required_inputs(folder: Path, trip_id: str) -> bool:
    return folder.is_dir() and all(
        (folder / f"{trip_id}.{ext}").is_file() for ext in ("track", "nodes", "arcs")
    )


def list_trip_ids() -> list[str]:
    base = dataset_root() / partition_name()
    if not base.is_dir():
        return []
    return sorted(
        path.name for path in base.iterdir()
        if path.is_dir() and TRIP_ID_PATTERN.fullmatch(path.name)
        and has_required_inputs(path, path.name)
    )


def read_gps_preview(trip_id: str, max_points: int = 450) -> dict:
    folder = trip_folder(trip_id)
    if not has_required_inputs(folder, trip_id):
        raise FileNotFoundError(f"Trip {trip_id} is not available")
    points = []
    with (folder / f"{trip_id}.track").open(encoding="utf-8-sig") as source:
        for line in source:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"Invalid .track row for trip {trip_id}")
            lon, lat, timestamp = (float(p) for p in parts)
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f"GPS coordinate outside valid range for {trip_id}")
            points.append([lat, lon, timestamp])
    if not points:
        raise ValueError(f"Trip {trip_id} has no GPS observations")
    n = len(points)
    if max_points < 2:
        raise ValueError("max_points must be >= 2")
    if n > max_points:
        indices = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
        points = [points[i] for i in indices]
    return {"trip_id": trip_id, "total_observations": n, "gps_points": points}
