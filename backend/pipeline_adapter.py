"""Real, reference-free Member 1 -> Member 2 -> Member 3 inference adapter.

Run from the PathSnap repository root. Only load model files you trust: joblib
artifacts are Python pickle files and must never be accepted from HTTP users.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path


DEFAULT_MODEL_PATH = Path("results/member2/final_review/final_model.pkl")


class ModelNotReadyError(RuntimeError):
    """A validated 40-feature model artifact is not installed on the server."""


def get_model_path() -> Path:
    return Path(os.environ.get("PATHSNAP_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()


def model_ready() -> bool:
    return get_model_path().is_file()


@lru_cache(maxsize=3)
def _load_artifact(path: str, modified_ns: int):
    # Lazy import permits /health, /trips and /preview to work without ML deps.
    from models.final_review_integration import load_final_model

    return load_final_model(path)


def predict_trip(trip_dir: Path, trip_id: str) -> dict:
    """Infer using only .track/.nodes/.arcs and retain disconnected route lines.

    Response route_coordinates is a list of polylines; each polyline contains
    [latitude, longitude] points. Never bridge Viterbi restart boundaries.
    """
    path = get_model_path()
    if not path.is_file():
        raise ModelNotReadyError(
            "The 40-feature production model is missing. Set PATHSNAP_MODEL_PATH "
            "to Member 2's trusted final_model.pkl; do not use an 8-feature artifact."
        )

    from preprocessing import prepare_trip
    from models.score_candidates import score_review1
    from reconstruction.route_reconstruction import reconstruct_prepared_trip

    prepared = prepare_trip(trip_id, trip_dir=Path(trip_dir), radius_m=75, top_k=8)
    artifact = _load_artifact(str(path), path.stat().st_mtime_ns)
    prediction = reconstruct_prepared_trip(
        prepared,
        score_review1,
        artifact=artifact,
    )
    result = prediction.api_response()
    reconstruction = prediction.reconstruction
    observations = len(prepared.trip.track)
    matched = len(reconstruction.observation_indices) if reconstruction is not None else 0
    result["metrics"] = {
        "gps_observations": observations,
        "candidate_rows": len(prepared.features),
        "observations_with_candidates": matched,
        "candidate_coverage": round(matched / observations, 6) if observations else 0.0,
        "route_segments": len(reconstruction.route_segments) if reconstruction else 0,
        "reconstructed_arc_count": len(reconstruction.route_arc_ids) if reconstruction else 0,
    }
    return result
