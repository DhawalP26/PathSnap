"""INTEGRATION POINT: replace with your teammates' genuine ML pipeline.

Never use .route to choose predicted arcs. Ground truth must remain server-side
and must be accessed *only after* prediction, for post-hoc evaluation.
"""
from __future__ import annotations
from pathlib import Path


def predict_trip(trip_dir: Path, trip_id: str) -> dict:
    """Expected output from Members 1-3 (all coordinates in lat, lon order).

    {
      "trip_id": "00000000",
      "predicted_arc_ids": [10, 11, 12],
      "route_coordinates": [[56.2, 54.9], [56.21, 54.91]],
      "metrics": {"route_continuity": 0.95},
    }

    metrics may include accuracy calculated on the server AFTER decoding.
    Do not include reference arc IDs or reference geometry in the response.
    """
    raise NotImplementedError(
        "Model pipeline not integrated. Ask Members 1-3 for their feature, "
        "score_candidates and reconstruct_route functions, then connect them "
        "inside backend/pipeline_adapter.py. No fake prediction is returned."
    )
