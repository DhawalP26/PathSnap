"""Opt-in Member 2 model loader and backend adapter factory."""
from pathlib import Path

import joblib
import numpy as np

from models.review1_extended_features import EXTENDED_VARIANTS
from models.schema import FEATURES
from models.score_candidates import score_review1

VARIANT = "review1_extended_lattice"
COLUMNS = FEATURES + EXTENDED_VARIANTS[VARIANT]


def load_final_model(model_path):
    """Load a trusted local model once per worker and verify the frozen contract."""
    artifact = joblib.load(model_path)
    if (artifact["feature_columns"] != COLUMNS or artifact["variant"] != VARIANT
            or list(artifact["model"].feature_names_in_) != COLUMNS or artifact["positive_class"] != 1):
        raise ValueError("Invalid final-review model schema")
    return artifact


def predict_prepared(prepared, graph, artifact):
    """Return the scorer and decoder results without accessing reference routes."""
    from reconstruction.route_reconstruction import reconstruct_route

    scored = score_review1(prepared, artifact)
    if scored.empty:
        return scored, None
    # A wholly zero probability group still has the same candidate ordering.
    scores = scored.groupby("gps_index", sort=False).score.transform("max")
    scored.loc[scores.eq(0), "score"] = np.finfo(float).tiny
    return scored, reconstruct_route(graph, scored, prepared.trip.track)


def create_predict_trip(model_path):
    """Build Member 4's predict_trip(folder, trip_id) callback."""
    from preprocessing import prepare_trip
    from reconstruction.road_graph import RoadGraph

    artifact = load_final_model(model_path)

    def predict_trip(folder, trip_id):
        folder = Path(folder)
        prepared = prepare_trip(trip_id, trip_dir=folder)
        graph = RoadGraph.from_files(folder / f"{trip_id}.nodes", folder / f"{trip_id}.arcs")
        scored, route = predict_prepared(prepared, graph, artifact)
        if route is None:
            return dict(predicted_arc_ids=[], route_coordinates=[],
                        metrics={"candidate_rows": 0})
        arcs = np.asarray(route.route_arc_ids, dtype=int)
        coordinates = prepared.network.nodes[prepared.network.arcs[arcs]].reshape(-1, 2)[:, ::-1]
        return dict(predicted_arc_ids=list(route.predicted_arc_ids),
                    route_coordinates=coordinates.tolist(),
                    metrics={"candidate_rows": len(scored),
                             "observations_with_candidates": len(route.observation_indices),
                             "route_segments": len(route.route_segments)})

    return predict_trip
