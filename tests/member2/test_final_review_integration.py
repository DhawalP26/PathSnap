import joblib
import numpy as np

from models.final_review_integration import create_predict_trip, load_final_model
from models.final_review_tuning import COLUMNS, VARIANT
from models.review1_extended_features import extend
from models.train_models import make_models
from test_review1 import setup_trip


def test_final_model_reloads_and_processes_multiple_reference_free_trips(tmp_path):
    from preprocessing import prepare_trip

    trip = setup_trip()
    trip.network.lengths = np.linalg.norm(trip.network.vectors, axis=1)
    frame = extend(trip)
    model = make_models(42)["LogisticRegression"].fit(frame[COLUMNS], [1, 1, 0, 1])
    model_path = tmp_path / "final_model.pkl"
    joblib.dump(dict(model=model, variant=VARIANT, feature_columns=COLUMNS, positive_class=1), model_path)
    assert load_final_model(model_path)["feature_columns"] == COLUMNS
    predict = create_predict_trip(model_path)
    for trip_id in ["00000001", "00000002"]:
        folder = tmp_path / trip_id
        folder.mkdir()
        (folder / f"{trip_id}.nodes").write_text("20 50\n20.0001 50\n20.0002 50\n")
        (folder / f"{trip_id}.arcs").write_text("0 1\n1 2\n1 0\n")
        (folder / f"{trip_id}.track").write_text("20 50 1\n20.0001 50 2\n20.0002 50 3\n")
        out = predict(folder, trip_id)
        assert len(out["predicted_arc_ids"]) == 3
        assert len(out["route_coordinates"]) >= 2
        assert out["metrics"]["candidate_rows"] == len(prepare_trip(trip_id, trip_dir=folder).features)
        assert not list(folder.glob("*.route"))


def test_empty_candidates_and_missing_samples_are_supported(tmp_path):
    trip = setup_trip()
    trip.network.lengths = np.linalg.norm(trip.network.vectors, axis=1)
    model = make_models(42)["DecisionTree"].fit(extend(trip)[COLUMNS], [1, 1, 0, 1])
    artifact = dict(model=model, variant=VARIANT, feature_columns=COLUMNS, positive_class=1)
    from models.score_candidates import score_review1
    trip.features = trip.features[trip.features.gps_index.ne(1)].copy()
    trip.features.loc[:, "heading_diff"] = np.inf
    scored = score_review1(trip, artifact)
    assert scored.gps_index.tolist() == [0, 2]
    assert np.isfinite(scored.score).all()
    trip.features = trip.features.iloc[:0]
    from models.final_review_integration import predict_prepared
    result, route = predict_prepared(trip, None, artifact)
    assert result.empty and route is None
