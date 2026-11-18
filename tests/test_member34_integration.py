"""Member 3/4 contract tests; run from repository root after overlaying files."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import backend.main as api
import backend.pipeline_adapter as adapter
from frontend.route_utils import route_lines


def make_trip(tmp_path: Path):
    folder = tmp_path / "student_test" / "00000000"
    folder.mkdir(parents=True)
    (folder / "00000000.track").write_text("54.1 56.2 1\n54.2 56.3 2\n", encoding="utf8")
    (folder / "00000000.nodes").write_text("54.1 56.2\n54.2 56.3\n", encoding="utf8")
    (folder / "00000000.arcs").write_text("0 1\n", encoding="utf8")
    (folder / "00000000.route").write_text("999\n", encoding="utf8")
    return folder


def test_data_health_preview_and_missing_model(tmp_path, monkeypatch):
    make_trip(tmp_path)
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_PARTITION", "student_test")
    monkeypatch.setenv("PATHSNAP_MODEL_PATH", str(tmp_path / "not-installed.pkl"))
    client = TestClient(api.app)
    health = client.get("/health")
    assert health.status_code == 200
    assert not health.json()["model_ready"]
    assert client.get("/trips").json()["trips"] == ["00000000"]
    preview = client.get("/preview/00000000")
    assert preview.json()["gps_points"][0] == [56.2, 54.1, 1.0]
    assert "route" not in str(preview.json()).lower()
    missing = client.post("/predict", json={"trip_id": "00000000"})
    assert missing.status_code == 503
    assert "model" in missing.json()["detail"].lower()
    assert client.post("/predict", json={"trip_id": "../../etc"}).status_code == 422
    assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404


def test_live_prediction_response_contract_and_no_ground_truth(tmp_path, monkeypatch):
    make_trip(tmp_path)
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_PARTITION", "student_test")
    payload = {
        "trip_id": "00000000",
        "predicted_arc_ids": [0, 1],
        "route_coordinates": [
            [[56.20, 54.10], [56.21, 54.11]],
            [[56.30, 54.20], [56.31, 54.21]],
        ],
        "route_breaks": 1,
        "metrics": {"candidate_rows": 16},
    }
    monkeypatch.setattr(api, "predict_trip", lambda folder, trip_id: payload.copy())
    response = TestClient(api.app).post("/predict", json={"trip_id": "00000000"})
    assert response.status_code == 200
    body = response.json()
    assert body["route_breaks"] == 1
    assert len(body["route_coordinates"]) == 2
    assert body["duration_ms"] >= 0
    assert body["predicted_arc_ids"] == [0, 1]
    assert "ground_truth" not in str(body).lower()
    assert "reference" not in str(body).lower()
    assert "999" not in str(body)


def test_route_lines_do_not_bridge_decoder_restarts():
    geometry = {
        "route_coordinates": [
            [[56.2, 54.1], [56.21, 54.11]],
            [[56.4, 54.3], [56.41, 54.31]],
        ]
    }
    lines = route_lines(geometry)
    assert len(lines) == 2
    assert lines[0][-1] != lines[1][0]
    assert lines[0][0] == (56.2, 54.1)
    assert lines[1][0] == (56.4, 54.3)


@pytest.mark.parametrize("invalid", [
    [[56.2, 54.1], [56.3, 54.2]],  # Flat legacy polyline: would join decoder breaks.
    [[[float("nan"), 54.1]]],
    [[[91.0, 54.1]]],
    [["not-a-coordinate", 54.1]],
    "invalid",
])
def test_invalid_geometry_is_rejected(invalid):
    with pytest.raises(ValueError):
        route_lines({"route_coordinates": invalid})


def test_adapter_calls_real_contracts_with_40_feature_model_path(tmp_path, monkeypatch):
    """Contract-isolated test; final run must also use a genuine model artifact."""
    folder = make_trip(tmp_path)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"stub -- model loader is mocked")
    monkeypatch.setenv("PATHSNAP_MODEL_PATH", str(model_path))

    token = object()
    monkeypatch.setattr(adapter, "_load_artifact", lambda path, modified_ns: token)
    prepared = SimpleNamespace(
        trip=SimpleNamespace(track=[(54.1, 56.2, 1), (54.2, 56.3, 2)]),
        features=[0, 1, 2, 3],
    )
    calls = {}

    def fake_prepare(trip_id, **kwargs):
        calls["prepared"] = (trip_id, kwargs)
        return prepared

    def fake_scorer(*args, **kwargs):
        raise AssertionError("Only the reconstruction entry point may invoke the scorer")

    class FakePrediction:
        reconstruction = SimpleNamespace(
            observation_indices=[0, 1], route_segments=[(3, 4), (6,)],
            route_arc_ids=[3, 4, 6]
        )

        def api_response(self):
            return {"trip_id": "00000000", "predicted_arc_ids": [3, 4, 6],
                    "route_coordinates": [[[56.2, 54.1], [56.21, 54.11]],
                                          [[56.3, 54.2], [56.31, 54.21]]],
                    "route_breaks": 1}

    def fake_reconstruct(p, scorer, *, artifact):
        assert p is prepared
        assert scorer is fake_scorer
        assert artifact is token
        calls["reconstructed"] = True
        return FakePrediction()

    mock_preprocess = ModuleType("preprocessing")
    mock_preprocess.prepare_trip = fake_prepare
    mock_models = ModuleType("models")
    mock_models.__path__ = []
    mock_score = ModuleType("models.score_candidates")
    mock_score.score_review1 = fake_scorer
    mock_reconstruction = ModuleType("reconstruction")
    mock_reconstruction.__path__ = []
    mock_route = ModuleType("reconstruction.route_reconstruction")
    mock_route.reconstruct_prepared_trip = fake_reconstruct
    for name, module in (
        ("preprocessing", mock_preprocess), ("models", mock_models),
        ("models.score_candidates", mock_score),
        ("reconstruction", mock_reconstruction),
        ("reconstruction.route_reconstruction", mock_route),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    result = adapter.predict_trip(folder, "00000000")
    assert calls["reconstructed"]
    assert calls["prepared"][1]["radius_m"] == 75
    assert calls["prepared"][1]["top_k"] == 8
    assert result["metrics"]["gps_observations"] == 2
    assert result["metrics"]["candidate_rows"] == 4
    assert result["metrics"]["candidate_coverage"] == 1.0
    assert result["route_breaks"] == 1


def test_member3_real_prepared_trip_decoder_contract(tmp_path):
    """Run with the merged team repository; skips in a standalone overlay ZIP."""
    pytest.importorskip("reconstruction.route_reconstruction")
    pytest.importorskip("preprocessing")
    from preprocessing import prepare_trip
    from reconstruction.route_reconstruction import reconstruct_prepared_trip

    trip_id = "00000000"
    folder = tmp_path / trip_id
    folder.mkdir()
    (folder / f"{trip_id}.nodes").write_text(
        "0.0 0.0\n0.0001 0.0\n0.0002 0.0\n", encoding="utf8"
    )
    (folder / f"{trip_id}.arcs").write_text("0 1\n1 2\n", encoding="utf8")
    (folder / f"{trip_id}.track").write_text(
        "0.00002 0.0 10\n0.0001 0.0 20\n0.00019 0.0 30\n", encoding="utf8"
    )
    # Deliberately absent .route. Inference must not require reference labels.
    prepared = prepare_trip(trip_id, trip_dir=folder)

    def fake_scorer(prepared_trip, *, artifact=None):
        frame = prepared_trip.features.copy()
        assert not frame.empty
        frame["score"] = 0.8
        return frame

    prediction = reconstruct_prepared_trip(prepared, fake_scorer)
    result = prediction.api_response()
    assert result["trip_id"] == trip_id
    assert result["predicted_arc_ids"]
    assert result["route_coordinates"]
    assert isinstance(result["route_coordinates"][0][0], list)
    assert "reference" not in result
    assert "accuracy" not in result
