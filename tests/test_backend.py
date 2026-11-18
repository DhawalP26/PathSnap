"""Minimal tests of API surface without an ML pipeline or data leak."""
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app


def _data(tmp_path: Path):
    trip = tmp_path / "student_test" / "00000000"
    trip.mkdir(parents=True)
    (trip / "00000000.track").write_text("54.1 56.2 1\n54.2 56.3 2\n")
    (trip / "00000000.nodes").write_text("54.1 56.2\n54.2 56.3\n")
    (trip / "00000000.arcs").write_text("0 1\n")
    # This reference route is intentionally never returned by the API.
    (trip / "00000000.route").write_text("7\n")


def test_trip_listing_preview_and_no_ground_truth(tmp_path, monkeypatch):
    _data(tmp_path)
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_PARTITION", "student_test")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    r = client.get("/trips")
    assert r.status_code == 200
    assert r.json()["trips"] == ["00000000"]
    preview = client.get("/preview/00000000")
    assert preview.status_code == 200
    assert preview.json()["gps_points"][0] == [56.2, 54.1, 1.0]
    assert "route" not in str(preview.json()).lower()


def test_predict_not_fake(tmp_path, monkeypatch):
    _data(tmp_path)
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_PARTITION", "student_test")
    client = TestClient(app)
    response = client.post("/predict", json={"trip_id": "00000000"})
    assert response.status_code == 503
    assert "not integrated" in response.json()["detail"].lower()
    assert client.post("/predict", json={"trip_id": "../etc"}).status_code == 422
