import numpy as np
import pandas as pd
import pytest

from preprocessing import FEATURE_COLUMNS, prepare_training_trip, prepare_trip
from preprocessing.data_loader import load_reference
from preprocessing.feature_engineering import motion
from preprocessing.pipeline import align_reference


ID = "00000005"


@pytest.fixture
def trip_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHSNAP_DATA_DIR", str(tmp_path))
    folder = tmp_path / "train" / ID
    folder.mkdir(parents=True)
    contents = {
        "nodes": "0 0\n0.001 0\n0.002 0\n0.001 0.001\n",
        "arcs": "0 1\n1 2\n1 0\n1 3\n",
        "track": "0.0018 0 3\n0.0002 0 1\n0.0007 0 2\n",
        "route": "0\n1\n",
    }
    for extension, source in contents.items():
        (folder / f"{ID}.{extension}").write_text(source, encoding="utf-8")
    return folder


def test_inference_has_no_route_dependency(trip_folder):
    path = trip_folder / f"{ID}.route"
    path.write_text("invalid and must never be parsed", encoding="utf-8")
    prepared = prepare_trip(ID, "train")
    path.unlink()
    without_reference = prepare_trip(ID, trip_dir=trip_folder)
    pd.testing.assert_frame_equal(prepared.features, without_reference.features)
    assert prepared.trip.track[:, 2].tolist() == [1, 2, 3]
    assert list(prepared.features.columns[3:]) == list(FEATURE_COLUMNS)
    assert prepared.features.query("gps_index == 1 and arc_id == 2").heading_diff.item() == 180


def test_supervised_labels_with_unequal_row_counts(trip_folder):
    labeled, diagnostics = prepare_training_trip(ID)
    assert diagnostics.aligned_arc_id.tolist() == [0, 0, 1]
    assert labeled.groupby("gps_index").label.sum().tolist() == [1, 1, 1]
    assert set(labeled.label) == {0, 1}
    inference = prepare_trip(ID, "train").features
    pd.testing.assert_frame_equal(labeled.drop(columns="label"), inference)


def test_candidate_miss_excluded_from_labels(trip_folder):
    # The reverse arc ties at zero distance, with lower ID than the reference.
    (trip_folder / f"{ID}.route").write_text("2\n", encoding="utf-8")
    labeled, diagnostics = prepare_training_trip(ID, radius_m=25, top_k=1)
    assert labeled.empty
    assert diagnostics.aligned_arc_id.tolist() == [2, 2, -1]
    assert not diagnostics.candidate_retrieved.any()


def test_queries_match_brute_force_and_find_long_arc_midpoints(trip_folder):
    prepared = prepare_trip(ID, "train")
    net = prepared.network
    points = net.project(np.array([[.0005, .00001], [.0017, .00014], [.01, .01]]))
    for point, (ids, distances, fractions) in zip(points, net.candidates(points, 20, 8)):
        expected, along = net.distances(point, np.arange(len(net.arcs)))
        order = np.lexsort((np.arange(len(net.arcs)), expected))
        order = order[expected[order] <= 20]
        np.testing.assert_array_equal(ids, order)
        np.testing.assert_allclose(distances, expected[order])
        np.testing.assert_allclose(fractions, along[order])
    assert prepared.features.iloc[0].distance_m < 0.01


def test_alignment_respects_repeated_route_occurrences(trip_folder):
    prepared = prepare_trip(ID, "train")
    track = np.array([[.0002, 0, 1], [.0008, 0, 2], [.0003, 0, 3],
                      [.0008, 0, 4], [.0018, 0, 5]])
    points = prepared.network.project(track)
    headings = motion(track, points)[0]
    matched = align_reference(prepared.network, points, headings, np.array([0, 2, 0, 1]))
    assert matched.tolist() == [0, 0, 2, 0, 1]


@pytest.mark.parametrize(("extension", "source", "message"), [
    ("track", "0 0 nan\n", r"track:1: non-finite"),
    ("nodes", "10 95\n", r"nodes:1: coordinate"),
    ("arcs", "0 999\n", "arc endpoint"),
])
def test_invalid_inputs_have_context(trip_folder, extension, source, message):
    (trip_folder / f"{ID}.{extension}").write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        prepare_trip(ID, "train")


def test_bad_reference_id_is_rejected(trip_folder):
    (trip_folder / f"{ID}.route").write_text("-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference arc"):
        load_reference(ID, "train", 4)


def test_missing_motion_values_and_empty_candidates(trip_folder):
    (trip_folder / f"{ID}.track").write_text("0.0003 0 2\n0.0005 0 2\n",
                                             encoding="utf-8")
    table = prepare_trip(ID, "train").features
    assert table.speed_mps.isna().all()
    assert table.heading_diff.isna().all()
    (trip_folder / f"{ID}.track").write_text("10 10 1\n", encoding="utf-8")
    assert prepare_trip(ID, "train").features.empty
