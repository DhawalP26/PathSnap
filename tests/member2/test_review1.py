from types import SimpleNamespace

import numpy as np
import pandas as pd

from models.review1_features import VARIANTS, enrich
from models.schema import FEATURES
from models.score_candidates import score_review1
from models.train_models import make_models


def setup_trip():
    track = np.array([[0., 0., 0.], [5., 0., 1.], [10., 0., 2.]])
    nodes = np.array([[0., 0.], [10., 0.], [20., 0.]])
    arcs = np.array([[0, 1], [1, 2], [1, 0]])
    net = SimpleNamespace(project=lambda value: value[:,:2], arcs=arcs, nodes=nodes,
                          starts=nodes[arcs[:,0]], vectors=nodes[arcs[:,1]] - nodes[arcs[:,0]],
                          headings=np.array([90., 90., 270.]))
    frame = pd.DataFrame(dict(trip_id="00000000", gps_index=[0, 1, 1, 2],
                              arc_id=[0, 0, 2, 1], distance_m=[0., 1., 3., 0.],
                              heading_diff=[np.nan, 0., 180., 0.], speed_mps=[0., 5., 5., 5.],
                              time_gap_s=[0., 1., 1., 1.], gps_step_m=[0., 5., 5., 5.],
                              segment_length_m=10., candidate_rank=[1, 1, 2, 1],
                              segment_fraction=[0., .5, .5, 0.]))
    return SimpleNamespace(trip=SimpleNamespace(track=track), network=net, features=frame)


def test_context_obeys_group_and_direction():
    trip = setup_trip()
    got = enrich(trip)
    assert got.distance_excess_m.tolist() == [0., 0., 2., 0.]
    assert got.nearest_margin_m.tolist() == [0., 2., 2., 0.]
    assert got.endpoint_distance_m.tolist() == [0., 5., 5., 0.]
    assert np.isnan(got.previous_candidate_connection_m.iloc[0])
    assert got.next_candidate_connection_m.iloc[1] == 0
    assert np.isnan(got.next_candidate_connection_m.iloc[2])
    assert np.isnan(got.next_segment_distance_m.iloc[-1])
    assert got.center_heading_diff.iloc[2] == 180
    trip.features["label"] = [0, 1, 0, 1]
    pd.testing.assert_frame_equal(got, enrich(trip).drop(columns="label"))


def test_review1_inference_keeps_candidate_keys():
    trip = setup_trip()
    columns = FEATURES + VARIANTS["full_context"]
    model = make_models(42)["DecisionTree"].fit(enrich(trip)[columns], [1, 1, 0, 1])
    artifact = dict(variant="full_context", model=model, feature_columns=columns, positive_class=1)
    scored = score_review1(trip, artifact)
    pd.testing.assert_frame_equal(scored.drop(columns="score"), trip.features)
    assert scored.score.between(0, 1).all()
    trip.features = trip.features.iloc[:0]
    assert score_review1(trip, artifact).empty
