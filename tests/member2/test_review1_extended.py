import numpy as np
import pandas as pd

from models.review1_extended_features import EXTENDED_VARIANTS, extend
from models.schema import FEATURES
from models.score_candidates import score_review1
from models.train_models import make_models
from test_review1 import setup_trip


def test_lattice_prefers_directed_continuation():
    trip = setup_trip()
    trip.network.lengths = np.linalg.norm(trip.network.vectors, axis=1)
    frame = extend(trip)
    assert frame.lattice_candidate_rank.iloc[1] == 1
    assert frame.lattice_candidate_rank.iloc[2] > 1
    assert frame.outgoing_arc_support.iloc[2] == 0
    trip.features["label"] = [1, 1, 0, 1]
    pd.testing.assert_frame_equal(frame, extend(trip).drop(columns="label"))


def test_extended_scoring_preserves_decoder_input():
    trip = setup_trip()
    trip.network.lengths = np.linalg.norm(trip.network.vectors, axis=1)
    frame = extend(trip)
    variant = "review1_extended_lattice"
    columns = FEATURES + EXTENDED_VARIANTS[variant]
    model = make_models(42)["DecisionTree"].fit(frame[columns], [1, 1, 0, 1])
    scored = score_review1(trip, dict(variant=variant, model=model,
                                     feature_columns=columns, positive_class=1))
    pd.testing.assert_frame_equal(scored.drop(columns="score"), trip.features)
    assert scored.score.between(0, 1).all()
