import joblib
import numpy as np
import pandas as pd
import pytest

from models.bayesian_tuning import optimize
from models.schema import FEATURES, prepare_frame
from models.score_candidates import load_model, score_candidates
from models.train_models import cross_validate, folds_for, make_models, metrics_for, partitions


@pytest.fixture
def candidates():
    rows = []
    for trip in range(8):
        for gps_index in range(6):
            for arc_id in range(2):
                rows.append(dict(trip_id=f"{trip:08d}", gps_index=gps_index, arc_id=arc_id,
                                 label=arc_id, **{key: float(arc_id + gps_index / 100)
                                                  for key in FEATURES}))
    return pd.DataFrame(rows)


def test_schema(candidates):
    candidates.loc[0, "heading_diff"] = np.inf
    candidates.loc[1, "speed_mps"] = np.nan
    result = prepare_frame(candidates, True)
    assert result.trip_id.iloc[0] == "00000000"
    assert result.loc[0, "heading_diff"] != result.loc[0, "heading_diff"]
    assert np.isnan(result.loc[1, "speed_mps"])
    with pytest.raises(ValueError, match="segment_fraction"):
        prepare_frame(candidates.drop(columns="segment_fraction"), True)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_frame(pd.concat([candidates, candidates.iloc[:1]]), True)


@pytest.mark.parametrize("label", [None, 2, "bad"])
def test_binary_label_validation(candidates, label):
    candidates["label"] = candidates.label.astype(object)
    candidates.loc[0, "label"] = label
    with pytest.raises(ValueError, match="binary"):
        prepare_frame(candidates, True)


def test_trip_splits_and_models(candidates):
    dev, held = partitions(candidates, 42)
    assert set(dev.trip_id).isdisjoint(held.trip_id)
    splits = folds_for(dev, 3)
    for train, validation in splits:
        assert set(dev.trip_id.iloc[train]).isdisjoint(dev.trip_id.iloc[validation])
    models = make_models(42)
    assert len(models) == 5
    for model in models.values():
        result, folds = cross_validate(model, dev, splits)
        assert 0 <= result["f1"] <= 1 and len(folds) == 3
    baseline = models["LogisticRegression"]
    assert cross_validate(baseline, dev, splits)[0] == cross_validate(baseline, dev, splits)[0]


def test_top_one_includes_all_negative_groups(candidates):
    frame = candidates.iloc[:4].copy()
    frame.loc[3, "label"] = 0
    assert metrics_for(frame, np.array([.1, .9, .3, .7]))["top1_accuracy"] == .5


def test_tuning_export_and_inference(candidates, tmp_path):
    dev, held = partitions(candidates, 42)
    name = "DecisionTree"
    baseline = cross_validate(make_models(42)[name], dev, folds_for(dev, 2))[0]["f1"]
    study, _, variant, fitted, comparison = optimize(dev, held, name, 2, 42, 1, baseline)
    assert len(study.trials) == 1 and len(comparison) == 2
    assert variant in {"tuned", "untuned"}
    fitted.fit(dev[FEATURES], dev.label)
    path = tmp_path / "model.pkl"
    joblib.dump(dict(model=fitted, feature_columns=FEATURES, positive_class=1), path)
    incoming = held.iloc[::-1].assign(reference_route="never expose")
    scored = score_candidates(incoming, artifact=load_model(path))
    pd.testing.assert_frame_equal(scored[["trip_id", "gps_index", "arc_id"]],
                                  incoming[["trip_id", "gps_index", "arc_id"]])
    assert scored.columns.tolist() == ["trip_id", "gps_index", "arc_id", *FEATURES, "score"]
    assert scored.score.between(0, 1).all()
    assert score_candidates(incoming.iloc[:0], model_path="missing").empty


def test_preprocessing_contract():
    assert tuple(FEATURES) == ("distance_m", "heading_diff", "speed_mps", "time_gap_s",
                               "gps_step_m", "segment_length_m", "candidate_rank", "segment_fraction")
    try:
        from preprocessing.feature_engineering import FEATURE_COLUMNS
    except ModuleNotFoundError:
        return
    assert tuple(FEATURES) == FEATURE_COLUMNS
