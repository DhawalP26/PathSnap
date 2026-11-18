import numpy as np
import optuna
import pandas as pd

from models.final_review_tuning import COLUMNS, FAMILIES, evaluate, measurements, suggest
from models.review1_extended_features import EXTENDED_VARIANTS
from models.schema import FEATURES
from models.train_models import make_models


def test_frozen_feature_schema():
    assert COLUMNS == FEATURES + EXTENDED_VARIANTS["review1_extended_lattice"]
    assert len(COLUMNS) == len(set(COLUMNS)) == 40


def test_candidate_objective_ties_and_misses():
    frame = pd.DataFrame(dict(trip_id=["a", "a", "b", "b"],
                              gps_index=[0, 0, 0, 0], arc_id=[2, 1, 1, 2],
                              label=[0, 1, 0, 1]))
    m, _ = measurements(frame, np.array([.5, .5, .7, .2]), pd.Series({"a": 2, "b": 2}))
    assert m["top1_accuracy"] == .5
    assert m["all_gps_top1"] == .25
    assert m["mean_trip_top1"] == .5


def test_optuna_configurations_fit_grouped_folds():
    # Separate trips, two candidate arcs per observation and a fixed predictive feature.
    ids = np.repeat(list("abcd"), 4)
    labels = np.tile([0, 1], 8)
    frame = pd.DataFrame({name: labels.astype(float) for name in COLUMNS})
    frame.loc[0, COLUMNS[1]] = np.nan
    frame["trip_id"] = ids
    frame["gps_index"] = np.tile([0, 0, 1, 1], 4)
    frame["arc_id"] = np.tile([1, 2], 8)
    frame["label"] = labels
    left, right = np.arange(8), np.arange(8, 16)
    assert not set(frame.trip_id.iloc[left]) & set(frame.trip_id.iloc[right])
    counts = pd.Series(2, index=list("abcd"))
    for name in FAMILIES:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42))
        trial = study.ask()
        params = suggest(trial, name)
        model = make_models(42)[name].set_params(**{f"model__{k}": v for k, v in params.items()})
        result = evaluate(model, frame, [(left, right), (right, left)], counts, trial)
        study.tell(trial, result["metrics"]["top1_accuracy"])
        assert study.best_value == result["metrics"]["top1_accuracy"]
        assert len(result["folds"]) == 2 and len(result["trips"]) == 4
        assert model.named_steps["prep"].named_steps["imputer"].__dict__.get("statistics_") is None
