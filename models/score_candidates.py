"""Probability scoring for route reconstruction and the API."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.schema import FEATURES, ID_COLUMNS, prepare_frame

DEFAULT_MODEL = Path(__file__).with_name("best_model.pkl")


def validate_artifact(artifact):
    if (not isinstance(artifact, dict) or not hasattr(artifact.get("model"), "classes_")
            or list(artifact.get("feature_columns", [])) != FEATURES
            or artifact.get("positive_class") != 1
            or set(artifact["model"].classes_) != {0, 1}):
        raise ValueError("Invalid model artifact or incompatible feature schema")
    return artifact


def load_model(model_path=DEFAULT_MODEL):
    """Load a trusted joblib artifact once for reuse across requests."""
    return validate_artifact(joblib.load(model_path))


def score_candidates(candidate_features: pd.DataFrame, model_path=DEFAULT_MODEL, *, artifact=None):
    """Return uncalibrated positive scores with the original index and candidate IDs."""
    model_input = prepare_frame(candidate_features)
    result = candidate_features.loc[:, ID_COLUMNS + FEATURES].copy()
    if result.empty:
        return result.assign(score=np.empty(0, dtype=float))
    saved = load_model(model_path) if artifact is None else validate_artifact(artifact)
    model = saved["model"]
    positive = list(model.classes_).index(saved["positive_class"])
    scores = model.predict_proba(model_input[FEATURES])[:, positive]
    if not (np.isfinite(scores) & (scores >= 0) & (scores <= 1)).all():
        raise ValueError("Expected finite candidate scores between zero and one")
    return result.assign(score=scores)
