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


def score_review1(prepared, artifact):
    """Score an entire prepared trip with a frozen Review 1 artifact."""
    from models.review1_features import VARIANTS, enrich

    variant = artifact["variant"]
    if variant in VARIANTS:
        features = VARIANTS[variant]
        builder = enrich
    else:
        from models.review1_extended_features import EXTENDED_VARIANTS, extend
        features = EXTENDED_VARIANTS[variant]
        builder = extend

    columns = FEATURES + features
    model = artifact["model"]
    if (columns != artifact["feature_columns"] or list(model.feature_names_in_) != columns
            or artifact["positive_class"] != 1 or list(model.classes_) != [0, 1]):
        raise ValueError("Review 1 artifact schema does not match the fitted model")
    result = prepared.features.loc[:, ID_COLUMNS + FEATURES].copy()
    if result.empty:
        return result.assign(score=np.empty(0, dtype=float))
    frame = builder(prepared) if features else prepared.features
    scores = model.predict_proba(frame[columns].replace([np.inf, -np.inf], np.nan))[:, 1]
    if not (np.isfinite(scores) & (scores >= 0) & (scores <= 1)).all():
        raise ValueError("Expected finite candidate scores between zero and one")
    return result.assign(score=scores)
