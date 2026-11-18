"""Stable inference interface for Member 3 / the API."""
from pathlib import Path
import joblib
import pandas as pd

DEFAULT_MODEL = Path(__file__).with_name("best_model.pkl")


def load_model(model_path=DEFAULT_MODEL):
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Unexpected model artifact format")
    return artifact


def score_candidates(candidate_features: pd.DataFrame, model_path=DEFAULT_MODEL):
    """Return P(label=1), preserving the exact input row order and index."""
    artifact = load_model(model_path)
    columns = artifact["feature_columns"]
    missing = [c for c in columns if c not in candidate_features.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    probabilities = artifact["model"].predict_proba(candidate_features[columns])[:, 1]
    result = candidate_features.copy()
    result["score"] = probabilities
    return result
