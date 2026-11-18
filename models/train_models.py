"""Train and compare five candidate-road classifiers using trip-wise CV."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

FEATURES = ["distance_m", "heading_diff", "speed_mps", "time_gap_s",
            "gps_step_m", "segment_length_m", "candidate_rank"]
ID_COLUMNS = ["trip_id", "gps_index", "arc_id"]


def make_models(seed: int):
    scaled = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())])
    unscaled = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return {
        "LogisticRegression": Pipeline([("prep", scaled), ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed))]),
        "DecisionTree": Pipeline([("prep", unscaled), ("model", DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=seed))]),
        "RandomForest": Pipeline([("prep", unscaled), ("model", RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed))]),
        "HistGradientBoosting": Pipeline([("prep", unscaled), ("model", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            random_state=seed))]),
        "MLP": Pipeline([("prep", scaled), ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001,
            learning_rate_init=0.001, max_iter=250, early_stopping=True,
            random_state=seed))]),
    }


def validate_frame(df: pd.DataFrame, require_label: bool, name: str):
    required = ID_COLUMNS + FEATURES + (["label"] if require_label else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if df[ID_COLUMNS].isnull().any().any():
        raise ValueError(f"{name} has missing identifiers in {ID_COLUMNS}")
    if require_label:
        labels = set(pd.to_numeric(df["label"], errors="coerce").dropna().unique())
        if not labels.issubset({0, 1}) or labels != {0, 1}:
            raise ValueError(f"{name}: label must contain both classes 0 and 1; got {labels}")


def metrics_for(y, pred, prob, df):
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "log_loss": log_loss(y, prob, labels=[0, 1]),
    }
    # Per-observation top-1: is the highest-scored candidate labelled positive?
    scored = df[ID_COLUMNS[:2]].copy()
    scored["label"] = np.asarray(y)
    scored["score"] = np.asarray(prob)
    top = scored.sort_values("score", ascending=False).drop_duplicates(
        ["trip_id", "gps_index"])
    out["top1_accuracy"] = float(top["label"].mean()) if len(top) else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def cross_validate(model, X, y, groups, df, n_splits, seed):
    splitter = GroupKFold(n_splits=n_splits)
    oof_prob = np.full(len(df), np.nan)
    fold_rows = []
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups), start=1):
        fitted = clone(model)
        t0 = time.perf_counter()
        fitted.fit(X.iloc[tr], y.iloc[tr])
        train_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        p = fitted.predict_proba(X.iloc[va])[:, 1]
        infer_ms_per_row = (time.perf_counter() - t1) * 1000 / max(len(va), 1)
        oof_prob[va] = p
        pred = (p >= 0.5).astype(int)
        m = metrics_for(y.iloc[va], pred, p, df.iloc[va])
        fold_rows.append({"fold": fold, "train_trips": len(set(groups.iloc[tr])),
                          "validation_trips": len(set(groups.iloc[va])),
                          "train_seconds": train_s,
                          "inference_ms_per_candidate": infer_ms_per_row, **m})
    if np.isnan(oof_prob).any():
        raise RuntimeError("Some rows did not receive out-of-fold predictions")
    oof_pred = (oof_prob >= 0.5).astype(int)
    overall = metrics_for(y, oof_pred, oof_prob, df)
    return overall, fold_rows


def evaluate_holdout(model, df):
    if "label" not in df.columns or df["label"].isna().any():
        return None
    y = df["label"].astype(int)
    p = model.predict_proba(df[FEATURES])[:, 1]
    pred = (p >= 0.5).astype(int)
    return metrics_for(y, pred, p, df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_features.csv")
    ap.add_argument("--student-test", default="data/student_test_features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out, model_dir = Path(args.out), Path(args.models_dir)
    out.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    validate_frame(train, True, "Training CSV")
    train["label"] = train["label"].astype(int)
    X, y, groups = train[FEATURES], train["label"], train["trip_id"].astype(str)
    n_groups = groups.nunique()
    if n_groups < args.folds:
        raise ValueError(f"Need at least {args.folds} distinct training trips; found {n_groups}")

    student = None
    if Path(args.student_test).exists():
        student = pd.read_csv(args.student_test)
        validate_frame(student, False, "Student-test CSV")
        if "label" in student.columns and student["label"].notna().all():
            student["label"] = student["label"].astype(int)
        else:
            print("Student-test labels unavailable: will save predictions only; no test metrics.")

    comparison, fold_records = [], []
    best_name, best_f1 = None, -1.0
    for name, model in make_models(args.seed).items():
        print(f"\n[{name}] running {args.folds}-fold GroupKFold across {n_groups} trips...")
        cv, folds = cross_validate(model, X, y, groups, train, args.folds, args.seed)
        for row in folds: fold_records.append({"model": name, **row})
        t0 = time.perf_counter(); model.fit(X, y); fit_s = time.perf_counter() - t0
        t1 = time.perf_counter(); model.predict_proba(X.iloc[:min(1000, len(X))]); infer_s = time.perf_counter() - t1
        row = {"model": name, "configuration": str(model.named_steps["model"].get_params()),
               "cv_train_seconds_full_fit": fit_s,
               "inference_ms_per_candidate_sample": infer_s * 1000 / min(1000, len(X)),
               **{f"cv_{k}": v for k, v in cv.items()}}
        if student is not None:
            holdout = evaluate_holdout(model, student)
            if holdout is not None:
                row.update({f"student_test_{k}": v for k, v in holdout.items()})
        comparison.append(row)
        print(f"CV F1={cv['f1']:.4f} | Top-1={cv['top1_accuracy']:.4f} | LogLoss={cv['log_loss']:.4f}")
        if cv["f1"] > best_f1:
            best_f1, best_name = cv["f1"], name

    pd.DataFrame(comparison).sort_values("cv_f1", ascending=False).to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(fold_records).to_csv(out / "cv_fold_results.csv", index=False)
    print(f"\nBest initial model by trip-wise CV F1: {best_name} ({best_f1:.4f})")
    print("Next: run bayesian_tuning.py; do not treat this untuned winner as final until tuning and student-test comparison.")

if __name__ == "__main__": main()
