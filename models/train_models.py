"""Train and compare the five candidate classifiers with shared trip-wise folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from models.schema import DEFAULT_OUT, DEFAULT_TRAIN, FEATURES, ID_COLUMNS, read_training


def make_models(seed):
    scaled = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                       ("scaler", StandardScaler())])
    unscaled = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    return {
        "LogisticRegression": Pipeline([("prep", scaled), ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed))]),
        "DecisionTree": Pipeline([("prep", unscaled), ("model", DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=seed))]),
        "RandomForest": Pipeline([("prep", unscaled), ("model", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=2, random_state=seed))]),
        "HistGradientBoosting": Pipeline([("prep", unscaled), ("model", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=.08, max_leaf_nodes=31, l2_regularization=1,
            early_stopping=False, random_state=seed))]),
        "MLP": Pipeline([("prep", scaled), ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=.001, max_iter=250, random_state=seed))]),
    }


def metrics_for(frame, probabilities):
    y = frame.label
    pred = probabilities >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ranked = frame[ID_COLUMNS + ["label"]].assign(score=probabilities)
    top = ranked.sort_values(["score", "arc_id"], ascending=[False, True]).drop_duplicates(
        ID_COLUMNS[:2])
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                log_loss=log_loss(y, probabilities, labels=[0, 1]), top1_accuracy=float(top.label.mean()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def folds_for(frame, folds):
    if not 2 <= folds <= frame.trip_id.nunique():
        raise ValueError("Need at least two folds and one distinct trip per fold")
    splits = list(GroupKFold(n_splits=folds).split(frame, frame.label, frame.trip_id))
    if any(frame.label.iloc[train].nunique() < 2 for train, _ in splits):
        raise ValueError("Each training fold must contain both labels")
    return splits


def fit_predict(model, train, validation):
    fitted = clone(model)
    start = time.perf_counter()
    fitted.fit(train[FEATURES], train.label)
    training_time = time.perf_counter() - start
    start = time.perf_counter()
    probabilities = fitted.predict_proba(validation[FEATURES])[:, 1]
    return probabilities, {"train_seconds": training_time,
                           "inference_seconds": time.perf_counter() - start}


def cross_validate(model, frame, splits):
    probabilities = np.full(len(frame), np.nan)
    rows = []
    for fold, (train, validation) in enumerate(splits, 1):
        held = frame.iloc[validation]
        prob, times = fit_predict(model, frame.iloc[train], held)
        probabilities[validation] = prob
        rows.append(dict(fold=fold, train_trips=frame.trip_id.iloc[train].nunique(),
                         validation_trips=held.trip_id.nunique(), **times, **metrics_for(held, prob)))
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Invalid out-of-fold probabilities")
    return metrics_for(frame, probabilities), rows


def partitions(frame, seed, max_trips=None):
    if max_trips is not None:
        ids = np.sort(frame.trip_id.unique())
        if not 3 <= max_trips <= len(ids):
            raise ValueError("max_trips must be between three and the total number of trips")
        sampled = np.random.default_rng(seed).choice(ids, max_trips, replace=False)
        frame = frame[frame.trip_id.isin(sampled)]
    dev, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
                        .split(frame, frame.label, frame.trip_id))
    return frame.iloc[dev], frame.iloc[holdout]


def add_arguments(parser):
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-trips", type=int)


def configuration(args):
    with open(args.train, "rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return dict(training_sha256=digest.hexdigest(), seed=args.seed, folds=args.folds,
                max_trips=args.max_trips, features=FEATURES)


def compare(args, dev, held, config):
    splits = folds_for(dev, args.folds)
    args.out.mkdir(parents=True, exist_ok=True)
    matrices = args.out / "confusion_matrices"
    matrices.mkdir(exist_ok=True)
    comparison, details = [], []
    for name, model in make_models(args.seed).items():
        metrics, rows = cross_validate(model, dev, splits)
        comparison.append(dict(model=name, **metrics,
                               train_seconds=sum(row["train_seconds"] for row in rows),
                               inference_seconds=sum(row["inference_seconds"] for row in rows)))
        details.extend(dict(model=name, **row) for row in rows)
        matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
        display = ConfusionMatrixDisplay(np.array(matrix)).plot(colorbar=False)
        display.figure_.savefig(matrices / f"{name}.png", bbox_inches="tight")
        plt.close(display.figure_)
        print(f"{name}: F1 {metrics['f1']:.4f}, top-1 {metrics['top1_accuracy']:.4f}", flush=True)
    table = pd.DataFrame(comparison).sort_values(["f1", "model"], ascending=[False, True])
    table.to_csv(args.out / "model_comparison.csv", index=False)
    pd.DataFrame(details).to_csv(args.out / "cv_fold_results.csv", index=False)
    ax = table.set_index("model")[["f1", "top1_accuracy"]].plot.bar(rot=20, ylim=(0, 1))
    ax.figure.savefig(args.out / "model_comparison.png", bbox_inches="tight")
    plt.close(ax.figure)
    summary = dict(configuration=config, best_model=table.iloc[0].model,
                   development_trips=sorted(dev.trip_id.unique()), development_rows=len(dev),
                   holdout_trips=sorted(held.trip_id.unique()), holdout_rows=len(held))
    coverage_path = Path(args.train).with_name("training_coverage.csv")
    if coverage_path.is_file():
        coverage = pd.read_csv(coverage_path, dtype={"trip_id": "string"})
        coverage = coverage[coverage.trip_id.isin(set(dev.trip_id) | set(held.trip_id))]
        if coverage.aligned.sum():
            summary["candidate_recall"] = float(coverage.retrieved.sum() / coverage.aligned.sum())
    (args.out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    dev, held = partitions(read_training(args.train), args.seed, args.max_trips)
    compare(args, dev, held, configuration(args))


if __name__ == "__main__":
    main()
