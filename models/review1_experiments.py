"""Grouped feature ablation and frozen Review 1 comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import prepare_trip
from models import train_models as bench
from models.review1_features import VARIANTS, enrich
from models.schema import FEATURES, ID_COLUMNS, read_training

def main(variants=VARIANTS, feature_builder=enrich):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference = json.loads((args.baseline / "evaluation_summary.json").read_text())
    config = reference["configuration"]
    with args.train.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1048576), b""):
            digest.update(chunk)
    if digest.hexdigest() != config["training_sha256"]:
        raise ValueError("Training CSV differs from baseline")
    added_columns = list(dict.fromkeys(column for columns in variants.values() for column in columns))
    dev_ids, held_ids = map(set, (reference["development_trips"], reference["holdout_trips"]))
    raw = read_training(args.train)
    coverage = pd.read_csv(args.train.with_name("training_coverage.csv"), dtype={"trip_id": str})
    counts = coverage.set_index("trip_id").observations
    assert not dev_ids & held_ids and len(dev_ids | held_ids) == 80
    assert set(raw.trip_id) <= dev_ids | held_ids
    assert set(counts.index) == dev_ids | held_ids
    assert len(raw) == reference["development_rows"] + reference["holdout_rows"]

    def load(ids):
        parts = []
        for trip_id, labels in raw[raw.trip_id.isin(ids)].groupby("trip_id", sort=False):
            path = args.out / "feature_cache" / f"{trip_id}.pkl"
            if path.exists():
                features = pd.read_pickle(path)
            else:
                prepared = prepare_trip(str(trip_id), partition="train")
                features = feature_builder(prepared)
                path.parent.mkdir(exist_ok=True)
                tmp = path.with_suffix(".tmp")
                features.to_pickle(tmp)
                tmp.replace(path)
            merged = labels.merge(features[ID_COLUMNS + added_columns],
                                  on=ID_COLUMNS, how="left", validate="one_to_one", indicator=True)
            if not merged["_merge"].eq("both").all():
                raise ValueError(f"Candidate mismatch for {trip_id}")
            parts.append(merged.drop(columns="_merge"))
        return pd.concat(parts, ignore_index=True)

    def model_set():
        models = bench.make_models(config["seed"])
        models["RandomForest"].set_params(model__n_jobs=args.jobs)
        return models

    def measure(frame, score):
        result = bench.metrics_for(frame, score)
        top = winners(frame, score)
        result["all_gps_top1"] = top.label.sum() / counts.reindex(frame.trip_id.unique()).sum()
        return result

    def winners(frame, score):
        return (frame[ID_COLUMNS + ["label"]].assign(score=score)
                .sort_values(["score", "arc_id"], ascending=[False, True])
                .drop_duplicates(ID_COLUMNS[:2]))

    def evaluate(model, frame, splits, columns, collect=False):
        from sklearn.base import clone
        oof = np.full(len(frame), np.nan)
        fold_rows = []
        for index, (train, valid) in enumerate(splits, 1):
            fit = clone(model).fit(frame.iloc[train][columns], frame.label.iloc[train])
            valid_score = fit.predict_proba(frame.iloc[valid][columns])[:, 1]
            oof[valid] = valid_score
            if collect:
                training_score = fit.predict_proba(frame.iloc[train][columns])[:, 1]
                fold_rows.append({"fold": index,
                                  "train_top1": bench.metrics_for(frame.iloc[train], training_score)["top1_accuracy"],
                                  "validation_top1": bench.metrics_for(frame.iloc[valid], valid_score)["top1_accuracy"]})
        return measure(frame, oof), fold_rows, oof

    dev = load(dev_ids)
    splits = bench.folds_for(dev, config["folds"])
    (args.out / "folds.json").write_text(json.dumps({
        "seed": config["seed"], "folds": [sorted(dev.trip_id.iloc[v].unique()) for _, v in splits],
        "development_trips": sorted(dev_ids), "holdout_trips": sorted(held_ids),
        "training_sha256": digest.hexdigest()}, indent=2), encoding="utf-8")
    closest = dev.sort_values(["distance_m", "arc_id"]).drop_duplicates(ID_COLUMNS[:2])
    positive = dev[dev.label.eq(1)]
    diagnostics = {"closest_candidate_top1": float(closest.label.mean()),
                   "positive_rank_distribution": positive.candidate_rank.value_counts().sort_index().to_dict(),
                   "development_observations_scored": int(len(closest)),
                   "all_development_observations": int(counts.reindex(sorted(dev_ids)).sum())}
    (args.out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    ablation = []
    for variant, features in variants.items():
        result, _, _ = evaluate(model_set()["HistGradientBoosting"], dev, splits, FEATURES + features)
        ablation.append(dict(variant=variant, **result))
        pd.DataFrame(ablation).to_csv(args.out / "ablation.csv", index=False)
        print("Ablation", variant, result["top1_accuracy"], flush=True)
    winner = max(ablation, key=lambda r: r["top1_accuracy"])["variant"]
    columns = FEATURES + variants[winner]
    (args.out / "selection.json").write_text(json.dumps(
        {"variant": winner, "columns": columns, "selection_metric": "development candidate top-1",
         "ablation_model": "HistGradientBoosting"}, indent=2), encoding="utf-8")

    rows = []
    for name, model in model_set().items():
        detail_path = args.out / f"{name}_development.json"
        if detail_path.exists():
            rows.append(json.loads(detail_path.read_text(encoding="utf-8")))
            continue
        metrics, folds, oof = evaluate(model, dev, splits, columns, collect=True)
        top = winners(dev, oof).groupby("trip_id").label.agg(["sum", "size"])
        top["conditional_top1"] = top["sum"] / top["size"]
        top["all_gps_top1"] = top["sum"] / counts
        top.to_csv(args.out / f"{name}_per_trip.csv")
        row = dict(model=name, **metrics, fold_metrics=folds)
        detail_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out / "comparison.csv", index=False)
        print(name, metrics["top1_accuracy"], flush=True)
    pd.DataFrame(rows).to_csv(args.out / "comparison.csv", index=False)
    best = max(rows, key=lambda r: r["top1_accuracy"])["model"]
    model = model_set()[best].fit(dev[columns], dev.label)
    held = load(held_ids)
    held_scores = model.predict_proba(held[columns])[:, 1]
    held_results = measure(held, held_scores)
    baseline_held = model_set()[best].fit(dev[FEATURES], dev.label)
    baseline_result = measure(held, baseline_held.predict_proba(held[FEATURES])[:, 1])
    (args.out / "holdout.json").write_text(json.dumps(
        {"model": best, "selected": held_results, "same_model_original_features": baseline_result},
        indent=2), encoding="utf-8")
    joblib.dump({"model": model, "model_name": best, "variant": winner, "feature_columns": columns,
                 "positive_class": 1, "training_configuration": config,
                 "development_trips": sorted(dev_ids), "holdout_trips": sorted(held_ids)},
                args.out / "review1_model.pkl", compress=3)


if __name__ == "__main__":
    main()
