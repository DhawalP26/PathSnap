"""Tune and freeze the 40-feature PathSnap classifier using development trips only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone

from models.review1_extended_features import EXTENDED_VARIANTS
from models.schema import FEATURES, ID_COLUMNS, read_training
from models.train_models import make_models, metrics_for

VARIANT = "review1_extended_lattice"
COLUMNS = FEATURES + EXTENDED_VARIANTS[VARIANT]
FAMILIES = ("HistGradientBoosting", "RandomForest", "LogisticRegression", "DecisionTree", "MLP")


def suggest(trial, name):
    if name == "HistGradientBoosting":
        return dict(learning_rate=trial.suggest_float("learning_rate", .04, .16, log=True),
                    max_iter=trial.suggest_int("max_iter", 120, 300, step=30),
                    max_leaf_nodes=trial.suggest_categorical("max_leaf_nodes", [15, 31, 63]),
                    max_depth=trial.suggest_categorical("max_depth", [None, 6, 10]),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 100, step=20),
                    l2_regularization=trial.suggest_float("l2_regularization", .05, 8, log=True),
                    max_bins=trial.suggest_categorical("max_bins", [127, 255]),
                    early_stopping=False)
    if name == "RandomForest":
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        return dict(n_estimators=trial.suggest_categorical("n_estimators", [80, 120, 180, 250]),
                    max_depth=trial.suggest_categorical("max_depth", [14, 20, 28, None]),
                    min_samples_split=trial.suggest_categorical("min_samples_split", [4, 12, 24]),
                    min_samples_leaf=trial.suggest_categorical("min_samples_leaf", [2, 5, 10, 20]),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", .20, .35]),
                    max_leaf_nodes=trial.suggest_categorical("max_leaf_nodes", [None, 4096, 8192]),
                    class_weight=trial.suggest_categorical(
                        "class_weight" if bootstrap else "nonbootstrap_class_weight",
                        [None, "balanced", "balanced_subsample"] if bootstrap else [None, "balanced"]),
                    bootstrap=bootstrap,
                    max_samples=trial.suggest_categorical("max_samples", [.65, .85, None]) if bootstrap else None)
    if name == "LogisticRegression":
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        return dict(solver=solver, C=trial.suggest_float("C", .02, 5, log=True),
                    penalty=trial.suggest_categorical("liblinear_penalty", ["l1", "l2"]) if solver == "liblinear" else "l2")
    if name == "DecisionTree":
        return dict(max_depth=trial.suggest_int("max_depth", 8, 24),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 40),
                    min_samples_split=trial.suggest_int("min_samples_split", 4, 40),
                    ccp_alpha=trial.suggest_float("ccp_alpha", 1e-8, 1e-4, log=True))
    width = trial.suggest_categorical("architecture", ["32", "64,32"])
    return dict(hidden_layer_sizes=tuple(map(int, width.split(","))),
                alpha=trial.suggest_float("alpha", .0001, .03, log=True),
                learning_rate_init=trial.suggest_float("learning_rate_init", .0003, .004, log=True),
                batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
                max_iter=trial.suggest_categorical("max_iter", [70, 110]))


def winners(frame, probabilities):
    return (frame[ID_COLUMNS + ["label"]].assign(score=probabilities)
            .sort_values(["score", "arc_id"], ascending=[False, True])
            .drop_duplicates(ID_COLUMNS[:2]))


def measurements(frame, probabilities, counts):
    top = winners(frame, probabilities)
    per_trip = top.groupby("trip_id").label.agg(correct="sum", observations="size")
    per_trip["top1"] = per_trip.correct / per_trip.observations
    per_trip["all_gps_top1"] = per_trip.correct / counts
    result = metrics_for(frame, probabilities)
    result.update(all_gps_top1=float(top.label.sum() / counts.reindex(frame.trip_id.unique()).sum()),
                  mean_trip_top1=float(per_trip.top1.mean()))
    return result, per_trip.reset_index().to_dict("records")


def evaluate(model, frame, splits, counts, trial=None):
    oof = np.full(len(frame), np.nan)
    folds = []
    for i, (train, valid) in enumerate(splits):
        fit = clone(model)
        training, validation = frame.iloc[train], frame.iloc[valid]
        start = time.perf_counter()
        fit.fit(training[COLUMNS], training.label)
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        p = fit.predict_proba(validation[COLUMNS])[:, 1]
        inference = time.perf_counter() - start
        oof[valid] = p
        top = winners(validation, p)
        train_top = winners(training, fit.predict_proba(training[COLUMNS])[:, 1])
        folds.append(dict(fold=i+1, correct=int(top.label.sum()), observations=len(top),
                          validation_top1=float(top.label.mean()), train_top1=float(train_top.label.mean()),
                          train_seconds=fit_time, inference_seconds=inference,
                          validation_rows=len(validation)))
        if trial is not None:
            trial.set_user_attr("folds", folds)
            trial.report(sum(r["correct"] for r in folds)/sum(r["observations"] for r in folds), i)
            if i < len(splits)-1 and trial.should_prune():
                raise optuna.TrialPruned()
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions must be finite and complete")
    scores, per_trip = measurements(frame, oof, counts)
    scores.update(mean_train_top1=float(np.mean([r["train_top1"] for r in folds])),
                  mean_validation_top1=float(np.mean([r["validation_top1"] for r in folds])),
                  fold_std=float(np.std([r["validation_top1"] for r in folds])),
                  train_seconds=sum(r["train_seconds"] for r in folds),
                  inference_seconds=sum(r["inference_seconds"] for r in folds))
    scores["gap"] = scores["mean_train_top1"] - scores["mean_validation_top1"]
    return dict(metrics=scores, folds=folds, trips=per_trip)


def digest(path):
    with path.open("rb") as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--out", default=Path("results/member2/final_review"), type=Path)
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--recover-interrupted", action="store_true",
                   help="Mark RUNNING trials failed after confirming the previous worker stopped")
    args = p.parse_args()
    if not args.finalize and not args.diagnostics and not args.cache_only and not args.family:
        p.error("--family or --finalize is required")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.frozen / "manifest.json").read_text())
    for filename, expected in manifest["sha256"].items():
        if digest(args.frozen / filename) != expected:
            raise ValueError(f"Frozen checksum mismatch: {filename}")
    cfg = json.loads((args.frozen / "folds.json").read_text())
    frozen_selection = json.loads((args.frozen / "selection.json").read_text())
    if (digest(args.train) != cfg["training_sha256"] or frozen_selection["columns"] != COLUMNS
            or manifest["frozen_features"] != COLUMNS or len(COLUMNS) != 40):
        raise ValueError("Frozen training fingerprint or feature order differs")
    dev_ids, held_ids = set(cfg["development_trips"]), set(cfg["holdout_trips"])
    assignments = {trip: i for i, fold in enumerate(cfg["folds"]) for trip in fold}
    if (len(dev_ids) != 60 or len(held_ids) != 20 or dev_ids & held_ids or set(assignments) != dev_ids
            or sum(map(len, cfg["folds"])) != 60):
        raise ValueError("Invalid frozen trip split")
    coverage = pd.read_csv(args.train.with_name("training_coverage.csv"), dtype={"trip_id": str}).set_index("trip_id")
    if set(coverage.index) != dev_ids | held_ids:
        raise ValueError("Unexpected trip inventory")

    raw = None

    def load(ids):
        nonlocal raw
        if raw is None:
            raw = read_training(args.train)
            if set(raw.trip_id) != dev_ids | held_ids:
                raise ValueError("Unexpected training CSV trip inventory")
        items = []
        for trip, labels in raw[raw.trip_id.isin(ids)].groupby("trip_id", sort=False):
            cache = args.frozen / "feature_cache" / f"{trip}.pkl"
            local_cache = args.out / "feature_cache" / cache.name
            if local_cache.is_file():
                cache = local_cache
            if cache.is_file():
                features = pd.read_pickle(cache)
            else:
                from preprocessing import prepare_trip
                from models.review1_extended_features import extend
                features = extend(prepare_trip(trip, partition="train"))
                local_cache.parent.mkdir(exist_ok=True)
                features.to_pickle(local_cache)
            cols = ID_COLUMNS + COLUMNS[len(FEATURES):]
            merged = labels.merge(features[cols], on=ID_COLUMNS, how="left", validate="one_to_one", indicator=True)
            if not merged["_merge"].eq("both").all():
                raise ValueError(f"Candidate mismatch for {trip}")
            items.append(merged.drop(columns="_merge"))
        return pd.concat(items, ignore_index=True)

    def models(name, params):
        model = make_models(cfg["seed"])[name]
        if name == "RandomForest":
            model.set_params(model__n_jobs=args.jobs)
        return model.set_params(**{f"model__{k}": v for k, v in params.items()})

    cache_path = args.out / "development_40.pkl"
    cache_key = {"manifest": digest(args.frozen / "manifest.json"),
                 "training": cfg["training_sha256"], "columns": COLUMNS}
    cache_metadata = args.out / "development_cache.json"
    if (cache_path.exists() and cache_metadata.exists()
            and json.loads(cache_metadata.read_text()) == cache_key):
        dev = pd.read_pickle(cache_path)
    else:
        dev = load(dev_ids)
        tmp = cache_path.with_suffix(".tmp")
        dev.to_pickle(tmp)
        tmp.replace(cache_path)
        write_json(cache_metadata, cache_key)
    if set(dev.trip_id) != dev_ids or len(dev) != coverage.loc[sorted(dev_ids)].candidate_rows.sum():
        raise ValueError("Development cache does not match the frozen inventory")
    if args.cache_only:
        print(f"Cached {len(dev):,} labeled development candidates", flush=True)
        return
    group = dev.trip_id.map(assignments).to_numpy()
    splits = [(np.flatnonzero(group != fold), np.flatnonzero(group == fold)) for fold in range(3)]
    db = args.out / "studies.sqlite3"
    storage = f"sqlite:///{db.resolve().as_posix()}"
    if args.diagnostics:
        from models.final_review_diagnostics import analyze
        candidate = json.loads((args.out / "selection.json").read_text())
        analyze(dev, splits, coverage, models(candidate["model"], candidate["params"]),
                args.frozen, args.out, cfg["seed"])
        return
    if not args.finalize:
        if (args.out / "selection.json").exists():
            raise ValueError("This release already has a frozen selection")
        name = args.family
        study = optuna.create_study(direction="maximize", study_name=name, storage=storage,
                                    load_if_exists=True, sampler=optuna.samplers.TPESampler(
                                        seed=cfg["seed"], n_startup_trials=3),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
        fingerprint = digest(args.frozen / "manifest.json")
        if study.user_attrs.get("frozen_manifest", fingerprint) != fingerprint:
            raise ValueError("Study belongs to a different frozen run")
        study.set_user_attr("frozen_manifest", fingerprint)
        running = study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
        if running and not args.recover_interrupted:
            raise ValueError("Study has running trials. Check for an existing worker before recovering.")
        for stale in running:
            study.tell(stale.number, state=optuna.trial.TrialState.FAIL)

        def objective(trial):
            params = suggest(trial, name)
            trial.set_user_attr("model_params", params)
            for old in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
                if old.user_attrs.get("model_params") == trial.user_attrs["model_params"]:
                    trial.set_user_attr("duplicate_of", old.number)
                    raise optuna.TrialPruned()
            result = evaluate(models(name, params), dev, splits, coverage.observations, trial)
            trial.set_user_attr("result", result)
            return result["metrics"]["top1_accuracy"]

        study.optimize(objective, n_trials=max(0, args.trials - sum(t.state in (optuna.trial.TrialState.COMPLETE,
                                             optuna.trial.TrialState.PRUNED) for t in study.trials)),
                       gc_after_trial=True)
        return

    baseline = pd.read_csv(args.frozen / "comparison.csv")
    summaries, rows, folds, trips, trials = {}, [], [], [], []
    candidates = []
    for row in baseline.to_dict("records"):
        name = row["model"]
        previous_folds = pd.read_json(args.frozen / f"{name}_development.json", typ="series")["fold_metrics"]
        previous_trips = pd.read_csv(args.frozen / f"{name}_per_trip.csv")
        candidates.append(dict(model=name, configuration="untuned", params={},
                               top1_accuracy=row["top1_accuracy"], f1=row["f1"],
                               mean_trip_top1=float(previous_trips.conditional_top1.mean()),
                               gap=float(np.mean([r["train_top1"] - r["validation_top1"] for r in previous_folds])),
                               fold_std=float(np.std([r["validation_top1"] for r in previous_folds]))))
        if not db.exists():
            continue
        try:
            study = optuna.load_study(study_name=name, storage=storage)
        except KeyError:
            continue
        if study.user_attrs.get("frozen_manifest") != digest(args.frozen / "manifest.json"):
            raise ValueError(f"Study {name} belongs to a different frozen run")
        if any(not t.state.is_finished() for t in study.trials):
            raise ValueError(f"Study {name} still has unfinished trials")
        trials.extend(dict(model=name, trial=t.number, state=t.state.name, top1=t.value,
                           seconds=t.duration.total_seconds() if t.duration else None,
                           parameters=json.dumps(t.user_attrs.get("model_params")),
                           intermediate=json.dumps(t.intermediate_values)) for t in study.trials)
        if name != "RandomForest":
            continue
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            continue
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        peak = max(t.value for t in complete)
        t = max((t for t in complete if t.value >= peak - .0003),
                key=lambda t: (t.user_attrs["result"]["metrics"]["mean_trip_top1"],
                               t.user_attrs["result"]["metrics"]["f1"], t.value))
        result = t.user_attrs["result"]
        params = t.user_attrs["model_params"]
        summaries[name] = dict(trial=t.number, params=params, **result["metrics"])
        rows.append(dict(model=name, **result["metrics"]))
        folds.extend(dict(model=name, **r) for r in result["folds"])
        trips.extend(dict(model=name, **r) for r in result["trips"])
        candidates.append(dict(model=name, configuration="tuned", params=params, **result["metrics"]))
    if not rows:
        raise ValueError("No completed tuning trials")
    selection_file = args.out / "selection.json"
    if selection_file.exists():
        candidate = json.loads(selection_file.read_text())
    else:
        floor = max(x["top1_accuracy"] for x in candidates if x["configuration"] == "untuned")
        peak = max(x["top1_accuracy"] for x in candidates)
        eligible = [x for x in candidates if x["top1_accuracy"] >= max(floor, peak - .0003)]
        candidate = max(eligible, key=lambda x: (x["mean_trip_top1"], x["f1"], -x["gap"],
                                                  -x["fold_std"], x["top1_accuracy"]))
        write_json(selection_file, candidate)
    write_json(args.out / "best_hyperparameters.json", summaries)
    write_json(args.out / "feature_schema.json", dict(variant=VARIANT, columns=COLUMNS))
    write_json(args.out / "evaluation_configuration.json", cfg)
    for suffix, data in [("tuning_trials", trials), ("model_comparison", rows),
                         ("fold_metrics", folds), ("per_trip_metrics", trips)]:
        pd.DataFrame(data).to_csv(args.out / f"{suffix}.csv", index=False)
    name, params = candidate["model"], candidate["params"]
    held = load(held_ids)
    for path, frame, trip_ids in [("evaluation_model.pkl", dev, dev_ids),
                                  ("final_model.pkl", pd.concat([dev, held], ignore_index=True), dev_ids | held_ids)]:
        destination = args.out / path
        if destination.exists():
            artifact = joblib.load(destination)
            if artifact["training_trips"] != sorted(trip_ids) or artifact["selection"] != candidate:
                raise ValueError(f"Existing artifact does not match the frozen selection: {destination}")
            model, fit_seconds = artifact["model"], artifact["fit_seconds"]
        else:
            reused = (path == "evaluation_model.pkl" and candidate["configuration"] == "untuned"
                      and name == manifest["frozen_model"])
            if reused:
                frozen_model = joblib.load(args.frozen / "review1_model.pkl")
                if (frozen_model["development_trips"] != sorted(trip_ids)
                        or frozen_model["feature_columns"] != COLUMNS or frozen_model["variant"] != VARIANT):
                    raise ValueError("Frozen evaluation model does not match selection")
                model, fit_seconds = frozen_model["model"], None
            else:
                started = time.perf_counter()
                model = models(name, params).fit(frame[COLUMNS], frame.label)
                fit_seconds = time.perf_counter() - started
            artifact = dict(model=model, variant=VARIANT, feature_columns=COLUMNS, model_name=name,
                            positive_class=1, training_trips=sorted(trip_ids), training_configuration=cfg,
                            selection=candidate, fit_seconds=fit_seconds, reused_frozen_fit=reused,
                            estimator_parameters=model.named_steps["model"].get_params())
            tmp = destination.with_suffix(".tmp")
            joblib.dump(artifact, tmp, compress=3)
            tmp.replace(destination)
        if path == "evaluation_model.pkl" and not (args.out / "holdout_evaluation.json").exists():
            started = time.perf_counter()
            probabilities = model.predict_proba(held[COLUMNS])[:, 1]
            inference_seconds = time.perf_counter() - started
            report, per_trip = measurements(held, probabilities, coverage.observations)
            report.update(inference_seconds=inference_seconds, fit_seconds=fit_seconds, candidate_recall=float(coverage.loc[sorted(held_ids)].retrieved.sum() /
                                                 coverage.loc[sorted(held_ids)].aligned.sum()),
                          per_trip=per_trip)
            write_json(args.out / "holdout_evaluation.json", report)
    print("Final selection:", name, candidate["top1_accuracy"], flush=True)


if __name__ == "__main__":
    main()
