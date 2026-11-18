"""Tune the development-CV winner and compare both variants on held-out trips."""
from argparse import ArgumentParser
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import optuna
import pandas as pd

from models.schema import FEATURES, read_training
from models.train_models import (add_arguments, compare, configuration, cross_validate, fit_predict,
                                 folds_for, make_models, metrics_for, partitions, plt)


def suggest(trial, name):
    if name == "LogisticRegression":
        return {"C": trial.suggest_float("C", .001, 10, log=True)}
    if name == "DecisionTree":
        return {"max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30)}
    if name == "RandomForest":
        return {"max_depth": trial.suggest_int("max_depth", 6, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
                "max_features": trial.suggest_float("max_features", .35, 1)}
    if name == "HistGradientBoosting":
        return {"learning_rate": trial.suggest_float("learning_rate", .02, .25, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 75),
                "l2_regularization": trial.suggest_float("l2_regularization", .001, 10, log=True)}
    return {"alpha": trial.suggest_float("alpha", .00001, .03, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", .0001, .01, log=True)}


def optimize(dev, held, model_name, folds, seed, trials, baseline_f1):
    if trials < 1:
        raise ValueError("At least one trial is required")
    splits = folds_for(dev, folds)
    baseline = make_models(seed)[model_name]

    def objective(trial):
        model = make_models(seed)[model_name]
        model.set_params(**{f"model__{key}": value for key, value in suggest(trial, model_name).items()})
        return cross_validate(model, dev, splits)[0]["f1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=2))
    start = time.perf_counter()
    study.optimize(objective, n_trials=trials)
    elapsed = time.perf_counter() - start
    tuned = make_models(seed)[model_name].set_params(
        **{f"model__{key}": value for key, value in study.best_params.items()})
    selected = "tuned" if study.best_value > baseline_f1 else "untuned"
    rows = []
    for variant, model, cv in [("untuned", baseline, baseline_f1),
                               ("tuned", tuned, study.best_value)]:
        probabilities, times = fit_predict(model, dev, held)
        rows.append(dict(variant=variant, development_cv_f1=cv,
                         **times, **metrics_for(held, probabilities)))
    return study, elapsed, selected, (tuned if selected == "tuned" else baseline), rows


def main():
    parser = ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    dev, held = partitions(read_training(args.train), args.seed, args.max_trips)
    config = configuration(args)
    summary_file = args.out / "evaluation_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    comparison_file = args.out / "model_comparison.csv"
    if summary.get("configuration") != config or not comparison_file.is_file():
        summary = compare(args, dev, held, config)
    name = summary["best_model"]
    baseline = pd.read_csv(comparison_file).set_index("model").loc[name, "f1"]
    study, elapsed, selected, model, rows = optimize(
        dev, held, name, args.folds, args.seed, args.trials, baseline)

    trials = study.trials_dataframe()
    trials.to_csv(args.out / "tuning_trials.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out / "tuned_vs_untuned.csv", index=False)
    ax = trials.set_index("number")["value"].plot(marker="o", xlabel="Trial",
                                                   ylabel="Development OOF F1")
    ax.figure.savefig(args.out / "tuning_history.png", bbox_inches="tight")
    plt.close(ax.figure)
    selected_f1 = next(row["development_cv_f1"] for row in rows if row["variant"] == selected)
    output = dict(model=name, selected_variant=selected, baseline_cv_f1=baseline,
                  best_trial_cv_f1=study.best_value, best_trial_parameters=study.best_params,
                  trials=args.trials, optimization_seconds=elapsed,
                  holdout=rows, configuration=config)
    (args.out / "tuning_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    model.fit(dev[FEATURES], dev.label)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    artifact = dict(model=model, feature_columns=FEATURES, positive_class=1, model_name=name,
                    variant=selected, hyperparameters=model.named_steps["model"].get_params(),
                    training_configuration=config | {"development_trips": summary["development_trips"],
                                                     "holdout_trips": summary["holdout_trips"]},
                    random_seed=args.seed, cross_validation={"f1": selected_f1},
                    dependency_versions={library: version(library) for library in
                                         ["scikit-learn", "numpy", "pandas", "joblib", "optuna"]})
    path = args.models_dir / "best_model.pkl"
    joblib.dump(artifact, path, compress=3)
    print(f"Saved {name} ({selected}) to {path}")


if __name__ == "__main__":
    main()
