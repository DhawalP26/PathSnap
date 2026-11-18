"""Optuna Bayesian optimisation for Random Forest using GroupKFold on training trips."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from train_models import FEATURES, ID_COLUMNS, validate_frame, metrics_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_features.csv")
    ap.add_argument("--student-test", default="data/student_test_features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out, model_dir = Path(args.out), Path(args.models_dir)
    out.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train); validate_frame(train, True, "Training CSV")
    train["label"] = train["label"].astype(int)
    X, y, groups = train[FEATURES], train["label"], train["trip_id"].astype(str)
    if groups.nunique() < args.folds: raise ValueError("Not enough distinct trips for GroupKFold")
    cv = GroupKFold(n_splits=args.folds)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8]),
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": args.seed,
        }
        scores = []
        for tr, va in cv.split(X, y, groups):
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("model", RandomForestClassifier(**params))])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            scores.append(f1_score(y.iloc[va], (p >= 0.5).astype(int), zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="pathsnap_rf_groupkfold",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    t0 = time.perf_counter(); study.optimize(objective, n_trials=args.trials)
    tuning_seconds = time.perf_counter() - t0
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out / "tuning_trials.csv", index=False)
    best_params = dict(study.best_params)
    final_params = {**best_params, "class_weight": "balanced_subsample", "n_jobs": -1,
                    "random_state": args.seed}
    final_model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                            ("model", RandomForestClassifier(**final_params))])
    t1 = time.perf_counter(); final_model.fit(X, y); final_fit_s = time.perf_counter() - t1
    artifact = {"model": final_model, "feature_columns": FEATURES,
                "positive_class": 1, "model_name": "TunedRandomForest",
                "best_params": final_params, "cv_best_f1": study.best_value,
                "cv_folds": args.folds, "random_seed": args.seed}
    joblib.dump(artifact, model_dir / "best_model.pkl")

    summary = {"best_cv_f1": study.best_value, "best_params": final_params,
               "trials": args.trials, "cv_folds": args.folds,
               "optimisation_seconds": tuning_seconds, "final_fit_seconds": final_fit_s}
    student_path = Path(args.student_test)
    if student_path.exists():
        student = pd.read_csv(student_path); validate_frame(student, False, "Student-test CSV")
        if "label" in student.columns and student["label"].notna().all():
            ytest = student["label"].astype(int)
            prob = final_model.predict_proba(student[FEATURES])[:, 1]
            pred = (prob >= 0.5).astype(int)
            summary["student_test_metrics"] = metrics_for(ytest, pred, prob, student)
        else:
            summary["student_test_metrics"] = "Not calculated: student-test labels absent or incomplete."
    with open(out / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print(f"Saved: {model_dir / 'best_model.pkl'}")

if __name__ == "__main__": main()
