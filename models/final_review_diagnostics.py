"""Development-only generalization diagnostics for the frozen final model."""
import json

import joblib
import numpy as np
import pandas as pd

from models.final_review_tuning import COLUMNS, winners, write_json
from models.review1_extended_features import LATTICE, TEMPORAL
from models.review1_features import GROUPS
from models.schema import FEATURES


def analyze(dev, splits, coverage, model, frozen, out, seed):
    candidate_groups = dev.groupby(["trip_id", "gps_index"]).label.agg(["size", "sum"])
    sample = dev[COLUMNS].sample(n=min(50000, len(dev)), random_state=seed)
    correlation = sample.corr()
    pairs = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), 1))
    highest = pairs.stack().abs().sort_values(ascending=False).head(15)
    pairs_out = [{"first": a, "second": b, "correlation": float(correlation.loc[a, b])}
                 for a, b in highest.index]
    folds = pd.read_csv(out / "fold_metrics.csv")
    per_trip = pd.read_csv(out / "per_trip_metrics.csv", dtype={"trip_id": str})
    selection = json.loads((out / "selection.json").read_text())
    name = selection["model"]
    original = pd.read_json(frozen / f"{name}_development.json", typ="series")
    old_trips = pd.read_csv(frozen / f"{name}_per_trip.csv", dtype={"trip_id": str})
    full_fold = (folds[(folds.model == name) & (folds.fold == 1)].iloc[0].to_dict()
                 if selection["configuration"] == "tuned" else original["fold_metrics"][0])
    curves = [{"training_trips": dev.iloc[splits[0][0]].trip_id.nunique(),
               "train_top1": full_fold["train_top1"],
               "validation_top1": full_fold["validation_top1"]}]
    fitted = joblib.load(out / "evaluation_model.pkl")["model"]
    estimator = fitted.named_steps["model"]
    groups = {"base": FEATURES, **GROUPS, "extended_temporal": TEMPORAL, "lattice": LATTICE}
    importance = [dict(feature_group=key, impurity_importance=float(sum(
        estimator.feature_importances_[COLUMNS.index(col)] for col in cols)))
                  for key, cols in groups.items()] if hasattr(estimator, "feature_importances_") else []
    pd.DataFrame(importance).to_csv(out / "feature_group_importance.csv", index=False)

    scores = (per_trip[per_trip.model.eq(name)][["trip_id", "top1", "all_gps_top1", "observations"]]
              if selection["configuration"] == "tuned" else old_trips.rename(
                  columns={"conditional_top1":"top1", "size":"observations"}))
    output = dict(
        model=name, candidate_rows=len(dev), gps_groups=len(candidate_groups),
        positive_candidate_fraction=float(dev.label.mean()),
        groups_without_positive=int(candidate_groups["sum"].eq(0).sum()),
        candidates_per_gps_quantiles=candidate_groups["size"].quantile([0, .25, .5, .75, 1]).to_list(),
        candidate_recall_development=float(coverage.loc[dev.trip_id.unique()].retrieved.sum() /
                                           coverage.loc[dev.trip_id.unique()].aligned.sum()),
        baseline_fold_metrics=original["fold_metrics"],
        tuned_fold_metrics=folds[folds.model.eq(name)].to_dict("records"),
        selected_gap=selection["gap"],
        baseline_error_slices=pd.read_csv(frozen / "baseline_error_slices.csv").to_dict("records"),
        hardest_trips=scores.nsmallest(8, "top1").to_dict("records"),
        abs_correlation_pairs=pairs_out,
        feature_group_impurity_importance=importance,
        fold_one_train_validation=curves[0],
        learning_curve_status="Skipped extra subset refits for the final priority change")
    write_json(out / "generalization_report.json", output)
