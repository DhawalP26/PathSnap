"""Development-only generalization diagnostics for the frozen final model."""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from models.final_review_tuning import COLUMNS, winners, write_json
from models.review1_extended_features import LATTICE, TEMPORAL
from models.review1_features import GROUPS
from models.schema import FEATURES


def analyze(dev, splits, coverage, model, frozen, out, seed):
    rng = np.random.default_rng(seed)
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
    train, validation = splits[0]
    training, held = dev.iloc[train], dev.iloc[validation]
    order = rng.permutation(training.trip_id.unique())
    curves = []
    fitted = None
    for fraction in [.25, .50, 1.0]:
        n = max(1, round(len(order) * fraction))
        subset = training[training.trip_id.isin(order[:n])]
        fitted = clone(model).fit(subset[COLUMNS], subset.label)
        curves.append({"training_trips": n,
                       "train_top1": float(winners(subset, fitted.predict_proba(subset[COLUMNS])[:, 1]).label.mean()),
                       "validation_top1": float(winners(held, fitted.predict_proba(held[COLUMNS])[:, 1]).label.mean())})
    pd.DataFrame(curves).to_csv(out / "learning_curve.csv", index=False)

    groups = {"base": FEATURES, **GROUPS, "extended_temporal": TEMPORAL, "lattice": LATTICE}
    validation_trips = rng.permutation(held.trip_id.unique())[:8]
    subset = held[held.trip_id.isin(validation_trips)]
    x = subset[COLUMNS].copy()
    top = winners(subset, fitted.predict_proba(x)[:, 1]).label.mean()
    importance = []
    for key, cols in groups.items():
        shuffled = x.copy()
        shuffled.loc[:, cols] = x[cols].to_numpy()[rng.permutation(len(x))]
        new_score = winners(subset, fitted.predict_proba(shuffled)[:, 1]).label.mean()
        importance.append(dict(feature_group=key, top1_drop=float(top - new_score)))
    pd.DataFrame(importance).sort_values("top1_drop", ascending=False).to_csv(
        out / "feature_group_importance.csv", index=False)

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
        validation_feature_group_permutation=importance,
        permutation_validation_trips=sorted(validation_trips),
        curve_validation_fold=1, learning_curve=curves)
    write_json(out / "generalization_report.json", output)
