"""Audit baseline misranking by overlapping geometry and motion proxies."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from preprocessing.data_loader import load_trip
from models.schema import FEATURES, ID_COLUMNS, read_training
from models.review1_features import VARIANTS
from models.train_models import folds_for, make_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.baseline / "evaluation_summary.json").read_text())
    dev_ids = set(report["development_trips"])
    raw = read_training(args.train)
    raw = raw[raw.trip_id.isin(dev_ids)]
    columns = FEATURES + VARIANTS["full_context"]
    parts, contexts = [], []
    for ident, rows in raw.groupby("trip_id", sort=False):
        f = pd.read_pickle(args.out / "feature_cache" / f"{ident}.pkl")
        parts.append(rows.merge(f[ID_COLUMNS + VARIANTS["full_context"]], on=ID_COLUMNS, validate="one_to_one"))
        ordered = f.sort_values(["distance_m", "arc_id"], kind="stable")
        groups = ordered.groupby("gps_index", sort=False)
        pair = groups.head(2)
        first = pair.drop_duplicates("gps_index").set_index("gps_index")
        second = pair[pair.duplicated("gps_index")].set_index("gps_index")
        first = first.loc[second.index]
        trip = load_trip(ident, "train")
        one, two = trip.arcs[first.arc_id], trip.arcs[second.arc_id]
        shared = (one[:, :, None] == two[:, None, :]).any(axis=(1, 2))
        a = trip.nodes[one[:, 1]] - trip.nodes[one[:, 0]]
        b = trip.nodes[two[:, 1]] - trip.nodes[two[:, 0]]
        a[:, 0] *= np.cos(np.radians(trip.nodes[one[:, 0], 1]))
        b[:, 0] *= np.cos(np.radians(trip.nodes[two[:, 0], 1]))
        cosine = np.divide(np.abs(np.einsum("ij,ij->i",a,b)), np.linalg.norm(a, axis=1)*np.linalg.norm(b,axis=1),
                           out=np.zeros(len(a)), where=(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)>0))
        contexts.append(pd.DataFrame(dict(
            trip_id=ident, gps_index=first.index,
            intersection_proxy=shared & ((first.endpoint_distance_m.to_numpy() <= 12) |
                                         (second.endpoint_distance_m.to_numpy() <= 12)),
            parallel_proxy=(~shared) & (cosine >= np.cos(np.radians(15))),
            turn_proxy=first.heading_turn.to_numpy() >= 30,
            ambiguous_proxy=second.distance_m.to_numpy()-first.distance_m.to_numpy() <= 1,
        )))
    frame = pd.concat(parts, ignore_index=True)
    for index, (train, valid) in enumerate(folds_for(frame, report["configuration"]["folds"])):
        model = clone(make_models(report["configuration"]["seed"])["HistGradientBoosting"])
        model.fit(frame.iloc[train][columns], frame.label.iloc[train])
        frame.loc[valid, "baseline_score"] = model.predict_proba(frame.iloc[valid][columns])[:, 1]
    winners = (frame.sort_values(["baseline_score", "arc_id"], ascending=[False,True])
               .drop_duplicates(ID_COLUMNS[:2])[ID_COLUMNS[:2]+["label"]])
    audit = winners.merge(pd.concat(contexts), on=ID_COLUMNS[:2], how="left")
    recorded = pd.read_csv(args.out / 'ablation.csv').set_index('variant')
    if not np.isclose(winners.label.mean(), recorded.loc['full_context', 'top1_accuracy']):
        raise ValueError('Baseline audit did not reproduce the recorded top-1')
    rows = []
    for context in ("intersection_proxy", "parallel_proxy", "turn_proxy", "ambiguous_proxy"):
        group = audit[audit[context].eq(True)]
        rows.append(dict(context=context, observations=len(group), wrong=int(group.label.eq(0).sum()),
                         top1=group.label.mean()))
    pd.DataFrame(rows).to_csv(args.out / "baseline_error_slices.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))



if __name__ == "__main__":
    main()
