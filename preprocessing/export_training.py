"""Export all supervised training trips and report alignment coverage."""
import argparse
from pathlib import Path

import pandas as pd

from .data_loader import trip_ids
from .pipeline import prepare_training_trip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/member1"))
    parser.add_argument("--radius-m", type=float, default=75)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-distance-m", type=float, default=75)
    args = parser.parse_args()
    ids = trip_ids("train")
    if not ids:
        parser.error("No training trips found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    summaries = []
    for trip_id in ids:
        table, diagnostics = prepare_training_trip(
            trip_id, radius_m=args.radius_m, top_k=args.top_k,
            max_distance_m=args.max_distance_m)
        aligned = int(diagnostics.aligned_arc_id.ge(0).sum())
        retrieved = int(diagnostics.candidate_retrieved.sum())
        summaries.append({"trip_id": trip_id, "observations": len(diagnostics),
                          "aligned": aligned, "retrieved": retrieved,
                          "candidate_recall": retrieved / aligned if aligned else None,
                          "candidate_rows": len(table)})
        tables.append(table)
        print(f"{trip_id}: {retrieved}/{aligned} aligned observations retrieved", flush=True)
    features = pd.concat(tables, ignore_index=True)
    if features.empty:
        raise ValueError("No labeled candidates. Check alignment and candidate coverage.")
    features.to_csv(args.out_dir / "train_features.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.out_dir / "training_coverage.csv", index=False)


if __name__ == "__main__":
    main()
