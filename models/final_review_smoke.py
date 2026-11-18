"""Run reference-free production inference on accessible student trips."""
import argparse
import json
import os
from pathlib import Path
import time

from models.final_review_integration import create_predict_trip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predict = create_predict_trip(args.model)
    root = Path(os.environ["PATHSNAP_DATA_DIR"]) / "student_test"
    ids = sorted(p.name for p in root.iterdir() if p.is_dir() and len(p.name) == 8 and p.name.isdigit())
    outcomes = []
    for trip_id in ids:
        start = time.perf_counter()
        try:
            payload = predict(root / trip_id, trip_id)
            assert payload["route_coordinates"] and payload["predicted_arc_ids"]
            outcomes.append(dict(trip_id=trip_id, status="passed", **payload["metrics"],
                                 seconds=time.perf_counter()-start))
        except Exception as error:
            outcomes.append(dict(trip_id=trip_id, status="failed", error=str(error)))
        args.out.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        print(outcomes[-1], flush=True)
    if len(ids) != 10 or any(r["status"] != "passed" for r in outcomes):
        raise SystemExit("Expected ten successful student-test trips")


if __name__ == "__main__":
    main()
