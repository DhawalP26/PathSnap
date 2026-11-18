"""Reference-free HTTP smoke test for all configured student-test trips.

Run after FastAPI starts:
python -m scripts.smoke_student_api --api http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import requests


def run(api: str, output: Path) -> list[dict]:
    api = api.rstrip("/")
    session = requests.Session()
    trips_response = session.get(f"{api}/trips", timeout=20)
    trips_response.raise_for_status()
    data = trips_response.json()
    if data.get("partition") != "student_test":
        raise RuntimeError("Refusing to smoke test a non-student_test partition")
    trips = data.get("trips", [])
    if len(trips) != 10:
        raise RuntimeError(f"Expected 10 student-test trips, found {len(trips)}")
    outcomes = []
    for trip in trips:
        start = time.perf_counter()
        try:
            response = session.post(f"{api}/predict", json={"trip_id": trip}, timeout=300)
            response.raise_for_status()
            result = response.json()
            lines = result["route_coordinates"]
            if not isinstance(lines, list) or not lines or not result["predicted_arc_ids"]:
                raise ValueError("No reconstructed route in prediction")
            if any(not isinstance(line, list) for line in lines):
                raise ValueError("Route coordinates are not nested polylines")
            if any(key in result for key in ("reference_route", "reference_arc_ids", "ground_truth")):
                raise ValueError("Reference labels leaked in API response")
            outcome = {
                "trip_id": trip, "status": "passed", "seconds": round(time.perf_counter()-start, 3),
                "route_polylines": len(lines), "route_breaks": result.get("route_breaks", 0),
                "predicted_arcs": len(result["predicted_arc_ids"]), "metrics": result.get("metrics", {}),
            }
        except Exception as exc:
            outcome = {"trip_id": trip, "status": "failed", "error": str(exc)}
        outcomes.append(outcome)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
        print(json.dumps(outcome), flush=True)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=Path("results/member4/student_test_api_smoke.json"))
    args = parser.parse_args()
    outcomes = run(args.api, args.out)
    sys.exit(0 if len(outcomes) == 10 and all(r["status"] == "passed" for r in outcomes) else 1)


if __name__ == "__main__":
    main()
