"""Audit reference routes and network metadata for PathSnap partitions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reconstruction.road_graph import RoadGraph
from reconstruction.route_analysis import audit_route, read_route, read_track


def audit_trip(partition: str, trip_dir: Path) -> dict[str, object]:
    trip_id = trip_dir.name
    prefix = trip_dir / trip_id
    graph = RoadGraph.from_files(prefix.with_suffix(".nodes"), prefix.with_suffix(".arcs"))
    route = read_route(prefix.with_suffix(".route"))
    observations = read_track(prefix.with_suffix(".track"))
    audit = audit_route(graph, route)
    components = graph.weakly_connected_components()
    return {
        "partition": partition,
        "trip_id": trip_id,
        "observations": len(observations),
        "nodes": len(graph.nodes),
        "arcs": len(graph.arcs),
        "weak_components": len(components),
        "route_arcs": audit.arc_count,
        "invalid_route_arc_ids": len(audit.invalid_arc_ids),
        "route_discontinuities": len(audit.discontinuity_indices),
        "route_continuity": audit.continuity_ratio,
        "route_length_m": audit.length_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train/ and student_test/ (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/route_audit.csv"),
        help="CSV report path (default: ./results/route_audit.csv)",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for partition in ("train", "student_test"):
        partition_dir = args.data_dir / partition
        if not partition_dir.is_dir():
            raise FileNotFoundError(f"Dataset partition not found: {partition_dir}")
        trip_dirs = sorted(path for path in partition_dir.iterdir() if path.is_dir())
        rows.extend(audit_trip(partition, trip_dir) for trip_dir in trip_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for partition in ("train", "student_test"):
        partition_rows = [row for row in rows if row["partition"] == partition]
        invalid = sum(int(row["invalid_route_arc_ids"]) for row in partition_rows)
        breaks = sum(int(row["route_discontinuities"]) for row in partition_rows)
        print(
            f"{partition}: {len(partition_rows)} trips, "
            f"{invalid} invalid route arc IDs, {breaks} discontinuities"
        )
    print(f"CSV report: {args.output}")


if __name__ == "__main__":
    main()
