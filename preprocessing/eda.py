"""Generate offline inventory, distribution plots and a selected-trip Folium map."""
import argparse
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .candidate_generator import RoadNetwork
from .data_loader import load_reference, load_trip, partition_path, trip_ids
from .feature_engineering import motion


def plot(path, title, x_label, series, *, bars=False):
    fig, ax = plt.subplots(figsize=(9, 4))
    if bars:
        ax.bar(np.arange(len(series)), series)
    else:
        ax.hist(series, bins=50)
    ax.set(title=title, xlabel=x_label, ylabel="GPS observations" if bars else "Count")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_map(path, trip, route, road_ids):
    gps = trip.track[:, [1, 0]].tolist()
    map_ = folium.Map(location=gps[len(gps) // 2], zoom_start=13)
    roads = folium.FeatureGroup(name="Roads near GPS").add_to(map_)
    reference = folium.FeatureGroup(name="Reference route (offline)").add_to(map_)
    for arc in trip.arcs[sorted(road_ids)]:
        folium.PolyLine(trip.nodes[arc][:, [1, 0]].tolist(), color="#53667a",
                        weight=2, opacity=.45).add_to(roads)
    if route is not None:
        for arc in trip.arcs[route]:
            folium.PolyLine(trip.nodes[arc][:, [1, 0]].tolist(), color="#159068",
                            weight=4).add_to(reference)
    points = folium.FeatureGroup(name="GPS observations").add_to(map_)
    folium.PolyLine(gps, color="#b93928", weight=2).add_to(points)
    stride = max(1, (len(gps) + 599) // 600)
    for i in range(0, len(gps), stride):
        folium.CircleMarker(gps[i], radius=2, color="#b93928",
                           tooltip=f"GPS {i}").add_to(points)
    map_.fit_bounds([[min(trip.track[:, 1]), min(trip.track[:, 0])],
                     [max(trip.track[:, 1]), max(trip.track[:, 0])]])
    folium.LayerControl().add_to(map_)
    map_.save(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/member1/eda"))
    parser.add_argument("--partition", nargs="+", default=["train", "student_test"])
    parser.add_argument("--map-partition", default="train")
    parser.add_argument("--trip", default="00000005")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory, gaps, speeds, lengths, nearest = [], [], [], [], []
    map_written = False
    for partition in args.partition:
        base = partition_path(partition)
        for trip_id in trip_ids(partition):
            trip = load_trip(trip_id, partition)
            reference_file = base / trip_id / f"{trip_id}.route"
            route = (load_reference(trip_id, partition, len(trip.arcs))
                     if reference_file.is_file() else None)
            net = RoadNetwork(trip)
            points = net.project(trip.track)
            _, speed, gap, _ = motion(trip.track, points)
            draw = (partition, trip_id) == (args.map_partition, args.trip)
            road_ids = set()
            closest = np.full(len(points), np.nan)
            for i, (ids, distances, _) in enumerate(net.candidates(points, 75, 8 if draw else 1)):
                if len(ids):
                    closest[i] = distances[0]
                    if draw:
                        road_ids.update(ids)
            graph = csr_matrix((np.ones(len(trip.arcs), dtype=bool), tuple(trip.arcs.T)),
                               shape=(len(trip.nodes), len(trip.nodes)))
            weak = connected_components(graph, directed=True, connection="weak", return_labels=False)
            strong = connected_components(graph, directed=True, connection="strong", return_labels=False)
            inventory.append(dict(
                partition=partition, trip_id=trip_id, gps_points=len(points),
                nodes=len(trip.nodes), directed_arcs=len(trip.arcs), weak_components=weak,
                strong_components=strong, route_arcs=len(route) if route is not None else None,
                route_discontinuities=int(np.count_nonzero(
                    trip.arcs[route[:-1], 1] != trip.arcs[route[1:], 0])) if route is not None else None,
                min_lon=trip.track[:, 0].min(), max_lon=trip.track[:, 0].max(),
                min_lat=trip.track[:, 1].min(), max_lat=trip.track[:, 1].max(),
                duplicate_gps_rows=len(points) - len(np.unique(trip.track, axis=0)),
                repeated_timestamps=int(np.count_nonzero(gap[1:] == 0)),
                gaps_over_10s=int(np.count_nonzero(gap > 10)),
                speeds_over_50mps=int(np.count_nonzero(speed > 50)),
                zero_length_arcs=int(np.count_nonzero(net.lengths == 0)),
                no_road_within_75m=int(np.count_nonzero(np.isnan(closest)))))
            gaps.append(gap[1:])
            speeds.append(speed[np.isfinite(speed)])
            lengths.append(net.lengths.astype(np.float32))
            nearest.append(closest[np.isfinite(closest)])
            if draw:
                draw_map(args.out_dir / f"{partition}_{trip_id}.html", trip, route, road_ids)
                map_written = True
    if not inventory:
        parser.error("No trips found")
    if not map_written:
        parser.error("Selected map trip was not found in the requested partitions")
    frame = pd.DataFrame(inventory)
    frame.to_csv(args.out_dir / "dataset_inventory.csv", index=False)
    plot(args.out_dir / "gps_counts.png", "GPS points per trip", "Trip order",
         frame.gps_points, bars=True)
    distributions = [
        ("time_gaps", "GPS sampling intervals", "Seconds", gaps),
        ("speeds", "GPS speeds", "Meters per second", speeds),
        ("segment_lengths", "Directed arc lengths", "Meters", lengths),
        ("road_distance", "Nearest road within 75 m", "Meters", nearest),
    ]
    summary = ["# PathSnap dataset EDA", ""]
    for partition, trips in frame.groupby("partition"):
        summary.append(f"- {partition} contains {len(trips)} trips and "
                       f"{trips.gps_points.sum():,} GPS observations.")
    for filename, title, unit, groups in distributions:
        values = np.concatenate(groups)
        plot(args.out_dir / f"{filename}.png", title, unit, values)
        if len(values):
            median, high = np.percentile(values, [50, 95])
            summary.append(f"- {title} has median {median:.2f} and 95th percentile {high:.2f} "
                           f"({unit.lower()}).")
    summary.extend([
        f"- {frame.duplicate_gps_rows.sum():,} duplicate GPS rows, "
        f"{frame.repeated_timestamps.sum():,} repeated timestamps, "
        f"{frame.gaps_over_10s.sum():,} gaps over 10 seconds, and "
        f"{frame.speeds_over_50mps.sum():,} speeds above 50 m/s.",
        f"- {frame.no_road_within_75m.sum():,} GPS observations had no road within 75 meters.",
        "",
        "GPS rows are retained in stable timestamp order. Speed is missing when a positive "
        "time interval is unavailable. Invalid input rows stop processing with file and line "
        "context. The inventory records route discontinuities and zero-length arcs.",
    ])
    (args.out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
