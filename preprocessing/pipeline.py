"""Public API for preparing candidate features and supervised examples."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidate_generator import RoadNetwork
from .data_loader import Trip, load_reference, load_trip
from .feature_engineering import angle_difference, feature_table, motion


@dataclass
class PreparedTrip:
    trip: Trip
    network: RoadNetwork
    features: pd.DataFrame


def prepare_trip(trip_id, partition="student_test", *, trip_dir=None, radius_m=75, top_k=8):
    """Generate prediction-time features using only track, nodes and arcs."""
    trip = load_trip(trip_id, partition, trip_dir=trip_dir)
    network = RoadNetwork(trip)
    points = network.project(trip.track)
    features = feature_table(trip, network, points, motion(trip.track, points), radius_m, top_k)
    return PreparedTrip(trip, network, features)


def align_reference(network, points, headings, route, *, max_distance_m=75):
    """Minimum-cost nondecreasing path through ordered reference arc occurrences."""
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("max_distance_m must be finite and positive")
    positions = np.arange(len(route))
    skip_cost = 2.0
    back = np.empty((len(points), len(route)), dtype=np.int32)
    previous = np.zeros(len(route))
    bearings = network.headings[route]
    for i, point in enumerate(points):
        distance, _ = network.distances(point, route)
        difference = np.nan_to_num(angle_difference(headings[i], bearings), nan=0)
        emission = np.minimum(distance + difference / 18, max_distance_m + 10)
        # A cumulative minimum makes all forward transitions linear in route length.
        if i:
            adjusted = previous - positions * skip_cost
            minima = np.minimum.accumulate(adjusted)
            records = adjusted < np.r_[np.inf, minima[:-1]]
            best_prefix = np.maximum.accumulate(np.where(records, positions, 0))
            advance = np.r_[np.inf, minima[:-1] + positions[1:] * skip_cost - skip_cost]
            use_advance = advance < previous
            back[i] = np.where(use_advance, np.r_[0, best_prefix[:-1]], positions)
            previous = emission + np.minimum(previous, advance)
        else:
            previous = emission
    current = int(np.argmin(previous))
    matched = np.full(len(points), -1, dtype=int)
    for i in range(len(points) - 1, -1, -1):
        arc = route[current]
        distance, _ = network.distances(points[i], route[current:current + 1])
        if distance[0] <= max_distance_m:
            matched[i] = arc
        if i:
            current = back[i, current]
    return matched


def prepare_training_trip(trip_id, *, radius_m=75, top_k=8, max_distance_m=75):
    """Return binary-labeled candidates and per-observation alignment diagnostics."""
    prepared = prepare_trip(trip_id, "train", radius_m=radius_m, top_k=top_k)
    route = load_reference(trip_id, "train", len(prepared.network.arcs))
    points = prepared.network.project(prepared.trip.track)
    headings = motion(prepared.trip.track, points)[0]
    matched = align_reference(prepared.network, points, headings, route,
                              max_distance_m=max_distance_m)
    table = prepared.features
    indices = table["gps_index"].to_numpy(dtype=int)
    positive = table["arc_id"].to_numpy(dtype=int) == matched[indices]
    retrieved = np.zeros(len(points), dtype=bool)
    retrieved[indices[positive]] = True
    keep = retrieved[indices]
    labeled = table.loc[keep].assign(label=positive[keep].astype(np.int8))
    diagnostics = pd.DataFrame({"gps_index": np.arange(len(points)), "aligned_arc_id": matched,
                                "candidate_retrieved": retrieved})
    return labeled, diagnostics
