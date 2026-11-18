"""Reference-route parsing, geometry, integrity checks, and comparison metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from reconstruction.road_graph import RoadGraph


Coordinate = tuple[float, float]


@dataclass(frozen=True)
class RouteAudit:
    arc_count: int
    invalid_arc_ids: tuple[int, ...]
    transition_count: int
    discontinuity_indices: tuple[int, ...]
    length_m: float

    @property
    def connected_transitions(self) -> int:
        return self.transition_count - len(self.discontinuity_indices)

    @property
    def continuity_ratio(self) -> float | None:
        if self.transition_count == 0:
            return None
        return self.connected_transitions / self.transition_count


@dataclass(frozen=True)
class RouteComparison:
    precision: float
    recall: float
    f1: float
    sequence_similarity: float
    exact_match: bool


def read_route(path: str | Path) -> list[int]:
    """Read an ordered route whose values are arc IDs."""
    route: list[int] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                route.append(int(value))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: arc ID must be an integer") from exc
    return route


def read_track(path: str | Path) -> list[tuple[float, float, float]]:
    """Read (longitude, latitude, timestamp) rows from a .track file."""
    observations: list[tuple[float, float, float]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.split()
            if len(values) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected longitude, latitude, timestamp"
                )
            try:
                longitude, latitude, timestamp = (float(value) for value in values)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: values must be numeric") from exc
            if not all(math.isfinite(value) for value in (longitude, latitude, timestamp)):
                raise ValueError(f"{path}:{line_number}: values must be finite")
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                raise ValueError(f"{path}:{line_number}: coordinates are out of range")
            observations.append((longitude, latitude, timestamp))
    return observations


def route_geometry(graph: RoadGraph, arc_ids: Iterable[int]) -> list[list[Coordinate]]:
    """Return route lines split at directed connectivity breaks."""
    lines: list[list[Coordinate]] = []
    current_line: list[Coordinate] = []
    previous_to_node: int | None = None

    for arc_id in arc_ids:
        arc = graph.arc(arc_id)
        if previous_to_node != arc.from_node:
            if current_line:
                lines.append(current_line)
            current_line = [graph.nodes[arc.from_node], graph.nodes[arc.to_node]]
        else:
            current_line.append(graph.nodes[arc.to_node])
        previous_to_node = arc.to_node

    if current_line:
        lines.append(current_line)
    return lines


def audit_route(graph: RoadGraph, arc_ids: Iterable[int]) -> RouteAudit:
    """Measure route references and connectivity without modifying the route."""
    route = list(arc_ids)
    invalid_arc_ids = tuple(
        arc_id for arc_id in route if not 0 <= arc_id < len(graph.arcs)
    )
    valid_arcs = [
        graph.arcs[arc_id] if 0 <= arc_id < len(graph.arcs) else None
        for arc_id in route
    ]
    valid_transitions = [
        (index, previous, current)
        for index, (previous, current) in enumerate(zip(valid_arcs, valid_arcs[1:]))
        if previous is not None and current is not None
    ]
    discontinuities = tuple(
        index
        for index, previous, current in valid_transitions
        if previous.to_node != current.from_node
    )
    transitions = len(valid_transitions)
    length_m = sum(arc.length_m for arc in valid_arcs if arc is not None)
    return RouteAudit(
        arc_count=len(route),
        invalid_arc_ids=invalid_arc_ids,
        transition_count=transitions,
        discontinuity_indices=discontinuities,
        length_m=length_m,
    )


def compare_routes(
    predicted_arc_ids: Iterable[int], reference_arc_ids: Iterable[int]
) -> RouteComparison:
    """Compare routes using arc multiplicity and edit-distance sequence similarity."""
    predicted = list(predicted_arc_ids)
    reference = list(reference_arc_ids)
    common_count = sum((Counter(predicted) & Counter(reference)).values())
    precision = common_count / len(predicted) if predicted else float(not reference)
    recall = common_count / len(reference) if reference else float(not predicted)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    denominator = max(len(predicted), len(reference))
    similarity = (
        1 - _edit_distance(predicted, reference) / denominator
        if denominator
        else 1.0
    )
    return RouteComparison(
        precision=precision,
        recall=recall,
        f1=f1,
        sequence_similarity=similarity,
        exact_match=predicted == reference,
    )


def write_route_geojson(
    graph: RoadGraph, arc_ids: Iterable[int], path: str | Path
) -> None:
    """Write a route as GeoJSON, retaining separate lines at connectivity breaks."""
    lines = route_geometry(graph, arc_ids)
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "MultiLineString",
            "coordinates": lines,
        },
    }
    Path(path).write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2),
        encoding="utf-8",
    )


def _edit_distance(first: list[int], second: list[int]) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for first_index, first_value in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_value in enumerate(second, start=1):
            current.append(
                min(
                    previous[second_index] + 1,
                    current[second_index - 1] + 1,
                    previous[second_index - 1] + (first_value != second_value),
                )
            )
        previous = current
    return previous[-1]
