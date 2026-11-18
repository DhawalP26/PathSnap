"""Directed road-graph loading and routing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path


@dataclass(frozen=True)
class Arc:
    arc_id: int
    from_node: int
    to_node: int
    length_m: float


@dataclass
class RoadGraph:
    nodes: list[tuple[float, float]]
    arcs: list[Arc]

    def __post_init__(self) -> None:
        self._outgoing: list[list[Arc]] = [[] for _ in self.nodes]
        self._undirected: list[set[int]] = [set() for _ in self.nodes]
        for arc in self.arcs:
            self._outgoing[arc.from_node].append(arc)
            self._undirected[arc.from_node].add(arc.to_node)
            self._undirected[arc.to_node].add(arc.from_node)

    @classmethod
    def from_files(cls, nodes_path: str | Path, arcs_path: str | Path) -> RoadGraph:
        nodes: list[tuple[float, float]] = []
        with Path(nodes_path).open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                values = line.split()
                if len(values) != 2:
                    raise ValueError(f"{nodes_path}:{line_number}: expected longitude and latitude")
                longitude, latitude = _parse_floats(values, nodes_path, line_number)
                if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                    raise ValueError(f"{nodes_path}:{line_number}: coordinates are out of range")
                nodes.append((longitude, latitude))

        if not nodes:
            raise ValueError(f"{nodes_path}: no nodes found")

        arcs: list[Arc] = []
        with Path(arcs_path).open(encoding="utf-8") as file:
            for arc_id, line in enumerate(file):
                values = line.split()
                if len(values) != 2:
                    raise ValueError(f"{arcs_path}:{arc_id + 1}: expected from_node and to_node")
                try:
                    from_node, to_node = (int(value) for value in values)
                except ValueError as exc:
                    raise ValueError(f"{arcs_path}:{arc_id + 1}: node IDs must be integers") from exc
                if not (0 <= from_node < len(nodes) and 0 <= to_node < len(nodes)):
                    raise ValueError(f"{arcs_path}:{arc_id + 1}: node ID is outside the node table")
                length_m = _haversine_m(nodes[from_node], nodes[to_node])
                arcs.append(Arc(arc_id, from_node, to_node, length_m))

        return cls(nodes, arcs)

    def outgoing_arcs(self, node_id: int) -> tuple[Arc, ...]:
        self._validate_node(node_id)
        return tuple(self._outgoing[node_id])

    def arc(self, arc_id: int) -> Arc:
        if not 0 <= arc_id < len(self.arcs):
            raise IndexError(f"Unknown arc ID: {arc_id}")
        return self.arcs[arc_id]

    def shortest_path(
        self,
        start_node: int,
        end_node: int,
        *,
        max_distance_m: float | None = None,
    ) -> tuple[list[int], float]:
        """Return directed shortest-path arc IDs and distance in meters.

        If max_distance_m is supplied, stop once no path within that distance
        can reach the destination.
        """
        self._validate_node(start_node)
        self._validate_node(end_node)
        if max_distance_m is not None and (
            not math.isfinite(max_distance_m) or max_distance_m < 0
        ):
            raise ValueError("max_distance_m must be finite and non-negative")
        if start_node == end_node:
            return [], 0.0

        distances = [math.inf] * len(self.nodes)
        previous: list[tuple[int, int] | None] = [None] * len(self.nodes)
        distances[start_node] = 0.0
        queue = [(0.0, start_node)]

        while queue:
            distance, node_id = heapq.heappop(queue)
            if distance != distances[node_id]:
                continue
            if max_distance_m is not None and distance > max_distance_m:
                break
            if node_id == end_node:
                break
            for arc in self._outgoing[node_id]:
                candidate_distance = distance + arc.length_m
                if max_distance_m is not None and candidate_distance > max_distance_m:
                    continue
                if candidate_distance < distances[arc.to_node]:
                    distances[arc.to_node] = candidate_distance
                    previous[arc.to_node] = (node_id, arc.arc_id)
                    heapq.heappush(queue, (candidate_distance, arc.to_node))

        if math.isinf(distances[end_node]):
            raise ValueError(f"No directed path from node {start_node} to node {end_node}")

        path: list[int] = []
        node_id = end_node
        while node_id != start_node:
            step = previous[node_id]
            if step is None:
                raise RuntimeError("Shortest-path predecessor chain is incomplete")
            node_id, arc_id = step
            path.append(arc_id)
        path.reverse()
        return path, distances[end_node]

    def weakly_connected_components(self) -> list[set[int]]:
        """Return connected components when arc directions are ignored."""
        remaining = set(range(len(self.nodes)))
        components: list[set[int]] = []
        while remaining:
            start = remaining.pop()
            component = {start}
            stack = [start]
            while stack:
                node_id = stack.pop()
                neighbors = self._undirected[node_id] & remaining
                remaining.difference_update(neighbors)
                component.update(neighbors)
                stack.extend(neighbors)
            components.append(component)
        return components

    def _validate_node(self, node_id: int) -> None:
        if not isinstance(node_id, int) or not 0 <= node_id < len(self.nodes):
            raise IndexError(f"Unknown node ID: {node_id}")


def _parse_floats(values: list[str], path: str | Path, line_number: int) -> tuple[float, float]:
    try:
        longitude, latitude = (float(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: coordinates must be numbers") from exc
    if not (math.isfinite(longitude) and math.isfinite(latitude)):
        raise ValueError(f"{path}:{line_number}: coordinates must be finite")
    return longitude, latitude


def _haversine_m(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    longitude_1, latitude_1 = map(math.radians, first)
    longitude_2, latitude_2 = map(math.radians, second)
    delta_longitude = longitude_2 - longitude_1
    delta_latitude = latitude_2 - latitude_1
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_1)
        * math.cos(latitude_2)
        * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * 6_371_000 * math.asin(min(1.0, math.sqrt(haversine)))
