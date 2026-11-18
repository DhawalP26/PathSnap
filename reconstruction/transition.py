"""Directed transition paths and scores for Member 1 candidate rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence

from reconstruction.road_graph import RoadGraph, _haversine_m


@dataclass(frozen=True)
class ArcPosition:
    gps_index: int
    arc_id: int
    fraction: float
    segment_length_m: float
    gps_step_m: float
    time_gap_s: float


@dataclass(frozen=True)
class ArcPath:
    arc_ids: tuple[int, ...]
    distance_m: float


class CandidateTransitionScorer:
    """Score transitions between candidates using their positions on directed arcs.

    Candidate rows must use Member 1's feature schema. Provide the original
    (longitude, latitude, timestamp) track to score skipped GPS indices using
    their actual elapsed time and displacement.
    """

    def __init__(
        self,
        graph: RoadGraph,
        candidate_rows: Iterable[Mapping[str, object]] | object,
        *,
        track: Sequence[Sequence[float]] | None = None,
        beta_m: float = 50.0,
        max_speed_mps: float = 70.0,
        gps_tolerance_m: float = 30.0,
        gap_uncertainty_m_per_s: float = 1.5,
    ) -> None:
        for name, value in (
            ("beta_m", beta_m),
            ("max_speed_mps", max_speed_mps),
            ("gps_tolerance_m", gps_tolerance_m),
            ("gap_uncertainty_m_per_s", gap_uncertainty_m_per_s),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if beta_m == 0 or max_speed_mps == 0:
            raise ValueError("beta_m and max_speed_mps must be positive")

        self.graph = graph
        self.beta_m = beta_m
        self.max_speed_mps = max_speed_mps
        self.gps_tolerance_m = gps_tolerance_m
        self.gap_uncertainty_m_per_s = gap_uncertainty_m_per_s
        self.track = _validate_track(track) if track is not None else None
        self.candidates: dict[int, dict[int, ArcPosition]] = {}
        self._node_path_cache: dict[
            tuple[int, int, float], tuple[tuple[int, ...], float] | None
        ] = {}

        if hasattr(candidate_rows, "to_dict"):
            candidate_rows = candidate_rows.to_dict(orient="records")
        for row_number, row in enumerate(candidate_rows, start=1):
            position = _parse_position(row, row_number)
            graph.arc(position.arc_id)
            if position.arc_id in self.candidates.setdefault(position.gps_index, {}):
                raise ValueError(
                    f"Duplicate arc {position.arc_id} at gps_index {position.gps_index}"
                )
            self.candidates[position.gps_index][position.arc_id] = position

        if not self.candidates:
            raise ValueError("No candidate rows supplied")
        self.observation_indices = tuple(sorted(self.candidates))
        if self.track is not None and max(self.observation_indices) >= len(self.track):
            raise ValueError("Candidate gps_index is outside the supplied track")
        self._observation_positions = {
            gps_index: position
            for position, gps_index in enumerate(self.observation_indices)
        }

    def transition_path(
        self,
        previous_arc_id: int,
        current_arc_id: int,
        timestep: int,
    ) -> ArcPath | None:
        """Return a feasible directed path between candidate positions.

        Timestep is the index in observation_indices for the current candidate
        set; it matches the Viterbi transition callback convention.
        """
        previous_index, current_index = self._indices_for_timestep(timestep)
        previous = self._candidate(previous_index, previous_arc_id)
        current = self._candidate(current_index, current_arc_id)
        _, elapsed_s = self._gps_motion(previous, current)
        max_distance_m = self.max_speed_mps * elapsed_s + self.gps_tolerance_m
        return route_between_positions(
            self.graph,
            previous,
            current,
            max_distance_m=max_distance_m,
            node_path=self._shortest_node_path,
        )

    def log_probability(
        self,
        previous_arc_id: int,
        current_arc_id: int,
        timestep: int,
    ) -> float:
        """Return a Laplace log transition score; unreachable paths are impossible."""
        previous_index, current_index = self._indices_for_timestep(timestep)
        previous = self._candidate(previous_index, previous_arc_id)
        current = self._candidate(current_index, current_arc_id)
        gps_distance_m, elapsed_s = self._gps_motion(previous, current)
        max_distance_m = self.max_speed_mps * elapsed_s + self.gps_tolerance_m
        path = route_between_positions(
            self.graph,
            previous,
            current,
            max_distance_m=max_distance_m,
            node_path=self._shortest_node_path,
        )
        if path is None:
            return -math.inf

        scale_m = self.beta_m + self.gap_uncertainty_m_per_s * max(elapsed_s - 1.0, 0)
        return -abs(path.distance_m - gps_distance_m) / scale_m - math.log(2 * scale_m)

    def _indices_for_timestep(self, timestep: int) -> tuple[int, int]:
        if not isinstance(timestep, int) or not 1 <= timestep < len(self.observation_indices):
            raise IndexError(f"Unknown transition timestep: {timestep}")
        return self.observation_indices[timestep - 1], self.observation_indices[timestep]

    def _candidate(self, gps_index: int, arc_id: int) -> ArcPosition:
        try:
            return self.candidates[gps_index][arc_id]
        except KeyError as exc:
            raise ValueError(
                f"Arc {arc_id} is not a candidate at gps_index {gps_index}"
            ) from exc

    def _gps_motion(
        self, previous: ArcPosition, current: ArcPosition
    ) -> tuple[float, float]:
        if self.track is None:
            if current.gps_index != previous.gps_index + 1:
                raise ValueError(
                    "Track data is required when candidate observations skip GPS indices"
                )
            if not math.isfinite(current.time_gap_s) or current.time_gap_s <= 0:
                raise ValueError(
                    f"Invalid time_gap_s at gps_index {current.gps_index}"
                )
            return current.gps_step_m, current.time_gap_s

        previous_observation = self.track[previous.gps_index]
        current_observation = self.track[current.gps_index]
        elapsed_s = current_observation[2] - previous_observation[2]
        if not math.isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError(
                f"Non-increasing timestamps between gps_index "
                f"{previous.gps_index} and {current.gps_index}"
            )
        gps_distance_m = _haversine_m(
            (previous_observation[0], previous_observation[1]),
            (current_observation[0], current_observation[1]),
        )
        return gps_distance_m, elapsed_s

    def _shortest_node_path(
        self, start_node: int, end_node: int, max_distance_m: float
    ) -> tuple[tuple[int, ...], float] | None:
        key = (start_node, end_node, round(max_distance_m, 6))
        if key not in self._node_path_cache:
            try:
                arc_ids, distance_m = self.graph.shortest_path(
                    start_node, end_node, max_distance_m=max_distance_m
                )
            except ValueError:
                self._node_path_cache[key] = None
            else:
                self._node_path_cache[key] = (tuple(arc_ids), distance_m)
        return self._node_path_cache[key]


def route_between_positions(
    graph: RoadGraph,
    previous: ArcPosition,
    current: ArcPosition,
    *,
    max_distance_m: float | None = None,
    node_path=None,
) -> ArcPath | None:
    """Find a shortest directed route between two positions along road arcs."""
    for candidate in (previous, current):
        if not 0 <= candidate.fraction <= 1 or candidate.segment_length_m < 0:
            raise ValueError("Candidate positions require a fraction in [0, 1] and non-negative length")
    if max_distance_m is not None and (
        not math.isfinite(max_distance_m) or max_distance_m < 0
    ):
        raise ValueError("max_distance_m must be finite and non-negative")

    previous_arc = graph.arc(previous.arc_id)
    current_arc = graph.arc(current.arc_id)
    previous_length = previous.segment_length_m
    current_length = current.segment_length_m

    if previous.arc_id == current.arc_id and current.fraction >= previous.fraction:
        distance_m = (current.fraction - previous.fraction) * previous_length
        if max_distance_m is not None and distance_m > max_distance_m:
            return None
        arc_ids = (previous.arc_id,) if distance_m > 0 else ()
        return ArcPath(arc_ids, distance_m)

    start_partial_m = (1 - previous.fraction) * previous_length
    end_partial_m = current.fraction * current_length
    connector_limit = (
        None
        if max_distance_m is None
        else max_distance_m - start_partial_m - end_partial_m
    )
    if connector_limit is not None and connector_limit < 0:
        return None

    if node_path is None:
        def find_node_path(start: int, end: int, limit: float | None):
            try:
                if limit is None:
                    ids, distance = graph.shortest_path(start, end)
                else:
                    ids, distance = graph.shortest_path(
                        start, end, max_distance_m=limit
                    )
            except ValueError:
                return None
            return tuple(ids), distance
    else:
        def find_node_path(start: int, end: int, limit: float | None):
            return node_path(start, end, math.inf if limit is None else limit)

    connector = find_node_path(
        previous_arc.to_node, current_arc.from_node, connector_limit
    )
    if connector is None:
        return None
    connector_arc_ids, connector_distance_m = connector
    distance_m = start_partial_m + connector_distance_m + end_partial_m
    if max_distance_m is not None and distance_m > max_distance_m:
        return None

    return ArcPath(
        (previous.arc_id, *connector_arc_ids, current.arc_id),
        distance_m,
    )


def expand_candidate_sequence(
    scorer: CandidateTransitionScorer,
    observation_indices: Sequence[int],
    arc_ids: Sequence[int],
) -> list[int]:
    """Insert directed intermediate arcs between a sequence of selected candidates."""
    if len(observation_indices) != len(arc_ids):
        raise ValueError("observation_indices and arc_ids must have equal lengths")
    if not arc_ids:
        return []

    expected = tuple(scorer.observation_indices)
    selected_indices = tuple(observation_indices)
    if not selected_indices:
        raise ValueError("A non-empty arc sequence requires observation indices")
    try:
        start_position = expected.index(selected_indices[0])
    except ValueError as exc:
        raise ValueError(
            "Selected observation indices must come from the candidate transition scorer"
        ) from exc
    if expected[start_position : start_position + len(selected_indices)] != selected_indices:
        raise ValueError(
            "Selected observation indices must be a contiguous subsequence of "
            "the candidate transition scorer"
        )

    route = [arc_ids[0]]
    for local_timestep, (previous_arc, current_arc) in enumerate(
        zip(arc_ids, arc_ids[1:]), start=1
    ):
        timestep = start_position + local_timestep
        path = scorer.transition_path(previous_arc, current_arc, timestep)
        if path is None:
            raise ValueError(
                f"No feasible directed transition from arc {previous_arc} "
                f"to arc {current_arc} at gps_index {observation_indices[timestep]}"
            )
        if path.arc_ids:
            route.extend(path.arc_ids[1:] if path.arc_ids[0] == route[-1] else path.arc_ids)
    return route


def _parse_position(row: Mapping[str, object], row_number: int) -> ArcPosition:
    required = {
        "gps_index",
        "arc_id",
        "segment_fraction",
        "segment_length_m",
        "gps_step_m",
        "time_gap_s",
    }
    missing = required - row.keys()
    if missing:
        raise ValueError(
            f"Candidate row {row_number} is missing fields: {sorted(missing)}"
        )

    gps_index = _integer(row["gps_index"], "gps_index", row_number)
    arc_id = _integer(row["arc_id"], "arc_id", row_number)
    fraction = _finite_number(row["segment_fraction"], "segment_fraction", row_number)
    segment_length_m = _finite_number(
        row["segment_length_m"], "segment_length_m", row_number
    )
    gps_step_m = _finite_number(row["gps_step_m"], "gps_step_m", row_number)
    time_gap_s = _finite_number(row["time_gap_s"], "time_gap_s", row_number)
    if not 0 <= fraction <= 1:
        raise ValueError(f"segment_fraction in row {row_number} must be between 0 and 1")
    if segment_length_m < 0 or gps_step_m < 0 or time_gap_s < 0:
        raise ValueError(f"Candidate row {row_number} contains a negative distance or time")
    return ArcPosition(
        gps_index,
        arc_id,
        fraction,
        segment_length_m,
        gps_step_m,
        time_gap_s,
    )


def _integer(value: object, field: str, row_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} in row {row_number} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} in row {row_number} must be non-negative")
    return result


def _finite_number(value: object, field: str, row_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} in row {row_number} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} in row {row_number} must be finite")
    return number


def _validate_track(
    track: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    observations: list[tuple[float, float, float]] = []
    for index, row in enumerate(track):
        if len(row) != 3:
            raise ValueError(f"Track row {index} must have longitude, latitude, timestamp")
        longitude, latitude, timestamp = (
            _finite_number(value, "track value", index + 1) for value in row
        )
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError(f"Track coordinates in row {index} are out of range")
        observations.append((longitude, latitude, timestamp))
    if not observations:
        raise ValueError("Track must contain at least one observation")
    return tuple(observations)
