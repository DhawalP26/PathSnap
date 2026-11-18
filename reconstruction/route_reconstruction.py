"""Reconstruct directed routes from scored candidate road segments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
import sys
from typing import Callable, Iterable, Mapping, Sequence

from reconstruction.hmm import candidate_scores_to_emissions
from reconstruction.road_graph import RoadGraph
from reconstruction.route_analysis import route_geometry
from reconstruction.transition import (
    CandidateTransitionScorer,
    expand_candidate_sequence,
)
from reconstruction.viterbi import decode_with_restarts


@dataclass(frozen=True)
class ReconstructedRoute:
    observation_indices: tuple[int, ...]
    predicted_arc_ids: tuple[int, ...]
    route_arc_ids: tuple[int, ...]
    log_probability: float
    segment_ranges: tuple[tuple[int, int], ...]
    route_segments: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PreparedTripPrediction:
    trip_id: str
    reconstruction: ReconstructedRoute | None
    route_lines: tuple[tuple[tuple[float, float], ...], ...]

    def api_response(self) -> dict[str, object]:
        """Return the Member 4 response shape, preserving disconnected lines."""
        return {
            "trip_id": self.trip_id,
            "predicted_arc_ids": (
                list(self.reconstruction.route_arc_ids)
                if self.reconstruction is not None
                else []
            ),
            "route_coordinates": [
                [[latitude, longitude] for longitude, latitude in line]
                for line in self.route_lines
            ],
            "route_breaks": (
                max(0, len(self.reconstruction.segment_ranges) - 1)
                if self.reconstruction is not None
                else 0
            ),
        }


def reconstruct_route(
    graph: RoadGraph,
    scored_candidates: Iterable[Mapping[str, object]] | object,
    track: Sequence[Sequence[float]],
    *,
    beta_m: float = 50.0,
    max_speed_mps: float = 70.0,
    gps_tolerance_m: float = 30.0,
    gap_uncertainty_m_per_s: float = 1.5,
) -> ReconstructedRoute:
    """Decode candidate scores and connect selected arcs through the road graph.

    Candidate rows use Member 1's feature schema plus Member 2's ``score``
    probability. The ``track`` is the original ordered longitude/latitude/time
    array, used to calculate displacement across any observations without
    candidates.
    """
    if hasattr(scored_candidates, "to_dict"):
        rows = scored_candidates.to_dict(orient="records")
    else:
        rows = list(scored_candidates)
    if not rows:
        raise ValueError("No scored candidate rows supplied")

    emissions = candidate_scores_to_emissions(rows)
    transition_scorer = CandidateTransitionScorer(
        graph,
        rows,
        track=track,
        beta_m=beta_m,
        max_speed_mps=max_speed_mps,
        gps_tolerance_m=gps_tolerance_m,
        gap_uncertainty_m_per_s=gap_uncertainty_m_per_s,
    )
    decoded = decode_with_restarts(
        emissions.log_emissions, transition_scorer.log_probability
    )
    predicted_arc_ids = tuple(int(arc_id) for arc_id in decoded.states)
    route_segments = tuple(
        tuple(
            expand_candidate_sequence(
                transition_scorer,
                emissions.observation_indices[start:end],
                predicted_arc_ids[start:end],
            )
        )
        for start, end in decoded.segment_ranges
    )
    route_arc_ids = tuple(arc_id for segment in route_segments for arc_id in segment)
    return ReconstructedRoute(
        observation_indices=emissions.observation_indices,
        predicted_arc_ids=predicted_arc_ids,
        route_arc_ids=route_arc_ids,
        log_probability=decoded.log_probability,
        segment_ranges=decoded.segment_ranges,
        route_segments=route_segments,
    )


def reconstruct_prepared_trip(
    prepared_trip: object,
    score_prepared_trip: Callable[..., object],
    *,
    artifact: object | None = None,
    beta_m: float = 50.0,
    max_speed_mps: float = 70.0,
    gps_tolerance_m: float = 30.0,
    gap_uncertainty_m_per_s: float = 1.5,
) -> PreparedTripPrediction:
    """Run Member 2 scoring and route reconstruction on Member 1's PreparedTrip.

    The scorer is injected so this module does not depend on Member 2's model
    package or artifact being present. Its contract matches Member 2's final
    ``score_review1(prepared_trip, artifact)`` API and returns a DataFrame with
    candidate IDs, transition features, and positive-class probabilities.
    """
    try:
        trip = prepared_trip.trip
        trip_id = trip.trip_id
        track = trip.track
    except AttributeError as exc:
        raise TypeError(
            "prepared_trip must provide trip.trip_id and trip.track"
        ) from exc

    graph = RoadGraph.from_trip(trip)
    scored_candidates = score_prepared_trip(prepared_trip, artifact=artifact)
    scored_rows = scored_candidates.to_dict(orient="records")
    if not scored_rows:
        return PreparedTripPrediction(
            trip_id=trip_id,
            reconstruction=None,
            route_lines=(),
        )

    scored_rows = _normalize_zero_score_groups(scored_rows)

    reconstruction = reconstruct_route(
        graph,
        scored_rows,
        track,
        beta_m=beta_m,
        max_speed_mps=max_speed_mps,
        gps_tolerance_m=gps_tolerance_m,
        gap_uncertainty_m_per_s=gap_uncertainty_m_per_s,
    )
    route_lines: list[tuple[tuple[float, float], ...]] = []
    for route_segment in reconstruction.route_segments:
        route_lines.extend(
            tuple((longitude, latitude) for longitude, latitude in line)
            for line in route_geometry(graph, route_segment)
        )
    return PreparedTripPrediction(
        trip_id=trip_id,
        reconstruction=reconstruction,
        route_lines=tuple(route_lines),
    )


def _normalize_zero_score_groups(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    max_score_by_observation: dict[int, float] = {}
    for row in rows:
        gps_index = row.get("gps_index")
        score = row.get("score")
        if (
            isinstance(gps_index, bool)
            or not isinstance(gps_index, Integral)
            or gps_index < 0
            or isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
            or not 0 <= score <= 1
        ):
            return rows
        max_score_by_observation[int(gps_index)] = max(
            max_score_by_observation.get(int(gps_index), 0.0),
            float(score),
        )

    zero_score_observations = {
        gps_index
        for gps_index, max_score in max_score_by_observation.items()
        if max_score == 0
    }
    if not zero_score_observations:
        return rows
    return [
        {**row, "score": sys.float_info.min}
        if row["gps_index"] in zero_score_observations
        else row
        for row in rows
    ]
