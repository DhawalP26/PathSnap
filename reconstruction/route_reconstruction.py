"""Reconstruct directed routes from scored candidate road segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from reconstruction.hmm import candidate_scores_to_emissions
from reconstruction.road_graph import RoadGraph
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
