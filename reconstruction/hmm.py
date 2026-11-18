"""Adapt scored candidate rows into HMM emission log probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CandidateEmissions:
    """Log emissions for each observed GPS index, in ascending index order."""

    observation_indices: tuple[int, ...]
    log_emissions: list[dict[int, float]]


def candidate_scores_to_emissions(
    scored_candidates: Iterable[Mapping[str, object]],
    observation_indices: Iterable[int] | None = None,
) -> CandidateEmissions:
    """Convert Member 2's scored candidate rows into per-observation emissions.

    Each row must contain ``gps_index``, ``arc_id``, and ``score``, where score
    is ``P(label=1)``. Scores are normalized across the candidates for each GPS
    observation to form a distribution over its available hidden states, then
    converted to natural-log probabilities for the Viterbi decoder.

    Pass all expected GPS indices to detect observations with no candidates.
    Without them, the returned indices identify exactly which observations are
    represented by the rows; observations are never silently inserted.
    """
    grouped: dict[int, dict[int, float]] = {}
    for row_number, row in enumerate(scored_candidates, start=1):
        missing = {"gps_index", "arc_id", "score"} - row.keys()
        if missing:
            raise ValueError(
                f"Candidate row {row_number} is missing required fields: "
                f"{sorted(missing)}"
            )

        gps_index = _integer_value(row["gps_index"], "gps_index", row_number)
        arc_id = _integer_value(row["arc_id"], "arc_id", row_number)
        score = _probability_value(row["score"], row_number)

        candidates = grouped.setdefault(gps_index, {})
        if arc_id in candidates:
            raise ValueError(
                f"Duplicate candidate arc_id {arc_id} for gps_index {gps_index}"
            )
        candidates[arc_id] = score

    if observation_indices is None:
        expected_indices = tuple(sorted(grouped))
    else:
        indices = tuple(
            _integer_value(value, "observation index", position)
            for position, value in enumerate(observation_indices, start=1)
        )
        if len(set(indices)) != len(indices):
            raise ValueError("observation_indices contains duplicates")
        expected_indices = tuple(sorted(indices))
        missing_indices = sorted(set(expected_indices) - grouped.keys())
        extra_indices = sorted(grouped.keys() - set(expected_indices))
        if missing_indices or extra_indices:
            raise ValueError(
                "Candidate observations do not match expected GPS indices; "
                f"missing candidates for {missing_indices}, unexpected indices {extra_indices}"
            )

    emissions: list[dict[int, float]] = []
    for gps_index in expected_indices:
        candidates = grouped[gps_index]
        total_score = math.fsum(candidates.values())
        if total_score <= 0:
            raise ValueError(
                f"All candidate scores are zero for gps_index {gps_index}"
            )
        emissions.append(
            {
                arc_id: math.log(score / total_score) if score > 0 else -math.inf
                for arc_id, score in candidates.items()
            }
        )

    return CandidateEmissions(expected_indices, emissions)


def _integer_value(value: object, field: str, row_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} in row {row_number} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} in row {row_number} must be non-negative")
    return result


def _probability_value(value: object, row_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"score in row {row_number} must be a numeric probability")
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            f"score in row {row_number} must be finite and between 0 and 1"
        )
    return probability
