"""Generic log-space Viterbi decoder for per-observation candidate states."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Hashable, Mapping, TypeVar


State = TypeVar("State", bound=Hashable)


@dataclass(frozen=True)
class ViterbiResult:
    states: list[Hashable]
    log_probability: float


@dataclass(frozen=True)
class SegmentedViterbiResult:
    states: list[Hashable]
    log_probability: float
    segment_ranges: tuple[tuple[int, int], ...]


def decode(
    emissions: list[Mapping[State, float]],
    transition_log_probability: Callable[[State, State, int], float],
) -> ViterbiResult:
    """Find the most likely state sequence from log emission scores.

    The transition callback receives (previous_state, current_state, timestep),
    where timestep is the index of the current observation. Impossible choices
    should return negative infinity.
    """
    if not emissions:
        return ViterbiResult([], 0.0)

    scores: dict[State, float] = {}
    backpointers: list[dict[State, State]] = []
    for timestep, timestep_emissions in enumerate(emissions):
        if not timestep_emissions:
            raise ValueError(f"No candidate states at timestep {timestep}")

        for state, score in timestep_emissions.items():
            _validate_log_score(score, f"emission at timestep {timestep}")

        if timestep == 0:
            scores = dict(timestep_emissions)
            if all(score == -math.inf for score in scores.values()):
                raise ValueError("No viable initial state")
            continue

        next_scores: dict[State, float] = {}
        timestep_backpointers: dict[State, State] = {}
        for current_state, emission_score in timestep_emissions.items():
            if emission_score == -math.inf:
                continue
            best_score = -math.inf
            best_previous: State | None = None
            for previous_state, previous_score in scores.items():
                if previous_score == -math.inf:
                    continue
                transition_score = transition_log_probability(
                    previous_state, current_state, timestep
                )
                _validate_log_score(transition_score, f"transition at timestep {timestep}")
                candidate_score = previous_score + transition_score
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = previous_state
            if best_previous is not None and best_score != -math.inf:
                next_scores[current_state] = best_score + emission_score
                timestep_backpointers[current_state] = best_previous

        if not next_scores:
            raise ValueError(f"No viable path reaches timestep {timestep}")
        scores = next_scores
        backpointers.append(timestep_backpointers)

    final_state = max(scores, key=scores.__getitem__)
    best_log_probability = scores[final_state]
    path: list[State] = [final_state]
    for timestep_backpointers in reversed(backpointers):
        path.append(timestep_backpointers[path[-1]])
    path.reverse()
    return ViterbiResult(path, best_log_probability)


def decode_with_restarts(
    emissions: list[Mapping[State, float]],
    transition_log_probability: Callable[[State, State, int], float],
) -> SegmentedViterbiResult:
    """Decode candidate sequences, restarting only when no transition is feasible.

    Segment ranges are half-open indexes into the input emissions. A restart
    represents a genuine break in route continuity, not an inferred transition.
    """
    if not emissions:
        return SegmentedViterbiResult([], 0.0, ())

    for timestep, timestep_emissions in enumerate(emissions):
        if not timestep_emissions:
            raise ValueError(f"No candidate states at timestep {timestep}")
        for score in timestep_emissions.values():
            _validate_log_score(score, f"emission at timestep {timestep}")

    states: list[Hashable] = []
    ranges: list[tuple[int, int]] = []
    total_log_probability = 0.0
    segment_start = 0
    scores = dict(emissions[0])
    if all(score == -math.inf for score in scores.values()):
        raise ValueError("No viable initial state")
    backpointers: list[dict[State, State]] = []

    for timestep in range(1, len(emissions)):
        next_scores: dict[State, float] = {}
        timestep_backpointers: dict[State, State] = {}
        for current_state, emission_score in emissions[timestep].items():
            if emission_score == -math.inf:
                continue
            best_score = -math.inf
            best_previous: State | None = None
            for previous_state, previous_score in scores.items():
                if previous_score == -math.inf:
                    continue
                transition_score = transition_log_probability(
                    previous_state, current_state, timestep
                )
                _validate_log_score(
                    transition_score, f"transition at timestep {timestep}"
                )
                candidate_score = previous_score + transition_score
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = previous_state
            if best_previous is not None and best_score != -math.inf:
                next_scores[current_state] = best_score + emission_score
                timestep_backpointers[current_state] = best_previous

        if next_scores:
            scores = next_scores
            backpointers.append(timestep_backpointers)
            continue

        segment_states, segment_score = _backtrack(scores, backpointers)
        states.extend(segment_states)
        ranges.append((segment_start, timestep))
        total_log_probability += segment_score
        segment_start = timestep
        scores = dict(emissions[timestep])
        if all(score == -math.inf for score in scores.values()):
            raise ValueError(f"No viable state at restart timestep {timestep}")
        backpointers = []

    segment_states, segment_score = _backtrack(scores, backpointers)
    states.extend(segment_states)
    ranges.append((segment_start, len(emissions)))
    total_log_probability += segment_score
    return SegmentedViterbiResult(states, total_log_probability, tuple(ranges))


def _backtrack(
    scores: Mapping[State, float], backpointers: list[dict[State, State]]
) -> tuple[list[State], float]:
    final_state = max(scores, key=scores.__getitem__)
    path: list[State] = [final_state]
    for timestep_backpointers in reversed(backpointers):
        path.append(timestep_backpointers[path[-1]])
    path.reverse()
    return path, scores[final_state]


def _validate_log_score(score: float, description: str) -> None:
    if math.isnan(score) or score == math.inf:
        raise ValueError(f"{description} must be finite or negative infinity")
