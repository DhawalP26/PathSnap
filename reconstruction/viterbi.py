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


def _validate_log_score(score: float, description: str) -> None:
    if math.isnan(score) or score == math.inf:
        raise ValueError(f"{description} must be finite or negative infinity")
