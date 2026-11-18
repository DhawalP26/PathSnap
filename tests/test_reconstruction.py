import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from reconstruction.road_graph import RoadGraph
from reconstruction.hmm import candidate_scores_to_emissions
from reconstruction.route_analysis import (
    audit_route,
    compare_routes,
    read_track,
    route_geometry,
    write_route_geojson,
)
from reconstruction.visualize_route import create_trip_map
from reconstruction.transition import (
    ArcPosition,
    CandidateTransitionScorer,
    expand_candidate_sequence,
    route_between_positions,
)
from reconstruction.route_reconstruction import (
    reconstruct_prepared_trip,
    reconstruct_route,
)
from reconstruction.viterbi import decode, decode_with_restarts


class RoadGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.nodes_path = root / "trip.nodes"
        self.arcs_path = root / "trip.arcs"
        self.nodes_path.write_text(
            "0 0\n0.001 0\n0.002 0\n0.001 0.001\n", encoding="utf-8"
        )
        self.arcs_path.write_text(
            "0 1\n1 2\n0 3\n3 2\n1 0\n", encoding="utf-8"
        )
        self.graph = RoadGraph.from_files(self.nodes_path, self.arcs_path)

    def test_arc_ids_follow_file_rows_and_edges_are_directed(self) -> None:
        self.assertEqual(self.graph.arc(2).arc_id, 2)
        self.assertEqual(self.graph.arc(2).from_node, 0)
        self.assertEqual(self.graph.arc(2).to_node, 3)
        self.assertEqual(self.graph.shortest_path(0, 1)[0], [0])
        with self.assertRaisesRegex(ValueError, "No directed path"):
            self.graph.shortest_path(2, 0)

    def test_shortest_path_returns_directed_arc_sequence_and_distance(self) -> None:
        arc_ids, distance = self.graph.shortest_path(0, 2)
        self.assertEqual(arc_ids, [0, 1])
        self.assertGreater(distance, 0)
        self.assertAlmostEqual(distance, self.graph.arc(0).length_m + self.graph.arc(1).length_m)

    def test_weak_components_ignore_direction(self) -> None:
        self.assertEqual(self.graph.weakly_connected_components(), [{0, 1, 2, 3}])

    def test_invalid_arc_endpoint_is_rejected(self) -> None:
        self.arcs_path.write_text("0 99\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside the node table"):
            RoadGraph.from_files(self.nodes_path, self.arcs_path)


class ViterbiTests(unittest.TestCase):
    def test_decoder_uses_transitions_to_choose_a_sequence(self) -> None:
        emissions = [
            {"a": math.log(0.6), "b": math.log(0.4)},
            {"a": math.log(0.4), "b": math.log(0.6)},
        ]

        def transition(previous: str, current: str, timestep: int) -> float:
            del timestep
            return math.log(0.9 if previous == current else 0.1)

        result = decode(emissions, transition)
        self.assertEqual(result.states, ["a", "a"])
        self.assertAlmostEqual(result.log_probability, math.log(0.6 * 0.9 * 0.4))

    def test_impossible_transition_fails_explicitly(self) -> None:
        emissions = [{"a": 0.0}, {"b": 0.0}]
        with self.assertRaisesRegex(ValueError, "No viable path"):
            decode(emissions, lambda _previous, _current, _timestep: -math.inf)

    def test_empty_observation_sequence_returns_empty_path(self) -> None:
        self.assertEqual(decode([], lambda _a, _b, _t: 0).states, [])

    def test_segmented_decoder_restarts_and_reports_break_ranges(self) -> None:
        emissions = [{"a": 0.0}, {"b": 0.0}, {"c": 0.0}]
        result = decode_with_restarts(
            emissions,
            lambda previous, current, _timestep: (
                0.0 if (previous, current) == ("a", "b") else -math.inf
            ),
        )
        self.assertEqual(result.states, ["a", "b", "c"])
        self.assertEqual(result.segment_ranges, ((0, 2), (2, 3)))
        self.assertEqual(result.log_probability, 0.0)

    def test_segmented_decoder_keeps_single_observation_segments(self) -> None:
        result = decode_with_restarts(
            [{"a": 0.0}, {"b": 0.0}],
            lambda _previous, _current, _timestep: -math.inf,
        )
        self.assertEqual(result.states, ["a", "b"])
        self.assertEqual(result.segment_ranges, ((0, 1), (1, 2)))

    def test_scored_candidate_rows_feed_viterbi_in_gps_order(self) -> None:
        scored_rows = [
            {"gps_index": 8, "arc_id": 12, "score": 0.8},
            {"gps_index": 3, "arc_id": 5, "score": 0.2},
            {"gps_index": 3, "arc_id": 6, "score": 0.3},
        ]
        emissions = candidate_scores_to_emissions(scored_rows)
        result = decode(
            emissions.log_emissions,
            lambda _previous, _current, _timestep: 0.0,
        )
        self.assertEqual(emissions.observation_indices, (3, 8))
        self.assertEqual(result.states, [6, 12])
        self.assertEqual(
            list(zip(emissions.observation_indices, result.states)),
            [(3, 6), (8, 12)],
        )
        self.assertAlmostEqual(emissions.log_emissions[0][5], math.log(0.4))
        self.assertAlmostEqual(emissions.log_emissions[0][6], math.log(0.6))

    def test_missing_observation_candidates_are_not_silently_skipped(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing candidates for \\[4\\]"):
            candidate_scores_to_emissions(
                [{"gps_index": 3, "arc_id": 5, "score": 1.0}],
                observation_indices=[3, 4],
            )

    def test_invalid_and_duplicate_candidate_scores_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            candidate_scores_to_emissions(
                [{"gps_index": 0, "arc_id": 1, "score": 1.2}]
            )
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            candidate_scores_to_emissions(
                [
                    {"gps_index": 0, "arc_id": 1, "score": 0.4},
                    {"gps_index": 0, "arc_id": 1, "score": 0.6},
                ]
            )
        with self.assertRaisesRegex(ValueError, "All candidate scores are zero"):
            candidate_scores_to_emissions(
                [{"gps_index": 0, "arc_id": 1, "score": 0.0}]
            )


class TransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        nodes_path = root / "trip.nodes"
        arcs_path = root / "trip.arcs"
        nodes_path.write_text(
            "0 0\n0.001 0\n0.002 0\n0.001 0.001\n", encoding="utf-8"
        )
        arcs_path.write_text(
            "0 1\n1 2\n0 3\n3 2\n1 0\n", encoding="utf-8"
        )
        self.graph = RoadGraph.from_files(nodes_path, arcs_path)

    def candidate(self, gps_index: int, arc_id: int, fraction: float) -> dict[str, object]:
        return {
            "gps_index": gps_index,
            "arc_id": arc_id,
            "segment_fraction": fraction,
            "segment_length_m": self.graph.arc(arc_id).length_m,
            "gps_step_m": 111.0,
            "time_gap_s": 2.0,
        }

    def test_directed_candidate_positions_create_connecting_route(self) -> None:
        rows = [self.candidate(0, 0, 0.5), self.candidate(1, 1, 0.5)]
        track = [(0.0005, 0.0, 1.0), (0.0015, 0.0, 3.0)]
        scorer = CandidateTransitionScorer(self.graph, rows, track=track)
        path = scorer.transition_path(0, 1, 1)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.arc_ids, (0, 1))
        self.assertAlmostEqual(
            path.distance_m,
            self.graph.arc(0).length_m / 2 + self.graph.arc(1).length_m / 2,
        )
        self.assertTrue(math.isfinite(scorer.log_probability(0, 1, 1)))
        self.assertEqual(expand_candidate_sequence(scorer, [0, 1], [0, 1]), [0, 1])

    def test_backward_motion_on_same_arc_uses_a_directed_loop(self) -> None:
        previous = ArcPosition(0, 0, 0.8, self.graph.arc(0).length_m, 0.0, 1.0)
        current = ArcPosition(1, 0, 0.2, self.graph.arc(0).length_m, 0.0, 1.0)
        path = route_between_positions(self.graph, previous, current)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.arc_ids, (0, 4, 0))
        self.assertAlmostEqual(path.distance_m, self.graph.arc(0).length_m * 1.4)

    def test_speed_infeasible_transition_is_impossible(self) -> None:
        rows = [self.candidate(0, 0, 0.5), self.candidate(1, 1, 0.5)]
        track = [(0.0005, 0.0, 1.0), (0.0015, 0.0, 2.0)]
        scorer = CandidateTransitionScorer(
            self.graph,
            rows,
            track=track,
            max_speed_mps=1.0,
            gps_tolerance_m=0.0,
        )
        self.assertIsNone(scorer.transition_path(0, 1, 1))
        self.assertEqual(scorer.log_probability(0, 1, 1), -math.inf)

    def test_track_supports_skipped_candidate_observations(self) -> None:
        rows = [self.candidate(0, 0, 0.1), self.candidate(2, 1, 0.9)]
        track = [
            (0.0001, 0.0, 1.0),
            (0.0006, 0.0, 2.0),
            (0.0019, 0.0, 4.0),
        ]
        scorer = CandidateTransitionScorer(self.graph, rows, track=track)
        self.assertTrue(math.isfinite(scorer.log_probability(0, 1, 1)))

    def test_skipped_candidates_require_track_for_transition_scoring(self) -> None:
        rows = [self.candidate(0, 0, 0.1), self.candidate(2, 1, 0.9)]
        scorer = CandidateTransitionScorer(self.graph, rows)
        with self.assertRaisesRegex(ValueError, "Track data is required"):
            scorer.log_probability(0, 1, 1)

    def test_scored_candidates_decode_and_expand_to_directed_route(self) -> None:
        rows = [
            {**self.candidate(0, 0, 0.5), "score": 0.9},
            {**self.candidate(0, 2, 0.5), "score": 0.1},
            {**self.candidate(1, 1, 0.5), "score": 0.9},
            {**self.candidate(1, 3, 0.5), "score": 0.1},
        ]
        track = [(0.0005, 0.0, 1.0), (0.0015, 0.0, 3.0)]
        result = reconstruct_route(self.graph, rows, track)
        self.assertEqual(result.observation_indices, (0, 1))
        self.assertEqual(result.predicted_arc_ids, (0, 1))
        self.assertEqual(result.route_arc_ids, (0, 1))
        self.assertEqual(result.segment_ranges, ((0, 2),))
        self.assertEqual(result.route_segments, ((0, 1),))
        self.assertTrue(math.isfinite(result.log_probability))

    def test_reconstruction_reports_disconnected_sequences_as_segments(self) -> None:
        rows = [
            {**self.candidate(0, 0, 0.5), "score": 1.0},
            {**self.candidate(1, 1, 0.5), "score": 1.0},
            {**self.candidate(2, 4, 0.5), "score": 1.0},
        ]
        track = [
            (0.0005, 0.0, 1.0),
            (0.0015, 0.0, 3.0),
            (0.001, 0.0, 5.0),
        ]
        result = reconstruct_route(self.graph, rows, track)
        self.assertEqual(result.predicted_arc_ids, (0, 1, 4))
        self.assertEqual(result.segment_ranges, ((0, 2), (2, 3)))
        self.assertEqual(result.route_segments, ((0, 1), (4,)))
        self.assertEqual(result.route_arc_ids, (0, 1, 4))

    def test_prepared_trip_integration_calls_scorer_and_returns_geo_lines(self) -> None:
        prepared = SimpleNamespace(
            trip=SimpleNamespace(
                trip_id="00000001",
                nodes=self.graph.nodes,
                arcs=[(arc.from_node, arc.to_node) for arc in self.graph.arcs],
                track=[(0.0005, 0.0, 1.0), (0.0015, 0.0, 3.0)],
            ),
            network=object(),
            features=pd.DataFrame(
                [
                    self.candidate(0, 0, 0.5),
                    self.candidate(0, 2, 0.5),
                    self.candidate(1, 1, 0.5),
                    self.candidate(1, 3, 0.5),
                ]
            ),
        )
        artifact = object()
        scorer_calls = []

        def score_prepared_trip(prepared_trip, artifact):
            scorer_calls.append((prepared_trip, artifact))
            return prepared_trip.features.assign(
                score=[
                    0.9 if arc_id in (0, 1) else 0.1
                    for arc_id in prepared_trip.features["arc_id"]
                ]
            )

        result = reconstruct_prepared_trip(
            prepared, score_prepared_trip, artifact=artifact
        )
        self.assertEqual(scorer_calls, [(prepared, artifact)])
        self.assertEqual(result.trip_id, "00000001")
        self.assertIsNotNone(result.reconstruction)
        assert result.reconstruction is not None
        self.assertEqual(result.reconstruction.predicted_arc_ids, (0, 1))
        self.assertEqual(
            result.route_lines,
            (((0.0, 0.0), (0.001, 0.0), (0.002, 0.0)),),
        )
        response = result.api_response()
        self.assertEqual(response["trip_id"], "00000001")
        self.assertEqual(response["predicted_arc_ids"], [0, 1])
        self.assertEqual(
            response["route_coordinates"],
            [[[0.0, 0.0], [0.0, 0.001], [0.0, 0.002]]],
        )
        self.assertEqual(response["route_breaks"], 0)

    def test_prepared_trip_integration_handles_zero_scores_and_no_candidates(self) -> None:
        prepared = SimpleNamespace(
            trip=SimpleNamespace(
                trip_id="00000002",
                nodes=self.graph.nodes,
                arcs=[(arc.from_node, arc.to_node) for arc in self.graph.arcs],
                track=[(0.0005, 0.0, 1.0)],
            ),
            network=object(),
            features=pd.DataFrame([self.candidate(0, 0, 0.5)]),
        )

        zero_scored = pd.DataFrame(
            [{**self.candidate(0, 0, 0.5), "score": 0.0}]
        )
        result = reconstruct_prepared_trip(
            prepared, lambda _trip, *, artifact: zero_scored
        )
        self.assertIsNotNone(result.reconstruction)
        assert result.reconstruction is not None
        self.assertEqual(result.reconstruction.predicted_arc_ids, (0,))

        empty_result = reconstruct_prepared_trip(
            prepared,
            lambda _trip, *, artifact: pd.DataFrame(
                columns=["gps_index", "arc_id", "score"]
            ),
        )
        self.assertIsNone(empty_result.reconstruction)
        self.assertEqual(
            empty_result.api_response(),
            {
                "trip_id": "00000002",
                "predicted_arc_ids": [],
                "route_coordinates": [],
                "route_breaks": 0,
            },
        )


class RouteAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        nodes_path = root / "trip.nodes"
        arcs_path = root / "trip.arcs"
        nodes_path.write_text(
            "0 0\n0.001 0\n0.002 0\n0.001 0.001\n", encoding="utf-8"
        )
        arcs_path.write_text(
            "0 1\n1 2\n0 3\n3 2\n1 0\n", encoding="utf-8"
        )
        self.graph = RoadGraph.from_files(nodes_path, arcs_path)

    def test_route_geometry_splits_at_disconnected_arc_pairs(self) -> None:
        lines = route_geometry(self.graph, [0, 2, 3])
        self.assertEqual(lines, [
            [self.graph.nodes[0], self.graph.nodes[1]],
            [self.graph.nodes[0], self.graph.nodes[3], self.graph.nodes[2]],
        ])

    def test_route_audit_reports_breaks_without_bridging_invalid_ids(self) -> None:
        audit = audit_route(self.graph, [0, 2, 999, 3])
        self.assertEqual(audit.invalid_arc_ids, (999,))
        self.assertEqual(audit.transition_count, 1)
        self.assertEqual(audit.discontinuity_indices, (0,))
        self.assertEqual(audit.connected_transitions, 0)
        self.assertAlmostEqual(
            audit.length_m,
            self.graph.arc(0).length_m
            + self.graph.arc(2).length_m
            + self.graph.arc(3).length_m,
        )

    def test_route_comparison_uses_arc_multiplicity_and_sequence_order(self) -> None:
        metrics = compare_routes([1, 2, 2], [2, 2, 3])
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)
        self.assertAlmostEqual(metrics.f1, 2 / 3)
        self.assertAlmostEqual(metrics.sequence_similarity, 1 / 3)
        self.assertFalse(metrics.exact_match)
        self.assertTrue(compare_routes([], []).exact_match)

    def test_read_track_validates_rows_and_geojson_is_written(self) -> None:
        track_path = Path(self.temp_dir.name) / "trip.track"
        track_path.write_text("0 0 1\n0.1 0.1 2\n", encoding="utf-8")
        self.assertEqual(read_track(track_path), [(0.0, 0.0, 1.0), (0.1, 0.1, 2.0)])
        output_path = Path(self.temp_dir.name) / "route.geojson"
        write_route_geojson(self.graph, [0, 2, 3], output_path)
        content = output_path.read_text(encoding="utf-8")
        self.assertIn('"MultiLineString"', content)
        self.assertIn('"coordinates"', content)

    def test_map_can_overlay_predicted_route(self) -> None:
        track_path = Path(self.temp_dir.name) / "trip.track"
        route_path = Path(self.temp_dir.name) / "trip.route"
        output_path = Path(self.temp_dir.name) / "trip.html"
        track_path.write_text("0 0 1\n0.001 0 2\n", encoding="utf-8")
        route_path.write_text("0\n", encoding="utf-8")
        create_trip_map(
            Path(self.temp_dir.name) / "trip.nodes",
            Path(self.temp_dir.name) / "trip.arcs",
            track_path,
            route_path,
            output_path,
            predicted_arc_ids=[0, 1],
        )
        html = output_path.read_text(encoding="utf-8")
        self.assertIn("Reference route", html)
        self.assertIn("Predicted route", html)

    def test_map_keeps_explicit_predicted_route_segments(self) -> None:
        track_path = Path(self.temp_dir.name) / "trip.track"
        route_path = Path(self.temp_dir.name) / "trip.route"
        output_path = Path(self.temp_dir.name) / "segmented_trip.html"
        track_path.write_text("0 0 1\n0.001 0 2\n", encoding="utf-8")
        route_path.write_text("0\n", encoding="utf-8")
        create_trip_map(
            Path(self.temp_dir.name) / "trip.nodes",
            Path(self.temp_dir.name) / "trip.arcs",
            track_path,
            route_path,
            output_path,
            predicted_route_segments=[[0, 1], [4]],
            predicted_route_name="Diagnostic routes",
        )
        html = output_path.read_text(encoding="utf-8")
        self.assertIn("Diagnostic routes", html)


if __name__ == "__main__":
    unittest.main()
