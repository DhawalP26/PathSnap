import math
from pathlib import Path
import tempfile
import unittest

from reconstruction.road_graph import RoadGraph
from reconstruction.viterbi import decode


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


if __name__ == "__main__":
    unittest.main()
