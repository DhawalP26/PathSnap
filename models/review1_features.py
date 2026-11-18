"""Prediction-time candidate context for offline trip scoring."""
import numpy as np
import pandas as pd

from models.schema import FEATURES

GROUPS = {
    "relative": ["distance_excess_m", "distance_share", "nearest_margin_m",
                 "heading_excess_deg", "candidate_count"],
    "motion": ["center_heading_diff", "forward_heading_diff", "heading_turn",
               "log_previous_step", "log_next_step", "log_next_gap"],
    "network": ["endpoint_distance_m", "source_indegree", "target_outdegree",
                "previous_segment_distance_m", "next_segment_distance_m",
                "previous_candidate_connection_m", "next_candidate_connection_m"],
}
VARIANTS = {
    "baseline": [],
    "relative_motion": GROUPS["relative"] + GROUPS["motion"],
    "full_context": sum(GROUPS.values(), []),
}


def enrich(prepared):
    """Use GPS and directed arcs, including all inference candidates in each group."""
    frame = prepared.features.copy()
    net = prepared.network
    track = prepared.trip.track
    xy = net.project(track)
    groups = frame.groupby("gps_index", sort=False)
    d, h = frame.distance_m, frame.heading_diff
    nearest = groups.distance_m.transform("min")
    first_two = frame.sort_values("distance_m", kind="stable").groupby(
        "gps_index", sort=False).head(2).groupby("gps_index").distance_m
    frame["distance_excess_m"] = d - nearest
    frame["distance_share"] = (d + 1) / (nearest + 1)
    frame["nearest_margin_m"] = frame.gps_index.map(first_two.max() - first_two.min())
    frame["heading_excess_deg"] = h - groups.heading_diff.transform("min")
    frame["candidate_count"] = groups.distance_m.transform("size")

    obs = frame.gps_index.to_numpy(dtype=int)
    arc = frame.arc_id.to_numpy(dtype=int)
    previous = np.maximum(obs - 1, 0)
    following = np.minimum(obs + 1, len(track) - 1)
    incoming = xy[obs] - xy[previous]
    outgoing = xy[following] - xy[obs]

    def bearing(v):
        angle = np.degrees(np.arctan2(v[:, 0], v[:, 1]))
        angle[np.linalg.norm(v, axis=1) == 0] = np.nan
        return angle

    def diff(a, b):
        return np.abs((a - b + 180) % 360 - 180)

    frame["center_heading_diff"] = diff(bearing(incoming + outgoing), net.headings[arc])
    frame["forward_heading_diff"] = diff(bearing(outgoing), net.headings[arc])
    frame["heading_turn"] = diff(bearing(incoming), bearing(outgoing))
    frame["log_previous_step"] = np.log1p(np.linalg.norm(incoming, axis=1))
    frame["log_next_step"] = np.log1p(np.linalg.norm(outgoing, axis=1))
    frame["log_next_gap"] = np.log1p(track[following, 2] - track[obs, 2])

    starts, ends = net.arcs.T
    frame["endpoint_distance_m"] = frame.segment_length_m * np.minimum(
        frame.segment_fraction, 1 - frame.segment_fraction)
    frame["source_indegree"] = np.bincount(ends, minlength=len(net.nodes))[starts[arc]]
    frame["target_outdegree"] = np.bincount(starts, minlength=len(net.nodes))[ends[arc]]
    segment_vector = net.vectors[arc]
    length_squared = frame.segment_length_m.to_numpy() ** 2

    for name, adjacent, node_side, arc_side in [
        ("previous", previous, "end", starts), ("next", following, "start", ends)
    ]:
        gap = xy[adjacent] - net.starts[arc]
        fraction = np.divide(np.einsum("ij,ij->i", gap, segment_vector), length_squared,
                             out=np.zeros(len(arc)), where=length_squared > 0)
        residual = gap - np.clip(fraction, 0, 1)[:, None] * segment_vector
        distance = np.linalg.norm(residual, axis=1)
        distance[obs == adjacent] = np.nan
        frame[f"{name}_segment_distance_m"] = distance
        neighbor_nodes = ends if node_side == "end" else starts
        # Encode observation/node pairs as integers for vectorized lookups.
        n = len(net.nodes)
        candidates = pd.Series(d.to_numpy(), index=obs * n + neighbor_nodes[arc])
        by_node = candidates.groupby(level=0).min()
        connected = by_node.reindex(adjacent * n + arc_side[arc]).to_numpy()
        same_arc = pd.Series(d.to_numpy(), index=obs * len(net.arcs) + arc)
        same = same_arc.reindex(adjacent * len(net.arcs) + arc).to_numpy()
        best = np.fmin(connected, same)
        best[obs == adjacent] = np.nan
        frame[f"{name}_candidate_connection_m"] = best
    return frame
