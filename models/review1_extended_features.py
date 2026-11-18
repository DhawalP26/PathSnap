"""Label-free temporal geometry and directed candidate-lattice features."""
import numpy as np

from models.review1_features import VARIANTS, enrich

TEMPORAL = [
    "signed_orientation_change", "arc_progress_m", "arc_remaining_m",
    "window_heading_error_deg", "window_straightness", "normal_offset_m",
    "signed_motion_along_arc_m", "group_center_heading_excess_deg",
]
LATTICE = [
    "incoming_path_cost", "outgoing_path_cost", "incoming_arc_support",
    "outgoing_arc_support", "bidirectional_path_excess", "lattice_candidate_rank",
]


def extend(prepared):
    frame = enrich(prepared)
    network = prepared.network
    xy = network.project(prepared.trip.track)
    obs = frame.gps_index.to_numpy(dtype=int)
    arcs = frame.arc_id.to_numpy(dtype=int)
    lengths = network.lengths[arcs]
    progress = frame.segment_fraction.to_numpy() * lengths
    frame["arc_progress_m"] = progress
    frame["arc_remaining_m"] = lengths - progress
    angle = np.radians(network.headings[arcs])
    tangent = np.column_stack((np.sin(angle), np.cos(angle)))
    offset = xy[obs] - network.starts[arcs]
    frame["normal_offset_m"] = tangent[:, 0] * offset[:, 1] - tangent[:, 1] * offset[:, 0]
    index = np.arange(len(xy))
    prev = np.maximum(index - 1, 0)
    nxt = np.minimum(index + 1, len(xy) - 1)
    left = xy[index] - xy[prev]
    right = xy[nxt] - xy[index]
    left_angle = np.degrees(np.arctan2(left[:, 0], left[:, 1]))
    right_angle = np.degrees(np.arctan2(right[:, 0], right[:, 1]))
    turn = (right_angle - left_angle + 180) % 360 - 180
    turn[(prev == index) | (nxt == index)] = np.nan
    frame["signed_orientation_change"] = turn[obs]
    frame["signed_motion_along_arc_m"] = np.einsum("ij,ij->i", tangent, (left + right)[obs])
    window_left = np.maximum(index - 2, 0)
    window_right = np.minimum(index + 2, len(xy) - 1)
    displacement = xy[window_right] - xy[window_left]
    norm = np.linalg.norm(displacement, axis=1)
    traveled = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    length = traveled[window_right] - traveled[window_left]
    straightness = np.divide(norm, length, where=length > 0, out=np.zeros_like(norm))
    heading = np.degrees(np.arctan2(displacement[:, 0], displacement[:, 1]))
    heading[norm == 0] = np.nan
    frame["window_heading_error_deg"] = np.abs((heading[obs] - network.headings[arcs] + 180) % 360 - 180)
    frame["window_straightness"] = straightness[obs]
    center_min = frame.groupby("gps_index").center_heading_diff.transform("min")
    frame["group_center_heading_excess_deg"] = frame.center_heading_diff - center_min

    if frame.empty:
        for name in LATTICE:
            frame[name] = np.empty(0)
        return frame

    counts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1, len(obs)]
    slices = [slice(a, b) for a, b in zip(counts[:-1], counts[1:])]
    source, target = network.arcs[arcs].T
    unary = np.log1p(frame.distance_m.to_numpy() / 6)
    unary += np.nan_to_num(frame.center_heading_diff.to_numpy(), nan=0) / 100
    forward = np.zeros(len(frame))
    backward = np.zeros(len(frame))
    support_in = np.zeros(len(frame))
    support_out = np.zeros(len(frame))
    transitions = []

    for a, b in zip(slices[:-1], slices[1:]):
        delta = obs[b.start] - obs[a.start]
        straight = np.linalg.norm(xy[obs[b.start]] - xy[obs[a.start]])
        same = arcs[a, None] == arcs[None, b]
        direct = target[a, None] == source[None, b]
        road_distance = np.where(same, progress[None, b] - progress[a, None],
                                 lengths[a, None] - progress[a, None] + progress[None, b])
        error = np.log1p(np.abs(road_distance - straight) / 8)
        error += np.maximum(-road_distance, 0) / 15
        connected = same | direct
        transition = np.where(connected, np.minimum(error, 5), 2.5)
        if delta != 1:
            transition.fill(0)
            connected.fill(False)
        transitions.append(transition)
        forward[b] = (forward[a, None] + unary[a, None] + transition).min(axis=0)
        forward[b] -= forward[b].min()
        support_in[b] = connected.sum(axis=0)
        support_out[a] = connected.sum(axis=1)

    for a, b, transition in reversed(list(zip(slices[:-1], slices[1:], transitions))):
        backward[a] = (transition + unary[None, b] + backward[None, b]).min(axis=1)
        backward[a] -= backward[a].min()
    frame["incoming_path_cost"] = forward
    frame["outgoing_path_cost"] = backward
    frame["incoming_arc_support"] = support_in
    frame["outgoing_arc_support"] = support_out
    costs = unary + forward + backward
    groups = frame.assign(path_cost=costs).groupby("gps_index").path_cost
    frame["bidirectional_path_excess"] = costs - groups.transform("min")
    frame["lattice_candidate_rank"] = groups.rank(method="min")
    return frame


EXTENDED_VARIANTS = {
    "review1_extended_temporal": VARIANTS["full_context"] + TEMPORAL,
    "review1_extended_lattice": VARIANTS["full_context"] + TEMPORAL + LATTICE,
}
