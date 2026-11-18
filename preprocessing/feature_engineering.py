"""Observation motion and candidate features shared by training and inference."""
import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "distance_m", "heading_diff", "speed_mps", "time_gap_s", "gps_step_m",
    "segment_length_m", "candidate_rank", "segment_fraction",
)
ID_COLUMNS = ("trip_id", "gps_index", "arc_id")


def motion(track, points):
    steps = np.diff(points, axis=0, prepend=points[:1])
    gap = np.diff(track[:, 2], prepend=track[0, 2])
    distance = np.linalg.norm(steps, axis=1)
    speed = np.divide(distance, gap, out=np.full(len(gap), np.nan), where=gap > 0)
    heading = np.degrees(np.arctan2(steps[:, 0], steps[:, 1])) % 360
    heading[(distance == 0) | (gap <= 0)] = np.nan
    return heading, speed, gap, distance


def angle_difference(left, right):
    return np.abs((left - right + 180) % 360 - 180)


def feature_table(trip, network, points, movement, radius_m, top_k):
    headings, speeds, gaps, steps = movement
    rows = []
    for i, (ids, distances, fractions) in enumerate(network.candidates(points, radius_m, top_k)):
        difference = angle_difference(headings[i], network.headings[ids])
        for rank, (arc, distance, fraction, angle) in enumerate(
                zip(ids, distances, fractions, difference), 1):
            rows.append((trip.trip_id, i, arc, distance, angle, speeds[i], gaps[i],
                         steps[i], network.lengths[arc], rank, fraction))
    table = pd.DataFrame(rows, columns=ID_COLUMNS + FEATURE_COLUMNS)
    if table.empty:
        dtypes = dict.fromkeys(FEATURE_COLUMNS, "float64")
        dtypes.update(trip_id="object", gps_index="int64", arc_id="int64", candidate_rank="int64")
        table = table.astype(dtypes)
    return table
