# PathSnap Member 1 preprocessing

Python 3.10+ is supported. Run these commands from the repository root.

```powershell
python -m pip install -r requirements-member1.txt
$env:PATHSNAP_DATA_DIR = 'C:\Users\dshiv\Downloads\Student folder'
python -m pytest -q
python -m preprocessing.eda
python -m preprocessing.export_training
```

The dataset root contains `train/<trip_id>/` and `student_test/<trip_id>/`. Each file is named after its eight-digit trip ID. Tracks contain longitude, latitude and timestamp in seconds. Nodes contain longitude and latitude. Arcs contain directed source and destination node IDs. Reference routes contain an ordered sequence of directed arc IDs. Arc and node IDs are zero-based indexes into their respective data rows. Files accept whitespace separators and ignore blank lines. Validation checks field count, numeric types, finite values, coordinate ranges and ID bounds. Invalid records fail with file context. GPS observations are sorted by timestamp with input order retained for ties.

## Member 2 · Train the scoring model

```python
import pandas as pd
from preprocessing import FEATURE_COLUMNS, prepare_training_trip

rows = pd.read_csv("results/member1/train_features.csv", dtype={"trip_id": str})
X = rows.loc[:, list(FEATURE_COLUMNS)]
y = rows["label"]
groups = rows["trip_id"]

# To inspect a single trip or rebuild it in Python:
rows_for_trip, diagnostics = prepare_training_trip("00000005")
```

Use trip-level splits such as GroupKFold and fit any imputers or scalers inside each training fold. Use the same radius and candidate count for both training and inference. Preserve `gps_index` and `arc_id` to pass predictions downstream. `training_coverage.csv` reports per-trip observation counts, alignment counts, retrieved positives, candidate recall among aligned observations and exported row counts.

| Feature | Definition |
|---|---|
| `distance_m` | Shortest planar distance to the closed line segment between the arc endpoints |
| `heading_diff` | Absolute angle in degrees between GPS movement and the directed arc, in [0, 180] |
| `speed_mps` | Distance from previous observation divided by elapsed time |
| `time_gap_s` | Seconds since previous observation, with zero for the first row |
| `gps_step_m` | Distance from previous observation, with zero for the first row |
| `segment_length_m` | Projected length of the arc |
| `candidate_rank` | One-based distance rank with ties resolved by arc ID |
| `segment_fraction` | Position of the nearest point along the directed arc, in [0, 1] |

Undefined headings and speeds are NaN. An observation can have zero candidate rows. A trip can return an empty feature table with the same column schema.

Candidate search builds length-bucketed spatial indexes of arc midpoints in a local azimuthal equidistant projection. Each search checks up to radius plus half the maximum length in that bucket and filters on actual point-to-segment distance. Defaults are 75 m and eight candidates.

Training aligns the entire GPS sequence to ordered route occurrences using dynamic programming. The cost combines distance, up to 10 m equivalent heading penalty, and 2 m equivalent per skipped interior occurrence. It can stay on an occurrence or move forward, and the track can start and end anywhere along the route. Repeated arc IDs retain separate states. Exact transition ties prefer staying, prefix ties prefer the earlier occurrence, and final ties prefer the earliest ending occurrence. Junction assignments and near-ties are heuristic labels. An aligned observation beyond the distance limit is represented as `-1` in diagnostics. Observations without an accepted match or without the matching arc in their candidate set are excluded from training. Every retained group contains one positive and any retrieved competing arcs as negatives. Reference alignment is run after feature generation and does not change feature values or ranking.

## Member 3 · Build the directed decoding graph

```python
from preprocessing import load_trip, prepare_trip

trip = load_trip("00000000", partition="student_test")
nodes, arcs = trip.nodes, trip.arcs

prepared = prepare_trip("00000000")
lengths_m = prepared.network.lengths
candidates_by_observation = prepared.features.groupby("gps_index")
```

`nodes[node_id]` gives longitude and latitude. `arcs[arc_id]` gives the source and destination node IDs in traversal order. For parallel arcs, use a directed multigraph or keep the arc ID as part of the edge key. Candidate rows use the sorted GPS index into `trip.track`. Iterate the full track index when decoding so candidate-free observations remain visible.

## Member 4 · Call the API from the FastAPI adapter

The existing adapter receives a server-resolved `trip_dir` and `trip_id`. Call the synchronous function from `predict_trip` and pass its results into the Member 2 scorer and Member 3 decoder.

```python
from preprocessing import FEATURE_COLUMNS, prepare_trip

prepared = prepare_trip(trip_id, trip_dir=trip_dir, radius_m=75, top_k=8)
model_input = prepared.features.loc[:, list(FEATURE_COLUMNS)]
# Score model_input and retain the candidate IDs and GPS indices for decoding.
# Use prepared.trip and prepared.network when constructing the route.
```

`prepare_trip()` reads only `.track`, `.nodes` and `.arcs`. Route reference loading belongs in offline training or after predictions for offline evaluation. Convert geographic coordinates to latitude, longitude when returning map coordinates through the existing adapter.

## EDA and exports

`python -m preprocessing.eda` scans both partitions, producing `dataset_inventory.csv`, `summary.md`, five PNG plots and an interactive Folium map under `results/member1/eda`. The map shows the GPS trace, road candidates and available offline reference. Use `--map-partition student_test --trip 00000000` to inspect a different trip. For a single-partition run, set both `--partition` and `--map-partition` to that partition.

`python -m preprocessing.export_training` writes `train_features.csv` and `training_coverage.csv` under `results/member1`. Both commands accept `--out-dir`. Generated files are reproducible from the dataset and excluded from Git at the default output path.
