# PathSnap

## Route reconstruction utilities

The reconstruction package contains directed road-graph helpers, reference-route
analysis, a generic log-space Viterbi decoder, and route-map generation.
`reconstruction.route_reconstruction.reconstruct_prepared_trip()` connects
Member 1's `PreparedTrip` with Member 2's frozen 40-feature
`score_review1()` API for the extended Random Forest, then applies directed,
arc-position-aware transitions and inserts intermediate road arcs.
When no feasible predecessor is available, it starts a new route segment and
reports the break in `segment_ranges` and `route_segments` rather than inventing
a connection.
Transition settings currently provide conservative starting values; evaluate
and tune them using training trips and student-test trips only.

With the model artifact available, the integration call is:

```python
from models.final_review_integration import load_final_model
from models.score_candidates import score_review1
from preprocessing import prepare_trip
from reconstruction.route_reconstruction import reconstruct_prepared_trip

prepared = prepare_trip(trip_id, partition="student_test")
artifact = load_final_model("results/member2/final_review/final_model.pkl")
prediction = reconstruct_prepared_trip(
    prepared,
    score_review1,
    artifact=artifact,
)
arc_ids = (
    prediction.reconstruction.route_arc_ids
    if prediction.reconstruction is not None
    else ()
)
route_lines = prediction.route_lines  # GeoJSON coordinate order: [longitude, latitude]
api_payload = prediction.api_response()  # Folium order: nested [latitude, longitude] lines
```

`score_review1()` receives the full prepared trip because the frozen model
builds temporal and directed-lattice features at inference time. Its scored
rows retain `gps_index`, `arc_id`, `segment_fraction`, `segment_length_m`,
`gps_step_m`, and `time_gap_s`, with `score` containing Member 2's
positive-class probability. Trips with no candidates return empty route fields.
The API payload's
`route_coordinates` is a list of polylines, so a UI can render disconnected
segments without drawing a false connector.
The map generator accepts optional predicted arc IDs for a separate overlay:

```python
create_trip_map(nodes_path, arcs_path, track_path, reference_route_path,
                output_path, predicted_arc_ids=result.route_arc_ids)
```
If the decoder had to restart after impossible transitions, pass
`result.route_segments` instead; these are drawn separately so the map does not
imply continuity across a decoder restart.

Place the organizer-provided dataset under `data/` (with `train/` and
`student_test/` partitions), or pass its location using `--data-dir`.

Audit every trip and write a CSV report:

```powershell
python -m reconstruction.audit_dataset --data-dir data --output results/route_audit.csv
```

Create a Folium map for one trip:

```powershell
python -m pip install -r requirements.txt
python -m reconstruction.visualize_route --trip-dir data/train/00000001 --output results/maps/train_00000001_reference.html
```

The audit reports invalid route arc IDs, directed connectivity breaks, route
length, trip sizes, and weakly connected road-network components. Map generation
shows the GPS track and reference route; it does not use or display predictions.
The supplied dataset is not included in this repository; follow the organizers'
rules before copying it into the repository or publishing it.

Run the unit tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```