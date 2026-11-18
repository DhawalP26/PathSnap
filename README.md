# PathSnap

## Route reconstruction utilities

The reconstruction package contains directed road-graph helpers, reference-route
analysis, a generic log-space Viterbi decoder, and route-map generation.
`reconstruction.hmm.candidate_scores_to_emissions()` adapts Member 2's scored
candidate rows (`gps_index`, `arc_id`, and `score` = `P(label=1)`) into normalized
per-observation log emissions. Pair the returned `observation_indices` with the
decoded states to retain their GPS-row mapping; pass the expected GPS indices
when missing candidate sets should be detected.
`reconstruction.route_reconstruction.reconstruct_route()` combines those
emissions with directed, arc-position-aware transition scoring and inserts
intermediate road arcs into the final route. When no feasible predecessor is
available, it explicitly starts a new route segment and reports `segment_ranges`
and `route_segments` rather than inventing a connecting path.
Transition settings currently provide conservative starting values; evaluate
and tune them using training trips and student-test trips only.
The map generator accepts optional predicted arc IDs for a separate overlay:

```python
create_trip_map(nodes_path, arcs_path, track_path, reference_route_path,
                output_path, predicted_arc_ids=result.route_arc_ids)
```
If the decoder had to restart after impossible transitions, pass
`result.route_segments` instead; these are drawn separately so the map does not
imply continuity across a decoder restart.

```python
scored = score_candidates(candidate_features)
emissions = candidate_scores_to_emissions(
    scored.to_dict("records"),
    observation_indices=range(len(track)),
)
decoded = decode(emissions.log_emissions, transition_log_probability)
predicted_arc_by_gps_index = dict(zip(emissions.observation_indices, decoded.states))
```

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