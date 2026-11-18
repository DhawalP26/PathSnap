# PathSnap

## Route reconstruction utilities

The reconstruction package contains directed road-graph helpers, reference-route
analysis, a generic log-space Viterbi decoder, and route-map generation.

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