# PathSnap Member 2 · Candidate scoring

Member 2 compares five supervised candidate classifiers, tunes the strongest by development cross-validation F1, and exports a fitted pipeline for Viterbi decoding and FastAPI. Python 3.10 or later is supported.

## Run locally

```powershell
python -m pip install -r requirements_member2.txt
python -m pytest -q tests/member2
python -m models.train_models --train results/member1/train_features.csv
python -m models.bayesian_tuning --train results/member1/train_features.csv --trials 20
```

Generate the CSV with Member 1's exporter or obtain the exported CSV and accompanying training coverage report. Use identical `--seed`, `--folds`, `--max-trips`, and `--out` settings for comparison and tuning. With a matching evaluation report and CSV fingerprint, tuning reuses the comparison. Otherwise it runs the comparison. Omit `--max-trips` to use every available trip.

The features in artifact order are `distance_m`, `heading_diff`, `speed_mps`, `time_gap_s`, `gps_step_m`, `segment_length_m`, `candidate_rank`, and `segment_fraction`. The IDs `trip_id`, `gps_index`, and `arc_id` are retained as metadata. Missing values and infinities enter a fold-local median imputer. Logistic Regression and MLP also use a fold-local scaler. Candidate keys must be unique and trip IDs retain leading zeros.

## Evaluation and model selection

A fixed seed holds out 25% of trips. Five classifiers use identical GroupKFold splits inside the development partition. The comparison aggregates out-of-fold probabilities and emits per-fold timing and metrics, confusion matrices, a comparison plot and an evaluation summary under `results/member2`. Per-observation top-1 includes all-negative groups present in the CSV. Member 1's export drops observations without a retrieved positive, so the top-1 metric on that export is conditional on retrieval. The accompanying coverage report measures recall among accepted reference alignments. These are candidate metrics. Route reconstruction accuracy requires separate evaluation after decoding.

Optuna TPE tunes the development winner, using the same folds for each trial. Two startup trials initialize the sampler. A tuned pipeline is selected when its development OOF F1 beats the untuned baseline. Both variants are evaluated on the same untouched held-out trips, with results in `tuned_vs_untuned.csv`. The development tuning score participates in selection. Use the held-out metrics to assess generalization. The exported pipeline is fitted on development trips, leaving the held-out trips excluded. Trial scores, best parameters and runtime are saved alongside the comparison outputs.

The default artifact is `models/best_model.pkl`. Transfer it separately or regenerate it with the commands above. Load only a trusted artifact and use the dependency versions recorded in its metadata. Its scores are uncalibrated positive-class probabilities.

## Members 3 and 4

```python
from preprocessing import prepare_trip
from models.score_candidates import load_model, score_candidates

artifact = load_model("models/best_model.pkl")  # Load once per process at startup.
prepared = prepare_trip(trip_id, trip_dir=trip_dir, radius_m=75, top_k=8)
scored = score_candidates(prepared.features, artifact=artifact)
```

For a one-off call, use `score_candidates(prepared.features, model_path="models/best_model.pkl")`. The result keeps candidate row order, index and the three identifiers, and adds finite scores in [0, 1]. For Member 3, iterate every index in the sorted GPS track and use an explicit missing-state or gap policy for observations without candidate rows. For Member 4, pass the scored rows and the prepared network into the decoder from the existing backend adapter. An empty candidate table returns an empty scored table.

## Measured run

The validation run used 12 of 80 exported training trips and 161,467 candidate rows. Nine development trips entered three-fold grouped CV, and three trips were reserved for holdout. The seed was 42. Candidate recall among accepted alignments on those 12 trips was 99.27%.

| Model | OOF accuracy | OOF precision | OOF recall | OOF F1 | OOF log loss | Conditional top-1 |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.9463 | 0.8313 | 0.8213 | 0.8263 | 0.1578 | 0.8455 |
| HistGradientBoosting | 0.9458 | 0.8411 | 0.8036 | 0.8219 | 0.1454 | 0.8471 |
| MLP | 0.9423 | 0.8531 | 0.7600 | 0.8038 | 0.2658 | 0.8409 |
| DecisionTree | 0.9092 | 0.6530 | 0.8883 | 0.7527 | 0.6799 | 0.8101 |
| LogisticRegression | 0.8694 | 0.5490 | 0.8996 | 0.6819 | 0.3088 | 0.8414 |

## Review 1 feature experiments

See [the Review 1 report](MEMBER2_REVIEW1_REPORT.md) for the fixed-model comparisons, feature schema, held-out validation, and reproduction command. The frozen candidate scorer is HistGradientBoosting with 26 features.
