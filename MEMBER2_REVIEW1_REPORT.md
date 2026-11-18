# PathSnap Member 2 Review 1

The frozen development selection is **HistGradientBoosting with 26 features**. The strongest original model reached 78.41% candidate top-1. The selected model reached 84.44% on the same folds.

## Evaluation protocol

The original 80-trip dataset contained 1,179,915 candidate rows. Seed 42 kept the original 60 development trips (834,209 rows) and 20 held-out trips (345,706 rows). Three-fold GroupKFold kept each trip intact. Feature selection used development out-of-fold HistGradientBoosting top-1. The five classifiers then used their existing fixed model configurations. No tuning was performed.
The student and teacher test sets were excluded from this experiment.

## Why candidate choice is difficult

Picking the nearest arc reached 55.32% among retrieved observations. The reference arc ranked second by distance in 47,699 development observations. The two nearest candidates were within 1 m of each other in 98,841 of 131,267 observations with at least two candidates. Nearest-arc accuracy was 46.56% for near ties and 81.55% for larger margins. Nearby and opposite-direction arcs can be difficult to distinguish from distance alone. Centered and forward headings add local direction, step lengths and time gaps indicate motion reliability, and directed candidate connections and adjacent-point distances capture short-range route continuity.

## Development feature ablation

| Feature set | Candidate top-1 | All-GPS top-1 | F1 |
|---|---:|---:|---:|
| baseline | 78.41% | 66.65% | 0.7682 |
| relative_motion | 83.97% | 71.38% | 0.8305 |
| full_context | 84.44% | 71.77% | 0.8336 |

## Fixed-model comparison

| Model | Original top-1 | Review 1 top-1 | Change (pp) | Original F1 | Review 1 F1 | Original all-GPS | Review 1 all-GPS | Mean fold train top-1 | Mean fold validation top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 78.41% | 84.44% | +6.02 | 0.7682 | 0.8336 | 66.65% | 71.77% | 87.21% | 84.36% |
| RandomForest | 78.04% | 84.38% | +6.34 | 0.7734 | 0.8368 | 66.33% | 71.72% | 99.34% | 84.30% |
| MLP | 78.01% | 83.13% | +5.12 | 0.7632 | 0.8167 | 66.31% | 70.66% | 87.47% | 83.11% |
| DecisionTree | 77.49% | 82.97% | +5.48 | 0.7264 | 0.7800 | 65.87% | 70.53% | 86.55% | 82.87% |
| LogisticRegression | 77.26% | 82.73% | +5.47 | 0.6684 | 0.7438 | 65.67% | 70.32% | 82.86% | 82.64% |

The selected model's mean train–validation gap is 2.84 percentage points. Its per-trip conditional top-1 ranges from 52.00% to 98.45%, with a median of 86.58%. A train–validation gap suggests sensitivity to trip-specific patterns. Low scores on both partitions suggest the fixed model may underfit. Random Forest reached 99.34% in training and 84.30% in validation, the clearest sign of overfitting in this comparison. Logistic Regression reached 82.86% in training and 82.64% in validation. Its smaller gap and lower training score are consistent with limited model capacity. Per-trip CSVs document the variation for each classifier.

## Held-out check after freezing selection

On the 20 held-out trips, HistGradientBoosting with original features reached 79.15% conditional top-1 and 78.12% all-GPS top-1. The frozen features reached 84.90% conditional top-1, 83.79% all-GPS top-1 and 0.8421 F1 (original F1 0.7759).

Candidate recall among reference-aligned GPS observations was 98.38% on development and 99.32% on holdout. Conditional top-1 uses observations with a retrieved reference candidate. All-GPS top-1 divides confirmed correct top candidates by every raw observation. Observations missing a retrieved reference candidate or a usable reference alignment contribute zero confirmed matches. The metric does not estimate correctness for unaligned observations.

## Frozen schema and reproduction

Features in order

`distance_m`, `heading_diff`, `speed_mps`, `time_gap_s`, `gps_step_m`, `segment_length_m`, `candidate_rank`, `segment_fraction`, `distance_excess_m`, `distance_share`, `nearest_margin_m`, `heading_excess_deg`, `candidate_count`, `center_heading_diff`, `forward_heading_diff`, `heading_turn`, `log_previous_step`, `log_next_step`, `log_next_gap`, `endpoint_distance_m`, `source_indegree`, `target_outdegree`, `previous_segment_distance_m`, `next_segment_distance_m`, `previous_candidate_connection_m`, `next_candidate_connection_m`.

Run `python -m models.review1_experiments --baseline <original results dir> --train <Member 1 CSV> --out <new results dir>` with the Member 1 package available and PATHSNAP_DATA_DIR set. The experiment recomputes features on the full inference candidate set before joining exported training labels, including neighboring GPS observations and neighboring candidates. This requires the whole trip at inference.

Load the trusted `review1_model.pkl` with joblib and call `score_review1(prepared_trip, artifact)`. The release manifest records hashes and dependency versions. Save any future tuned run separately.
