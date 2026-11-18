# PathSnap Review 1 extended features

The development winner was **RandomForest with 40 features** at **89.9485% candidate top-1**. The frozen Review 1 benchmark was 84.4373%.

## Fixed evaluation protocol

Reused the original 60 development and 20 holdout trips and the exact three trip-wise development folds. Candidate keys, labels, candidate generation and scoring tie-break rules followed the original benchmark. Model and feature selection used development out-of-fold top-1. The holdout was evaluated after selection. The student-test and teacher-test sets were excluded. All five classifiers used their existing fixed settings. Random Forest used eight workers.

## Baseline errors

The frozen 26-feature HistGradientBoosting scores below are out of fold. The proxies overlap and use the two nearest retrieved arcs. A shared endpoint within 12 m approximates intersections. The parallel proxy uses distinct endpoints and bearings within 15 degrees modulo 180 degrees. The turn proxy uses a GPS heading change of at least 30 degrees. The ambiguity proxy uses a distance difference of at most 1 m.

| Proxy | Observations | Incorrect | Baseline top-1 |
|---|---:|---:|---:|
| Intersection | 48,387 | 10,554 | 78.1884% |
| Parallel | 19,038 | 1,256 | 93.4027% |
| Turn | 84,031 | 17,490 | 79.1863% |
| Ambiguous | 98,841 | 17,342 | 82.4546% |

The intersection and turn slices were weaker than the overall baseline. The parallel proxy was easier on average, although it still contained 1,256 errors. These categories are inferred from geometry and GPS motion.

## Feature ablation and group-aware ranking

The temporal extension adds signed turn and motion, arc progress and distance remaining, signed lateral offset, wider GPS-window heading and straightness, and relative heading agreement. The directed lattice links same-arc and shared-node candidates at neighboring samples, combines heading and distance evidence, and uses forward/backward minimum costs to estimate each candidate's support. Features are built from all inference candidates before joining the training labels. The scorer requires the whole trip.

| HistGradientBoosting feature set | Conditional top-1 | All-GPS top-1 | F1 |
|---|---:|---:|---:|
| full_context | 84.4373% | 71.7719% | 0.8336 |
| review1_extended_temporal | 87.5598% | 74.4261% | 0.8656 |
| review1_extended_lattice | 89.5762% | 76.1400% | 0.8933 |

The label-free lattice alone ranked the correct candidate first in 115,447 of 131,622 development observations (87.7110%). The best development result used the lattice costs and ranks as classifier inputs.

## All five classifiers

| Model | Frozen Review 1 top-1 | Extended top-1 | Change (pp) | F1 before | F1 after | All-GPS top-1 after | Mean train top-1 | Mean validation top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 84.3795% | 89.9485% | +5.5690 | 0.8368 | 0.8988 | 76.4564% | 99.4452% | 89.7548% |
| HistGradientBoosting | 84.4373% | 89.5762% | +5.1390 | 0.8336 | 0.8933 | 76.1400% | 93.0983% | 89.4460% |
| LogisticRegression | 82.7293% | 88.6630% | +5.9337 | 0.7438 | 0.8178 | 75.3637% | 88.8154% | 88.4549% |
| MLP | 83.1282% | 88.3864% | +5.2582 | 0.8167 | 0.8731 | 75.1287% | 93.8682% | 88.2493% |
| DecisionTree | 82.9747% | 87.7642% | +4.7895 | 0.7800 | 0.8454 | 74.5998% | 92.7008% | 87.6682% |

The selected model's mean fold gap was 9.69 percentage points. Its per-trip conditional top-1 ranged from 52.77% to 99.44% (median 91.74%). Random Forest's larger 9.69-point gap indicates greater sensitivity to its training trips. The linear model's lower training score provides evidence of a capacity limit under the fixed configuration. Per-trip and fold records are included for all five classifiers.

## Holdout and integration

The frozen Review 1 model reached 84.8998% conditional top-1, 83.7888% all-GPS top-1 and 0.8421 F1. The extended RandomForest reached 89.2258% conditional top-1, 88.0582% all-GPS top-1 and 0.8931 F1 on the same holdout.

Candidate recall among aligned observations stayed at 98.3825% for development and 99.3218% for holdout. Conditional top-1 uses observations with a retrieved reference candidate. All-GPS top-1 divides confirmed correct winners by every raw observation. Candidate misses and unaligned observations count as zero confirmed matches.

The saved model produced finite scores for 9,611 candidates on an original holdout trip. It preserved candidate IDs, order and GPS groups, and Member 3's actual Viterbi decoder reconstructed the trip. Member 4's FastAPI /predict handler accepted the combined pipeline through a test-only adapter. The sandbox blocked HTTP transport startup, and the production adapter remains a stub.

## Reproduction and scoring

Run `python -m models.review1_extended_experiments --baseline <original 80-trip results> --train <Member 1 CSV> --out <new extended results dir>` with `PATHSNAP_DATA_DIR` and the Member 1 preprocessing package available. Use `python -m models.review1_extended_audit` with the same paths to reproduce the baseline error slices.

Load the trusted `review1_model.pkl` from the extended results and pass it to `score_review1(prepared_trip, artifact)`. The manifest records versions, checksums, and the frozen feature order. The original Review 1 model and results remain available separately.
