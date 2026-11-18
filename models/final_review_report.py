"""Generate the final-review report, comparison plots and release checksums."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from models.final_review_tuning import FAMILIES, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/member2/final_review"))
    args = parser.parse_args()
    out = args.out
    baseline = pd.read_csv(args.frozen / "comparison.csv").set_index("model")
    review1 = pd.read_csv(args.frozen.parent / "member2-review1" / "comparison.csv").set_index("model")
    original = pd.read_csv(args.frozen.parent / "member2-full" / "model_comparison.csv").set_index("model")
    tuned = pd.read_csv(out / "model_comparison.csv").set_index("model")
    trials = pd.read_csv(out / "tuning_trials.csv")
    best = json.loads((out / "best_hyperparameters.json").read_text())
    selected = json.loads((out / "selection.json").read_text())
    hold = json.loads((out / "holdout_evaluation.json").read_text())
    smoke = json.loads((out / "student_smoke.json").read_text())
    generalization = json.loads((out / "generalization_report.json").read_text())
    final = joblib.load(out / "final_model.pkl")
    evaluation = joblib.load(out / "evaluation_model.pkl")
    cfg = json.loads((out / "evaluation_configuration.json").read_text())
    assert len(final["training_trips"]) == 80 and len(evaluation["training_trips"]) == 60
    assert set(evaluation["training_trips"]) == set(cfg["development_trips"])
    assert set(final["training_trips"]) == set(cfg["development_trips"]) | set(cfg["holdout_trips"])
    assert final["feature_columns"] == evaluation["feature_columns"] and len(final["feature_columns"]) == 40
    dependencies = {name: version(name) for name in
                    ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyproj",
                     "optuna", "matplotlib", "fastapi", "starlette"]}
    (out / "requirements_lock.txt").write_text(
        "\n".join(f"{n}=={v}" for n, v in dependencies.items()) + "\n", encoding="utf-8")

    comparison = pd.DataFrame({
        "original_8_top1": original.top1_accuracy, "review1_26_top1": review1.top1_accuracy,
        "extended_40_top1": baseline.top1_accuracy, "tuned_40_top1": tuned.top1_accuracy,
        "tuned_40_f1": tuned.f1, "tuned_40_gap": tuned.gap,
    })
    comparison["improvement_pp"] = (comparison.tuned_40_top1 - comparison.extended_40_top1) * 100
    comparison.to_csv(out / "before_after_tuning.csv")
    ax = (comparison.iloc[:, :4] * 100).plot.bar(figsize=(10, 5), rot=14, ylim=(70, 94))
    ax.set_ylabel("Development candidate top-1 (%)")
    ax.legend(["8 features", "26 features", "40 features", "40 features tuned"], loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "review_comparison.png", dpi=140)
    plt.close(ax.figure)
    curve = pd.read_csv(out / "learning_curve.csv").set_index("training_trips")
    ax = (curve * 100).plot(marker="o")
    ax.set(xlabel="Development training trips in fold 1", ylabel="Candidate top-1 (%)")
    ax.figure.savefig(out / "learning_curve.png", dpi=140, bbox_inches="tight")
    plt.close(ax.figure)
    matrices = out / "confusion_matrices"
    matrices.mkdir(exist_ok=True)
    for name, row in [*tuned.iterrows(), ("selected_holdout", hold)]:
        display = ConfusionMatrixDisplay(np.array([[int(row["tn"]), int(row["fp"])],
                                                   [int(row["fn"]), int(row["tp"])]]))
        display.plot(colorbar=False).ax_.set_title(name)
        display.figure_.savefig(matrices / f"{name}.png", dpi=130, bbox_inches="tight")
        plt.close(display.figure_)

    counts = trials.groupby(["model", "state"]).size().unstack(fill_value=0)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = dict(selected=selected, estimator_parameters=final["estimator_parameters"],
                    seed=cfg["seed"], schema=final["feature_columns"], dataset_sha256=cfg["training_sha256"],
                    frozen_manifest_sha256=digest(args.frozen / "manifest.json"),
                    evaluation_fit_seconds=evaluation["fit_seconds"],
                    production_fit_seconds=final["fit_seconds"], evaluation_trips=60,
                    production_trips=80, dependency_versions=dependencies, source_commit=source)
    write_json(out / "model_metadata.json", metadata)
    name, configuration = selected["model"], selected["configuration"]
    passed = sum(r["status"] == "passed" for r in smoke)
    lines = [
        "# PathSnap Member 2 final review", "",
        f"Selected **{name} ({configuration})** with 40 features. Development OOF candidate top-1 was "
        f"**{selected['top1_accuracy']:.4%}** and F1 was **{selected['f1']:.4f}**. "
        f"The evaluation artifact reached **{hold['top1_accuracy']:.4%}** conditional top-1 on the internal holdout.",
        "", "## Data and evaluation", "",
        "The existing 80 training trips contain 1,179,915 candidate rows. The frozen split contains "
        "60 development trips with 834,209 candidate rows and 20 internal holdout trips with 345,706 rows. "
        "The seed is 42 and the frozen three GroupKFold validation trip lists are included. "
        "The internal holdout was evaluated in earlier reviews and is no longer fully pristine. "
        "It was revisited only after freezing selection in this release. The two external test sets were excluded "
        "from fitting and selection. The student set was used for reference-free inference checks after export.",
        "",
        "Conditional top-1 uses aligned GPS observations with a retrieved reference candidate. "
        "All-GPS top-1 counts candidate misses and unaligned observations as zero confirmed matches. "
        "Scores tie by ascending arc ID. Row F1, accuracy, precision and recall use the fixed 0.5 threshold.",
        "", "## Feature and tuning comparison", "",
        "| Classifier | 8-feature top-1 | 26-feature top-1 | 40-feature baseline | Tuned top-1 | Change (pp) | Tuned F1 | Gap (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in FAMILIES:
        row = comparison.loc[label]
        prefix = (f"| {label} | {row.original_8_top1:.4%} | {row.review1_26_top1:.4%} | "
                  f"{row.extended_40_top1:.4%} |")
        lines.append(prefix + (f" {row.tuned_40_top1:.4%} | {row.improvement_pp:+.4f} | "
                                f"{row.tuned_40_f1:.4f} | {100*row.tuned_40_gap:.2f} |"
                               if pd.notna(row.tuned_40_top1) else " Untuned | — | — | — |"))
    lines += [
        "", "The 26-feature review introduced relative candidate geometry, heading and local network context. "
        "The 40-feature extension added temporal geometry and a forward/backward directed candidate lattice. "
        "This review freezes those features and searches estimator settings.", "",
        "## Optuna studies", "",
        "Seeded TPE used three startup trials per family and median pruning based on pooled development fold results. "
        "Estimators and imputers were fit inside each fold. HistGradientBoosting early stopping was disabled, so "
        "there is no internal validation split. RF explored depth, tree count, leaf size, split size, "
        "feature and row subsampling, leaf caps, class weighting and conditional bootstrap settings. "
        "Boosting explored learning rate, iteration count, leaf count, depth, minimum leaf size, L2 and bins. "
        "The other searches covered solver and penalty, tree pruning, and MLP size, learning rate and regularization.",
        "", "| Family | Completed | Pruned | Failed |", "|---|---:|---:|---:|",
    ]
    for family, row in counts.iterrows():
        lines.append(f"| {family} | {int(row.get('COMPLETE', 0))} | {int(row.get('PRUNED', 0))} | {int(row.get('FAIL', 0))} |")
    lines += [
        "", "The full folds contain hundreds of thousands of rows and the forest trials take several minutes per fit. "
        "The bounded trial budgets prioritize the leading families. The SQLite study is resumable before selection. "
        "Trial duration, intermediate fold scores and configurations are in the CSV and study database.",
        "",
        "Within 0.03 percentage points of a family's best pooled top-1, macro trip top-1 breaks near ties, "
        "followed by F1 and pooled top-1. Final selection also considers fold variability and the train–validation gap, "
        "with a floor at the best frozen untuned top-1. The holdout is outside the selection rule.",
        "", "### Development-selected hyperparameters by family", "",
    ]
    for family, result in best.items():
        lines.append(f"- **{family}**, trial {result['trial']}, "
                     + ", ".join(f"`{k}={v}`" for k, v in result["params"].items()))
    lines += [
        "", "## Generalization", "",
        f"The selected mean fold train–validation gap was **{100*selected['gap']:.2f} percentage points**. "
        "The frozen Random Forest gap was approximately 9.69 points and the frozen boosting gap was 3.65 points. "
        f"Positive candidates represented {generalization['positive_candidate_fraction']:.2%} of development rows. "
        f"The median candidate group contained {generalization['candidates_per_gps_quantiles'][2]:.0f} candidates. "
        "RF class weighting was varied while candidate generation and trip grouping were fixed.",
        "",
        "Development learning curves, fold variation, hard trips, correlations and grouped feature permutation "
        "diagnostics are included. Permuting a feature family breaks some cross-family relationships, so treat "
        "the resulting drop as a sensitivity check. The error audit covers intersection, turn, parallel-road "
        "and near-tie proxies from the frozen benchmark.",
        "",
        f"The selected model's mean per-trip OOF top-1 was {selected['mean_trip_top1']:.4%}. "
        f"Internal holdout all-GPS top-1 was **{hold['all_gps_top1']:.4%}**, F1 **{hold['f1']:.4f}**, "
        f"and recall of reference candidates **{hold['candidate_recall']:.4%}** on aligned observations. "
        "The full-80 production refit shares the selected settings and includes the 20 internal evaluation trips.",
        "", "## Artifacts and handoff", "",
        "- `results/member2/final_review/evaluation_model.pkl` is the 60-trip fit used for the holdout report.",
        "- `results/member2/final_review/final_model.pkl` is the 80-trip production fit.",
        "- `feature_schema.json`, `model_metadata.json` and `checksums.json` record feature order, exact estimator "
        "parameters, dependencies, split and hashes.",
        "",
        f"Reference-free student trips completed successfully **{passed}/{len(smoke)}**. Each outcome is in `student_smoke.json`.",
        "",
        "```python",
        "from models.final_review_integration import create_predict_trip, load_final_model",
        "from models.score_candidates import score_review1",
        "",
        'artifact = load_final_model("results/member2/final_review/final_model.pkl")',
        "scored = score_review1(prepared_trip, artifact)",
        '# Wire this into Member 4’s predict_trip(folder, trip_id) entry point.',
        'predict_trip = create_predict_trip("results/member2/final_review/final_model.pkl")',
        "```",
        "",
        "The factory loads a trusted model once per worker, prepares the full trip, and passes candidate scores "
        "to Member 3's directed Viterbi decoder. It accepts the trip folder and ID from Member 4 and returns "
        "predicted arcs and latitude/longitude geometry. It needs Member 1 preprocessing and Member 3 reconstruction "
        "available on the import path. The production backend branch still contains its original adapter stub. "
        "The combined handler was exercised with the opt-in adapter. HTTP TestClient startup was blocked by the sandbox socket restriction.",
        "", "## Pull request handoff", "",
        "PR #4 targets main. GitHub reports a merge conflict. Both branches appended ignore rules "
        "at the same place in .gitignore. When syncing main into member2-ml, keep both "
        "`results/member1/` and `results/member2/` and one `.pytest_cache/` entry. Keep teammates’ "
        "preprocessing changes from main. Run the tests after the branch merge.", "",
        "## Reproduction", "",
        "Use the package versions in `requirements_lock.txt`. Reuse the signed-off frozen inputs and original dataset paths. "
        "A sample command for each family is below. Set a total attempt budget per study. Use `--recover-interrupted` "
        "only after confirming that the previous worker stopped. Resume before freezing final selection.",
        "",
        "```text",
        "python -m models.final_review_tuning --train <training CSV> --frozen <extended Review 1 results> --family HistGradientBoosting --trials 12",
        "python -m models.final_review_tuning --train <training CSV> --frozen <extended Review 1 results> --finalize",
        "python -m models.final_review_tuning --train <training CSV> --frozen <extended Review 1 results> --diagnostics",
        "python -m models.final_review_smoke --model results/member2/final_review/final_model.pkl --out results/member2/final_review/student_smoke.json",
        "```", "",
    ]
    out_report = "\n".join(lines)
    readme = Path("README_MEMBER2.md")
    base = readme.read_text(encoding="utf-8").split("## Final review")[0].rstrip()
    readme.write_text(base + "\n\n## Final review\n\n"
                      "[Final benchmark and handoff](MEMBER2_FINAL_REVIEW_REPORT.md). "
                      f"The production scorer is {name} with 40 features.\n", encoding="utf-8")
    (out / "MEMBER2_FINAL_REVIEW_REPORT.md").write_text(out_report, encoding="utf-8")
    Path("MEMBER2_FINAL_REVIEW_REPORT.md").write_text(out_report, encoding="utf-8")
    checksums = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob("*")
                 if p.is_file() and p.name != "checksums.json" and "feature_cache" not in p.parts}
    write_json(out / "checksums.json", checksums)
    print(comparison.to_string())
    print(counts.to_string())


if __name__ == "__main__":
    main()
