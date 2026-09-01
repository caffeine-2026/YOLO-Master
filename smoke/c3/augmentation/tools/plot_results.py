#!/usr/bin/env python3
"""Generate augmentation figures only from the aggregate CSV evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = [
        FIGURES / "augmentation_strength_accuracy.png",
        FIGURES / "per_class_ap_comparison.png",
        FIGURES / "deeppcb_scaling_accuracy.png",
        FIGURES / "figure_manifest.json",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite result figures")

    search = rows("initial_search.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for axis, dataset in zip(axes, ("neu", "deeppcb"), strict=True):
        subset = sorted((row for row in search if row["dataset"] == dataset), key=lambda row: int(row["strength"]))
        axis.plot(
            [int(row["strength"]) for row in subset],
            [float(row["best_validation_map50_95"]) for row in subset],
            marker="o",
        )
        axis.set_xticks([0, 1, 2, 3], ["none", "mild", "medium", "strong"])
        axis.set_title("NEU-DET" if dataset == "neu" else "DeepPCB")
        axis.set_xlabel("Augmentation strength (validation search)")
        axis.set_ylabel("Best validation mAP50-95")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outputs[0], dpi=180)
    plt.close(fig)

    frozen = json.loads((RESULTS / "frozen_selection.json").read_text(encoding="utf-8"))
    per_class = rows("per_class_test_summary.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for axis, dataset in zip(axes, ("neu", "deeppcb"), strict=True):
        augmented = frozen["datasets"][dataset]["frozen_policy"]
        subset = [
            row
            for row in per_class
            if row["dataset"] == dataset and row["sample_size"] == "100" and row["metric"] == "ap50_95"
        ]
        by_policy: dict[str, dict[str, float]] = defaultdict(dict)
        for row in subset:
            by_policy[row["policy"]][row["class_name"]] = float(row["mean"])
        class_names = list(by_policy["baseline"])
        x = list(range(len(class_names)))
        width = 0.38
        axis.bar(
            [value - width / 2 for value in x],
            [by_policy["baseline"][name] for name in class_names],
            width,
            label="no augmentation",
        )
        if augmented != "baseline":
            axis.bar(
                [value + width / 2 for value in x],
                [by_policy[augmented][name] for name in class_names],
                width,
                label=augmented,
            )
        axis.set_xticks(x, class_names, rotation=20, ha="right")
        axis.set_ylabel("Locked-test AP50-95 (3-seed mean)")
        axis.set_title("NEU-DET" if dataset == "neu" else "DeepPCB")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outputs[1], dpi=180)
    plt.close(fig)
    scaling = rows("scaling_comparison.csv")
    fig, axis = plt.subplots(figsize=(7, 4.5))
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scaling:
        by_method[row["method"]].append(row)
    for method, method_rows in by_method.items():
        ordered = sorted(method_rows, key=lambda row: int(row["sample_size"]))
        axis.plot(
            [int(row["sample_size"]) for row in ordered],
            [float(row["map50_95_mean"]) for row in ordered],
            marker="o",
            label=method,
        )
    axis.set_xscale("log")
    axis.set_xticks([10, 50, 100, 500], ["10", "50", "100", "500"])
    axis.set_xlabel("Training images (shot scale)")
    axis.set_ylabel("Locked-test mAP50-95 (3-seed mean)")
    axis.set_title("DeepPCB scaling after validation-only augmentation freeze")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(outputs[2], dpi=180)
    plt.close(fig)

    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "figures": {
            outputs[0].name: {
                "source": "smoke/c3/augmentation/results/initial_search.csv",
                "source_sha256": sha256(RESULTS / "initial_search.csv"),
                "plotted_rows": search,
            },
            outputs[1].name: {
                "source": "smoke/c3/augmentation/results/per_class_test_summary.csv",
                "source_sha256": sha256(RESULTS / "per_class_test_summary.csv"),
                "filter": {"sample_size": 100, "metric": "ap50_95"},
                "plotted_rows": [
                    row for row in per_class if row["sample_size"] == "100" and row["metric"] == "ap50_95"
                ],
            },
            outputs[2].name: {
                "source": "smoke/c3/augmentation/results/scaling_comparison.csv",
                "source_sha256": sha256(RESULTS / "scaling_comparison.csv"),
                "plotted_rows": scaling,
            },
        },
    }
    outputs[3].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "sources": ["initial_search.csv", "per_class_test_summary.csv"],
                "figures": [path.name for path in outputs[:3]],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
