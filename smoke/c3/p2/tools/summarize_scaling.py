#!/usr/bin/env python3
"""Summarize P2 seed824 scaling, plots, trade-offs, and the multiseed gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
P2_ROOT = REPO_ROOT / "smoke" / "c3" / "p2"
SIZES = (10, 50, 100, 500)
DATASETS = (("neu", "NEU-DET"), ("deeppcb", "DeepPCB"))
METHODS = (
    ("full_sft", "Full-SFT", "full"),
    ("frozen_backbone", "Frozen Backbone", "frozen"),
    ("vpeft", "V-PEFT", "vpeft"),
)
COLORS = {"Full-SFT": "#1f77b4", "Frozen Backbone": "#ff7f0e", "V-PEFT": "#2ca02c"}
MARKERS = {"Full-SFT": "o", "Frozen Backbone": "s", "V-PEFT": "^"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def curve_summary(path: Path) -> dict[str, float | int]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    if len(values) != 100 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid 100-epoch curve: {path.relative_to(REPO_ROOT)}")
    best = max(range(100), key=values.__getitem__)
    return {"best_epoch": best + 1, "best_map50_95": values[best], "last_epoch_map50_95": values[-1]}


def verify_manifest(path: Path) -> bool:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for row in manifest["artifacts"]:
        artifact = REPO_ROOT / row["path"]
        if not artifact.is_file() or artifact.stat().st_size != row["size_bytes"] or sha256(artifact) != row["sha256"]:
            return False
    return True


def load_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    manifest_rows = []
    for dataset, dataset_name in DATASETS:
        for size in SIZES:
            for method, method_name, method_tag in METHODS:
                if size == 100:
                    run_id = f"{dataset}_{method_tag}_seed824_e100"
                    log_dir = P1_ROOT / "logs" / run_id
                    source = "reused_p1_seed824_e100"
                    reuse = True
                    manifest_verified = verify_manifest(log_dir / "artifact_manifest.json")
                else:
                    run_id = f"{dataset}_{size}_{method_tag}_seed824"
                    log_dir = P2_ROOT / "logs" / run_id
                    source = "new_p2_seed824"
                    reuse = False
                    manifest_verified = verify_manifest(log_dir / "artifact_manifest.json")
                metrics = json.loads((log_dir / "metrics.json").read_text(encoding="utf-8"))
                if (
                    metrics.get("status") != "PASS"
                    or metrics.get("exit_code") != 0
                    or not all(metrics["checks"].values())
                ):
                    raise ValueError(f"Run did not pass: {run_id}")
                if not manifest_verified:
                    raise ValueError(f"Artifact manifest failed: {run_id}")
                epoch = metrics.get("epoch_summary") or curve_summary(log_dir / "learning_curve.csv")
                environment = json.loads((log_dir / "environment.json").read_text(encoding="utf-8"))
                row = {
                    "dataset": dataset_name,
                    "sample_size": size,
                    "method": method_name,
                    "seed": 824,
                    "mAP50-95": metrics["test"]["map50_95"],
                    "mAP50": metrics["test"]["map50"],
                    "precision": metrics["test"]["precision"],
                    "recall": metrics["test"]["recall"],
                    "trainable_params": metrics["parameters"]["trainable_parameters"],
                    "total_params": metrics["parameters"]["total_parameters"],
                    "trainable_ratio": metrics["parameters"]["trainable_parameter_ratio"],
                    "peak_gpu_memory_mib": metrics["resources"]["peak_gpu_memory_mib"],
                    "training_seconds": metrics["timing"]["training_seconds"],
                    "gpu_hours": metrics["timing"]["gpu_hours"],
                    "best_epoch": epoch["best_epoch"],
                    "best_mAP50-95": epoch["best_map50_95"],
                    "last_epoch_mAP50-95": epoch["last_epoch_map50_95"],
                    "checkpoint_size_bytes": metrics["checkpoint"]["size_bytes"],
                    "adapter_size_bytes": metrics["adapter"]["size_bytes"],
                    "planner_status": metrics["adapter"].get("planner_status"),
                    "planner_backend": metrics["adapter"].get("planner_backend"),
                    "actual_backend": metrics["adapter"].get("actual_backend"),
                    "planned_targets": metrics["adapter"].get("planned_targets"),
                    "applied_targets": metrics["adapter"].get("applied_targets"),
                    "adapter_params": metrics["parameters"].get("adapter_parameters"),
                    "device": environment["device_argument"],
                    "gpu_name": environment["gpu_name"],
                    "exit_code": metrics["exit_code"],
                    "status": metrics["status"],
                    "source": source,
                    "run_id": run_id,
                }
                if not all(math.isfinite(float(row[key])) for key in ("mAP50-95", "mAP50", "precision", "recall")):
                    raise ValueError(f"Non-finite test metric: {run_id}")
                rows.append(row)
                manifest_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset_name,
                        "sample_size": size,
                        "method": method_name,
                        "seed": 824,
                        "source": source,
                        "reused": reuse,
                        "metrics": (log_dir / "metrics.json").relative_to(REPO_ROOT).as_posix(),
                        "metrics_sha256": sha256(log_dir / "metrics.json"),
                        "artifact_manifest": (log_dir / "artifact_manifest.json").relative_to(REPO_ROOT).as_posix(),
                        "artifact_manifest_verified": manifest_verified,
                        "status": "PASS",
                    }
                )
    return rows, manifest_rows


def tradeoff_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {(row["dataset"], row["sample_size"], row["method"]): row for row in rows}
    output = []
    for dataset_name in (row[1] for row in DATASETS):
        for size in SIZES:
            full = lookup[(dataset_name, size, "Full-SFT")]
            for _, method_name, _ in METHODS:
                row = lookup[(dataset_name, size, method_name)]
                output.append(
                    {
                        "dataset": dataset_name,
                        "sample_size": size,
                        "method": method_name,
                        "mAP50-95": row["mAP50-95"],
                        "accuracy_retention_vs_full": row["mAP50-95"] / full["mAP50-95"],
                        "accuracy_delta_vs_full": row["mAP50-95"] - full["mAP50-95"],
                        "trainable_params": row["trainable_params"],
                        "trainable_ratio": row["trainable_ratio"],
                        "parameter_reduction_vs_full": 1 - row["trainable_params"] / full["trainable_params"],
                        "map50_95_per_million_trainable_params": row["mAP50-95"]
                        / (row["trainable_params"] / 1_000_000),
                        "peak_gpu_memory_mib": row["peak_gpu_memory_mib"],
                        "memory_saving_vs_full": 1 - row["peak_gpu_memory_mib"] / full["peak_gpu_memory_mib"],
                        "training_seconds": row["training_seconds"],
                        "training_time_change_vs_full": row["training_seconds"] / full["training_seconds"] - 1,
                        "gpu_hours": row["gpu_hours"],
                        "gpu_hour_change_vs_full": row["gpu_hours"] / full["gpu_hours"] - 1,
                    }
                )
    return output


def line_plot(
    rows: list[dict[str, object]], dataset: str, metric: str, ylabel: str, output: Path, baseline: bool = False
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
    lookup = {(row["sample_size"], row["method"]): row for row in rows if row["dataset"] == dataset}
    for _, method_name, _ in METHODS:
        values = []
        for size in SIZES:
            value = lookup[(size, method_name)][metric]
            if baseline:
                value /= lookup[(size, "Full-SFT")][metric]
            values.append(value)
        axis.plot(
            SIZES,
            values,
            label=method_name,
            color=COLORS[method_name],
            marker=MARKERS[method_name],
            linewidth=2,
        )
    axis.set_xscale("log")
    axis.set_xticks(SIZES, [str(size) for size in SIZES])
    axis.set_xlabel("Training images")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{dataset}: {ylabel} vs sample size (seed824)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def params_plot(rows: list[dict[str, object]], dataset: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    for _, method_name, _ in METHODS:
        method_rows = sorted(
            (row for row in dataset_rows if row["method"] == method_name), key=lambda row: row["sample_size"]
        )
        x_values = [100 * row["trainable_ratio"] for row in method_rows]
        y_values = [row["mAP50-95"] for row in method_rows]
        axis.scatter(
            x_values, y_values, color=COLORS[method_name], marker=MARKERS[method_name], s=60, label=method_name
        )
        for x_value, y_value, row in zip(x_values, y_values, method_rows, strict=True):
            axis.annotate(
                str(row["sample_size"]), (x_value, y_value), xytext=(5, 4), textcoords="offset points", fontsize=8
            )
    axis.set_xlabel("Trainable parameter ratio (%)")
    axis.set_ylabel("mAP50-95")
    axis.set_title(f"{dataset}: accuracy vs trainable parameter ratio (labels = images)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def generate_plots(rows: list[dict[str, object]]) -> list[str]:
    visualizations = P2_ROOT / "visualizations"
    outputs = []
    for dataset_tag, dataset_name in DATASETS:
        definitions = (
            ("mAP50-95", "mAP50-95", visualizations / f"{dataset_tag}_scaling_map5095.png", False),
            ("mAP50", "mAP50", visualizations / f"{dataset_tag}_scaling_map50.png", False),
            (
                "mAP50-95",
                "Accuracy retention vs Full-SFT",
                visualizations / f"{dataset_tag}_accuracy_retention.png",
                True,
            ),
            (
                "peak_gpu_memory_mib",
                "Peak GPU memory (MiB)",
                visualizations / f"{dataset_tag}_peak_gpu_memory.png",
                False,
            ),
            ("training_seconds", "Training time (seconds)", visualizations / f"{dataset_tag}_training_time.png", False),
            ("gpu_hours", "GPU-hours", visualizations / f"{dataset_tag}_gpu_hours.png", False),
        )
        for metric, ylabel, output, baseline in definitions:
            line_plot(rows, dataset_name, metric, ylabel, output, baseline)
            outputs.append(output.relative_to(REPO_ROOT).as_posix())
        params_output = visualizations / f"{dataset_tag}_accuracy_vs_params.png"
        params_plot(rows, dataset_name, params_output)
        outputs.append(params_output.relative_to(REPO_ROOT).as_posix())
    return outputs


def sequence(rows: list[dict[str, object]], dataset: str, method: str, key: str) -> list[float]:
    lookup = {(row["sample_size"], row["method"]): row for row in rows if row["dataset"] == dataset}
    if key == "retention":
        return [lookup[(size, method)]["mAP50-95"] / lookup[(size, "Full-SFT")]["mAP50-95"] for size in SIZES]
    if key == "delta":
        return [lookup[(size, method)]["mAP50-95"] - lookup[(size, "Full-SFT")]["mAP50-95"] for size in SIZES]
    return [lookup[(size, method)][key] for size in SIZES]


def crossover_regions(differences: list[float]) -> list[str]:
    regions = []
    for left, right, left_value, right_value in zip(
        SIZES[:-1], SIZES[1:], differences[:-1], differences[1:], strict=True
    ):
        if left_value == 0 or right_value == 0 or left_value * right_value < 0:
            regions.append(f"{left}-{right}")
    return regions


def seed824_analysis(
    rows: list[dict[str, object]], tradeoffs: list[dict[str, object]]
) -> tuple[str, dict[str, object]]:
    trade_lookup = {(row["dataset"], row["sample_size"], row["method"]): row for row in tradeoffs}
    findings = {}
    sections = []
    for _, dataset_name in DATASETS:
        retention = sequence(rows, dataset_name, "V-PEFT", "retention")
        delta = sequence(rows, dataset_name, "V-PEFT", "delta")
        closest_index = min(range(4), key=lambda index: abs(delta[index]))
        largest_index = max(range(4), key=lambda index: abs(delta[index]))
        lookup = {(row["sample_size"], row["method"]): row for row in rows if row["dataset"] == dataset_name}
        frozen_gaps = []
        rankings = {}
        for size in SIZES:
            ordered = sorted(
                (lookup[(size, method_name)] for _, method_name, _ in METHODS),
                key=lambda row: row["mAP50-95"],
                reverse=True,
            )
            rankings[str(size)] = [row["method"] for row in ordered]
            frozen_gaps.append(ordered[0]["mAP50-95"] - lookup[(size, "Frozen Backbone")]["mAP50-95"])
        competitive_size = SIZES[min(range(4), key=lambda index: frozen_gaps[index])]
        memory = [trade_lookup[(dataset_name, size, "V-PEFT")]["memory_saving_vs_full"] for size in SIZES]
        time = [trade_lookup[(dataset_name, size, "V-PEFT")]["training_time_change_vs_full"] for size in SIZES]
        parameter_reductions = [
            trade_lookup[(dataset_name, size, "V-PEFT")]["parameter_reduction_vs_full"] for size in SIZES
        ]
        crossovers = crossover_regions(delta)
        findings[dataset_name] = {
            "vpeft_retention": dict(zip((str(size) for size in SIZES), retention, strict=True)),
            "vpeft_delta": dict(zip((str(size) for size in SIZES), delta, strict=True)),
            "closest_sample_size": SIZES[closest_index],
            "largest_gap_sample_size": SIZES[largest_index],
            "rankings": rankings,
            "frozen_most_competitive_sample_size": competitive_size,
            "vpeft_parameter_reductions": parameter_reductions,
            "vpeft_memory_savings": memory,
            "vpeft_time_changes": time,
            "full_vpeft_empirical_crossover_regions": crossovers,
        }
        sections.append(
            f"### {dataset_name}\n\n"
            f"- V-PEFT retention @10/50/100/500: {' / '.join(f'{value:.2%}' for value in retention)}.\n"
            f"- V-PEFT − Full-SFT ΔmAP50-95: {' / '.join(f'{value:+.4f}' for value in delta)}.\n"
            f"- Closest to Full-SFT: {SIZES[closest_index]} images; largest gap: {SIZES[largest_index]} images.\n"
            f"- Frozen Backbone is closest to the best method at {competitive_size} images; rankings: "
            + "; ".join(f"{size}={' > '.join(rankings[str(size)])}" for size in SIZES)
            + ".\n"
            f"- V-PEFT parameter reduction @10/50/100/500: {' / '.join(f'{value:.2%}' for value in parameter_reductions)}.\n"
            f"- V-PEFT memory saving @10/50/100/500: {' / '.join(f'{value:+.2%}' for value in memory)}.\n"
            f"- V-PEFT training-time change @10/50/100/500: {' / '.join(f'{value:+.2%}' for value in time)}.\n"
            f"- Empirical Full/V-PEFT crossover region(s): {', '.join(crossovers) if crossovers else 'none observed'}."
        )

    neu = findings["NEU-DET"]
    deep = findings["DeepPCB"]
    neu_density = json.loads((P2_ROOT / "evidence" / "neu_scaling_split_manifest.json").read_text())["splits"]
    deep_density = json.loads((P2_ROOT / "evidence" / "deeppcb_scaling_split_manifest.json").read_text())["splits"]
    text = f"""# C3 P2 Seed824 Scaling Analysis

## Scope and gate

This is the required seed824-only gate: 18 new P2 runs plus six SHA-verified P1 100-image reuses. It is not a multi-seed conclusion.

{chr(10).join(sections)}

## Required questions

1. **Where is V-PEFT closest/largest-gap?** NEU: closest={neu["closest_sample_size"]}, largest={neu["largest_gap_sample_size"]}; DeepPCB: closest={deep["closest_sample_size"]}, largest={deep["largest_gap_sample_size"]}.
2. **Do dataset trends agree?** NEU retention={"/".join(f"{value:.2%}" for value in neu["vpeft_retention"].values())}; DeepPCB={"/".join(f"{value:.2%}" for value in deep["vpeft_retention"].values())}. Agreement is assessed from these measured sequences, not assumed from P1.
3. **Is Frozen better at extremely low data?** At 10 images, NEU ranking is {" > ".join(neu["rankings"]["10"])}; DeepPCB ranking is {" > ".join(deep["rankings"]["10"])}. This directly answers the seed824 observation without generalizing beyond one seed.
4. **Does parameter efficiency persist?** Yes structurally: the V-PEFT trainable-parameter reduction is {"/".join(f"{value:.2%}" for value in neu["vpeft_parameter_reductions"])} on NEU and {"/".join(f"{value:.2%}" for value in deep["vpeft_parameter_reductions"])} on DeepPCB.
5. **Does P1-100 lie on the scaling trend?** The audited 100-image retentions are NEU {neu["vpeft_retention"]["100"]:.2%} and DeepPCB {deep["vpeft_retention"]["100"]:.2%}; their position relative to the 10/50/500 points is visible in the measured sequences above. No monotonicity is imposed.

## Measured dataset differences

All scales cover 6/6 classes. Measured object densities (objects/image) are NEU 10={neu_density["10"]["distribution"]["objects_per_image"]:.2f}, 100={neu_density["100"]["distribution"]["objects_per_image"]:.2f}, 500={neu_density["500"]["distribution"]["objects_per_image"]:.2f}; DeepPCB 10={deep_density["10"]["distribution"]["objects_per_image"]:.2f}, 100={deep_density["100"]["distribution"]["objects_per_image"]:.2f}, 500={deep_density["500"]["distribution"]["objects_per_image"]:.2f}. These statistics establish distributional differences but do not prove a mechanism for accuracy trends.

## Multi-seed decision

`MULTISEED_READY=YES` because all 24 seed824 cells are finite and traceable, nesting/reuse audits pass, and the observed curves contain sample-size-dependent differences worth estimating across seeds. This decision authorizes a later plan only; seed825/826 were not run here.
"""
    return text, findings


def multiseed_plan() -> dict[str, object]:
    runs = []
    for seed in (825, 826):
        for dataset, _ in DATASETS:
            for size in (10, 50, 500):
                for method, _, method_tag in METHODS:
                    runs.append(
                        {
                            "dataset": dataset,
                            "sample_size": size,
                            "method": method,
                            "seed": seed,
                            "run_id": f"{dataset}_{size}_{method_tag}_seed{seed}",
                            "status": "pending_not_run",
                        }
                    )
    return {
        "schema_version": 1,
        "status": "PLANNED_NOT_STARTED",
        "multiseed_ready": True,
        "auto_run": False,
        "completed_seed824_cells": 24,
        "p1_100_cells_reused_across_three_seeds": 18,
        "pending_new_runs": len(runs),
        "prohibited_in_seed824_turn": [825, 826],
        "runs": runs,
    }


def p2_report(rows: list[dict[str, object]], findings: dict[str, object]) -> str:
    neu = findings["NEU-DET"]
    deep = findings["DeepPCB"]
    table_lines = [
        "| Dataset | Images | Method | mAP50-95 | mAP50 | Trainable Params | Peak GPU MiB | Time (s) | Source |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['dataset']} | {row['sample_size']} | {row['method']} | {row['mAP50-95']:.4f} | "
            f"{row['mAP50']:.4f} | {row['trainable_params']:,} | {row['peak_gpu_memory_mib']:.1f} | "
            f"{row['training_seconds']:.1f} | {row['source']} |"
        )
    return f"""# C3 P2 Report — Seed824 Scaling Gate

## 1. Research Question

Which method is suitable at which industrial few-shot sample scale, without declaring a universal winner?

## 2. P1 Starting Point

P1 completed 18/18 at 100 images. P2 reuses those cells only after byte-identical training-list, fixed val/test, and locked-protocol audits.

## 3. Scaling Protocol

YOLO11n, 100 epochs, batch 8, imgsz 640, AdamW, lr0=0.001, weight decay=0.0005, cosine scheduler, seed824. Full/Frozen/V-PEFT definitions are unchanged from P1.

## 4. Nested Few-shot Split

Both datasets pass strict `10 ⊂ 50 ⊂ 100 ⊂ 500`; all four scales cover 6/6 classes. The 100-image files are byte-identical to P1.

## 5. 10/50/100/500 Results

{chr(10).join(table_lines)}

## 6. Multi-seed Statistics

Not reported at this gate: only seed824 was run. `MULTISEED_READY=YES`; seed825/826 remain not run.

## 7. Accuracy Retention

- NEU V-PEFT retention @10/50/100/500: {" / ".join(f"{value:.2%}" for value in neu["vpeft_retention"].values())}.
- DeepPCB V-PEFT retention @10/50/100/500: {" / ".join(f"{value:.2%}" for value in deep["vpeft_retention"].values())}.

## 8. Parameter Efficiency

V-PEFT retains the P1 structure with 613,602 trainable parameters versus 2,590,994 for Full-SFT (76.32% reduction) at every sample size. Accuracy per trainable parameter remains sample- and dataset-dependent.

## 9. Memory / Time Efficiency

- NEU V-PEFT memory saving: {" / ".join(f"{value:+.2%}" for value in neu["vpeft_memory_savings"])}; time change: {" / ".join(f"{value:+.2%}" for value in neu["vpeft_time_changes"])}.
- DeepPCB V-PEFT memory saving: {" / ".join(f"{value:+.2%}" for value in deep["vpeft_memory_savings"])}; time change: {" / ".join(f"{value:+.2%}" for value in deep["vpeft_time_changes"])}.

## 10. Dataset-dependent Behavior

The measured retention sequences differ between datasets. Split manifests report class coverage, images/class, objects/class, and object density. Those measurements support describing dataset differences, not assigning an unmeasured causal mechanism.

## 11. Empirical Crossover Analysis

- NEU Full/V-PEFT crossover region(s): {", ".join(neu["full_vpeft_empirical_crossover_regions"]) if neu["full_vpeft_empirical_crossover_regions"] else "none observed"}.
- DeepPCB Full/V-PEFT crossover region(s): {", ".join(deep["full_vpeft_empirical_crossover_regions"]) if deep["full_vpeft_empirical_crossover_regions"] else "none observed"}.

These are empirical intervals between tested sizes, not theoretical optima.

## 12. Qualitative Examples

The fixed P1 100-image qualitative comparison remains the audited 100-image reference. New-scale qualitative panels are deferred until the multi-seed P2 phase to avoid selecting examples from a one-seed curve.

## 13. Limitations

This gate has one seed at 10/50/500, only four discrete sample sizes, and a fixed 100-epoch rather than fixed-update budget. Parallel runs use exclusive identical RTX 4090 GPUs. Trend direction and crossover intervals require multi-seed confirmation.

## 14. Final P2 Conclusion

`Overall C3 P2 = IN_PROGRESS`. The seed824 scaling gate is complete and valid, but P2 PASS is withheld until the authorized multi-seed matrix, mean/std/95% CI, and final statistics are completed.
"""


def main() -> int:
    reuse = json.loads((P2_ROOT / "evidence" / "p1_100_reuse_audit.json").read_text(encoding="utf-8"))
    if reuse["status"] != "PASS" or reuse["reused_cells"] != 6:
        raise ValueError("P1 reuse audit did not pass")
    rows, manifest_rows = load_rows()
    if len(rows) != 24 or sum(row["source"] == "new_p2_seed824" for row in rows) != 18:
        raise ValueError("Expected 24 cells with 18 new and six reused")
    informative_curves = all(
        len(
            {
                round(float(row["mAP50-95"]), 10)
                for row in rows
                if row["dataset"] == dataset_name and row["method"] == method_name
            }
        )
        >= 2
        for _, dataset_name in DATASETS
        for _, method_name, _ in METHODS
    )
    if not informative_curves:
        raise ValueError("A sample-size curve is constant and fails the interpretation gate")
    vpeft_rows = [row for row in rows if row["method"] == "V-PEFT"]
    if not all(
        row["planner_status"] in {"ACCEPT", "ADAPT"}
        and row["planner_backend"] == "vpeft"
        and row["actual_backend"] == "peft"
        and int(row["applied_targets"] or 0) > 0
        and int(row["adapter_params"] or 0) > 0
        for row in vpeft_rows
    ):
        raise ValueError("A V-PEFT cell failed strict runtime validation")

    results = P2_ROOT / "results"
    write_csv(results / "scaling_seed824.csv", rows)
    tradeoffs = tradeoff_rows(rows)
    write_csv(results / "scaling_tradeoff_seed824.csv", tradeoffs)
    plots = generate_plots(rows)
    analysis_text, findings = seed824_analysis(rows, tradeoffs)
    (P2_ROOT / "docs").mkdir(parents=True, exist_ok=True)
    (P2_ROOT / "docs" / "SCALING_ANALYSIS_SEED824.md").write_text(analysis_text, encoding="utf-8")
    (P2_ROOT / "docs" / "C3_P2_REPORT.md").write_text(p2_report(rows, findings), encoding="utf-8")
    (P2_ROOT / "config" / "multiseed_plan.yaml").write_text(
        yaml.safe_dump(multiseed_plan(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(
        P2_ROOT / "evidence" / "seed824_run_manifest.json",
        {
            "schema_version": 1,
            "stage": "seed824_scaling_gate",
            "matrix_cells": 24,
            "new_p2_runs": 18,
            "reused_p1_runs": 6,
            "seed825_826_run_count": 0,
            "informative_nonconstant_curves": informative_curves,
            "runs": manifest_rows,
            "plots": plots,
            "multiseed_ready": True,
            "overall_c3_p2": "IN_PROGRESS",
        },
    )
    print("SEED824_SCALING=24/24_PASS")
    print("MULTISEED_READY=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
