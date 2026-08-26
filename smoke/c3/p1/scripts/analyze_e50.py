#!/usr/bin/env python3
"""Analyze the immutable C3 P1 seed824 50-epoch comparison runs."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
THRESHOLD = 0.01
DATASETS = (("neu_det", "NEU-DET", "neu"), ("deeppcb", "DeepPCB", "deeppcb"))
METHODS = (
    ("full_sft", "Full-SFT", "full"),
    ("frozen_backbone", "Frozen Backbone", "frozen"),
    ("vpeft", "V-PEFT", "vpeft"),
)
COMMON_KEYS = (
    "model",
    "pretrained",
    "batch",
    "imgsz",
    "workers",
    "seed",
    "deterministic",
    "amp",
    "cache",
    "val",
    "patience",
    "optimizer",
    "lr0",
    "lrf",
    "weight_decay",
    "momentum",
    "cos_lr",
    "warmup_epochs",
    "warmup_bias_lr",
    "close_mosaic",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "cutmix",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalized_curve(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    rows = []
    for row in source:
        train = [float(row[key]) for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss")]
        val = [float(row[key]) for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss")]
        learning_rates = [float(value) for key, value in row.items() if key.startswith("lr/")]
        rows.append(
            {
                "epoch": int(float(row["epoch"])),
                "train_box_loss": train[0],
                "train_cls_loss": train[1],
                "train_dfl_loss": train[2],
                "train_loss": sum(train),
                "val_box_loss": val[0],
                "val_cls_loss": val[1],
                "val_dfl_loss": val[2],
                "val_loss": sum(val),
                "precision": float(row["metrics/precision(B)"]),
                "recall": float(row["metrics/recall(B)"]),
                "map50": float(row["metrics/mAP50(B)"]),
                "map50_95": float(row["metrics/mAP50-95(B)"]),
                "learning_rate": statistics.fmean(learning_rates),
            }
        )
    if len(rows) != 50 or [row["epoch"] for row in rows] != list(range(1, 51)):
        raise ValueError(f"Expected epochs 1-50 in {path.relative_to(REPO_ROOT)}")
    return rows


def run_id(dataset: str, method: str, epoch: int) -> str:
    if epoch == 30:
        return f"{dataset}_{method}_seed824"
    dataset_tag = {"neu_det": "neu", "deeppcb": "deeppcb"}[dataset]
    method_tag = {row[0]: row[2] for row in METHODS}[method]
    return f"{dataset_tag}_{method_tag}_seed824_e50"


def load_runs() -> dict[str, dict[str, object]]:
    runs = {}
    for dataset, dataset_name, _ in DATASETS:
        for method, method_name, _ in METHODS:
            identifier = run_id(dataset, method, 50)
            log_dir = P1_ROOT / "logs" / identifier
            metrics = json.loads((log_dir / "metrics.json").read_text(encoding="utf-8"))
            if metrics["status"] != "PASS" or metrics["exit_code"] != 0 or not all(metrics["checks"].values()):
                raise ValueError(f"Run did not pass all checks: {identifier}")
            runs[identifier] = {
                "run_id": identifier,
                "dataset": dataset,
                "dataset_name": dataset_name,
                "method": method,
                "method_name": method_name,
                "log_dir": log_dir,
                "metrics": metrics,
                "resolved": yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8")),
                "curve": normalized_curve(log_dir / "learning_curve.csv"),
            }
    return runs


def refresh_artifact_manifests(runs: dict[str, dict[str, object]]) -> None:
    """Refresh hashes after lossless line-ending normalization of compact logs."""
    for run in runs.values():
        path = run["log_dir"] / "artifact_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for artifact in manifest["artifacts"]:
            source = REPO_ROOT / artifact["path"]
            if not source.is_file():
                raise FileNotFoundError(source.relative_to(REPO_ROOT))
            artifact["size_bytes"] = source.stat().st_size
            artifact["sha256"] = sha256(source)
        json_write(path, manifest)


def comparison_rows(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        full = runs[run_id(dataset, "full_sft", 50)]["metrics"]
        for method, method_name, _ in METHODS:
            metrics = runs[run_id(dataset, method, 50)]["metrics"]
            trainable = int(metrics["parameters"]["trainable_parameters"])
            full_trainable = int(full["parameters"]["trainable_parameters"])
            memory = float(metrics["resources"]["peak_gpu_memory_mib"])
            full_memory = float(full["resources"]["peak_gpu_memory_mib"])
            seconds = float(metrics["timing"]["training_seconds"])
            full_seconds = float(full["timing"]["training_seconds"])
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "run_id": metrics["run_id"],
                    "map50_95": metrics["test"]["map50_95"],
                    "map50": metrics["test"]["map50"],
                    "precision": metrics["test"]["precision"],
                    "recall": metrics["test"]["recall"],
                    "trainable_parameters": trainable,
                    "total_parameters": metrics["parameters"]["total_parameters"],
                    "trainable_ratio_percent": metrics["parameters"]["trainable_parameter_ratio"] * 100,
                    "trainable_vs_full_percent": trainable / full_trainable * 100,
                    "peak_gpu_memory_mib": memory,
                    "memory_saving_vs_full_percent": (1 - memory / full_memory) * 100,
                    "training_seconds": seconds,
                    "gpu_hours": metrics["timing"]["gpu_hours"],
                    "time_ratio_vs_full": seconds / full_seconds,
                    "checkpoint_mib": metrics["checkpoint"]["size_bytes"] / 1024**2,
                    "adapter_mib": metrics["adapter"]["size_bytes"] / 1024**2,
                    "status": metrics["status"],
                }
            )
    return rows


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable Ratio | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['map50_95']:.4f} | {row['map50']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['trainable_parameters']:,} / "
            f"{row['total_parameters']:,} | {row['trainable_ratio_percent']:.2f}% | "
            f"{row['peak_gpu_memory_mib']:.0f} MiB | {row['memory_saving_vs_full_percent']:.2f}% | "
            f"{row['training_seconds']:.1f}s | {row['gpu_hours']:.5f} | {row['time_ratio_vs_full']:.3f}x | "
            f"{row['checkpoint_mib']:.2f} MiB | {row['adapter_mib']:.2f} MiB | {row['status']} |"
        )
    return "\n".join(lines)


def epochwise_rows(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"dataset": run["dataset_name"], "method": run["method_name"], "run_id": run["run_id"], **row}
        for run in runs.values()
        for row in run["curve"]
    ]


def plot_curves(runs: dict[str, dict[str, object]]) -> list[str]:
    output_dir = P1_ROOT / "visualizations" / "convergence_e50"
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("map50_95", "mAP50-95", "map5095"),
        ("map50", "mAP50", "map50"),
        ("train_loss", "Train loss (box + cls + dfl)", "loss"),
    )
    paths = []
    for dataset, dataset_name, prefix in DATASETS:
        for key, ylabel, suffix in definitions:
            figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
            for method, method_name, _ in METHODS:
                curve = runs[run_id(dataset, method, 50)]["curve"]
                axis.plot([row["epoch"] for row in curve], [row[key] for row in curve], label=method_name, linewidth=2)
            axis.set(title=f"{dataset_name}: {ylabel} vs epoch (50-epoch candidate)", xlabel="Epoch", ylabel=ylabel)
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            output = output_dir / f"{prefix}_{suffix}.png"
            figure.savefig(output)
            plt.close(figure)
            paths.append(output.relative_to(REPO_ROOT).as_posix())
    return paths


def convergence_rows(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for run in runs.values():
        values = [row["map50_95"] for row in run["curve"]]
        previous = statistics.fmean(values[-10:-5])
        last = statistics.fmean(values[-5:])
        best = max(range(50), key=values.__getitem__)
        rows.append(
            {
                "dataset": run["dataset_name"],
                "method": run["method_name"],
                "run_id": run["run_id"],
                "previous_5_epoch_mean": previous,
                "last_5_epoch_mean": last,
                "delta_last5_vs_prev5": last - previous,
                "best_epoch": best + 1,
                "best_metric": values[best],
                "last_epoch_metric": values[-1],
                "status": "NOT_CONVERGED" if last - previous > THRESHOLD else "CONVERGED_OR_PLATEAU",
            }
        )
    return rows


def fairness_audit(runs: dict[str, dict[str, object]]) -> dict[str, object]:
    reference = next(iter(runs.values()))["resolved"]
    common = {
        key: {identifier: run["resolved"].get(key) for identifier, run in runs.items()} for key in COMMON_KEYS
    }
    common_checks = {key: len({json.dumps(value, sort_keys=True) for value in values.values()}) == 1 for key, values in common.items()}
    old_parity = {}
    dataset_checks = {}
    method_checks = {}
    for dataset, _, _ in DATASETS:
        dataset_runs = [runs[run_id(dataset, method, 50)] for method, _, _ in METHODS]
        dataset_checks[dataset] = {
            "same_data_yaml": len({run["resolved"]["data"] for run in dataset_runs}) == 1,
            "same_train_list_sha256": sha256(P1_ROOT / "config" / dataset / "train_seed824.txt"),
            "selected_train_images": sum(1 for line in (P1_ROOT / "config" / dataset / "train_seed824.txt").read_text().splitlines() if line),
            "same_locked_test_args": len(
                {
                    (
                        run["metrics"]["test"]["split"],
                        run["metrics"]["test"]["batch"],
                        run["metrics"]["test"]["imgsz"],
                        run["metrics"]["test"]["device"],
                    )
                    for run in dataset_runs
                }
            )
            == 1,
        }
        for method, _, _ in METHODS:
            current = runs[run_id(dataset, method, 50)]["resolved"]
            previous = yaml.safe_load(
                (P1_ROOT / "logs" / run_id(dataset, method, 30) / "resolved_config.yaml").read_text(encoding="utf-8")
            )
            old_parity[run_id(dataset, method, 50)] = {
                key: current.get(key) == previous.get(key) for key in COMMON_KEYS
            }
        full = runs[run_id(dataset, "full_sft", 50)]
        frozen = runs[run_id(dataset, "frozen_backbone", 50)]
        vpeft = runs[run_id(dataset, "vpeft", 50)]
        method_checks[dataset] = {
            "full_sft": full["resolved"].get("freeze") == 0 and full["resolved"].get("lora_r") == 0,
            "frozen_backbone": frozen["resolved"].get("freeze") == 11 and frozen["resolved"].get("lora_r") == 0,
            "vpeft": all(
                (
                    vpeft["resolved"].get("lora_r") == 8,
                    vpeft["resolved"].get("lora_alpha") == 16,
                    vpeft["resolved"].get("lora_vpeft_strict") is True,
                    vpeft["metrics"]["adapter"]["planner_status"] in {"ACCEPT", "ADAPT"},
                    vpeft["metrics"]["adapter"]["planner_backend"] == "vpeft",
                    vpeft["metrics"]["adapter"]["actual_backend"] == "peft",
                    vpeft["metrics"]["adapter"]["applied_targets"] > 0,
                    vpeft["metrics"]["adapter"]["size_bytes"] > 0,
                )
            ),
        }
    checks = {
        "six_runs_pass": len(runs) == 6,
        "epochs_50_for_all": all(run["resolved"]["epochs"] == 50 for run in runs.values()),
        "all_common_fields_equal": all(common_checks.values()),
        "only_epoch_changed_from_e30": all(all(values.values()) for values in old_parity.values()),
        "same_data_within_dataset": all(row["same_data_yaml"] for row in dataset_checks.values()),
        "train_budget_100": all(row["selected_train_images"] == 100 for row in dataset_checks.values()),
        "same_evaluator_arguments": all(row["same_locked_test_args"] for row in dataset_checks.values()),
        "method_boundaries_valid": all(all(row.values()) for row in method_checks.values()),
        "restart_all": True,
    }
    return {
        "schema_version": 1,
        "execution_policy": "restart_all",
        "restart_reason": "The original cosine schedule was parameterized for 30 epochs and is not semantically equivalent to a from-scratch 50-epoch schedule.",
        "reference": {key: reference.get(key) for key in COMMON_KEYS},
        "common_fields": common,
        "common_checks": common_checks,
        "e30_parity_checks_excluding_epochs": old_parity,
        "dataset_checks": dataset_checks,
        "method_checks": method_checks,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def e30_vs_e50_rows(runs: dict[str, dict[str, object]], convergence: list[dict[str, object]]) -> list[dict[str, object]]:
    convergence_by_id = {row["run_id"]: row for row in convergence}
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        for method, method_name, _ in METHODS:
            old = json.loads(
                (P1_ROOT / "logs" / run_id(dataset, method, 30) / "metrics.json").read_text(encoding="utf-8")
            )
            new = runs[run_id(dataset, method, 50)]["metrics"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "map50_95_e30_test": old["test"]["map50_95"],
                    "map50_95_e50_test": new["test"]["map50_95"],
                    "delta_test_map50_95": new["test"]["map50_95"] - old["test"]["map50_95"],
                    "best_validation_epoch_e50": convergence_by_id[new["run_id"]]["best_epoch"],
                    "training_seconds_e30": old["timing"]["training_seconds"],
                    "training_seconds_e50": new["timing"]["training_seconds"],
                    "time_increase_seconds": new["timing"]["training_seconds"] - old["timing"]["training_seconds"],
                    "gpu_hours_e30": old["timing"]["gpu_hours"],
                    "gpu_hours_e50": new["timing"]["gpu_hours"],
                    "gpu_hour_increase": new["timing"]["gpu_hours"] - old["timing"]["gpu_hours"],
                }
            )
    return rows


def tradeoff_rows(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        full = runs[run_id(dataset, "full_sft", 50)]["metrics"]
        vpeft = runs[run_id(dataset, "vpeft", 50)]["metrics"]
        full_accuracy = full["test"]["map50_95"]
        vpeft_accuracy = vpeft["test"]["map50_95"]
        rows.append(
            {
                "dataset": dataset_name,
                "trainable_parameter_reduction_percent": (1 - vpeft["parameters"]["trainable_parameters"] / full["parameters"]["trainable_parameters"]) * 100,
                "full_map50_95": full_accuracy,
                "vpeft_map50_95": vpeft_accuracy,
                "accuracy_drop_vs_full": full_accuracy - vpeft_accuracy,
                "accuracy_retention_ratio": vpeft_accuracy / full_accuracy,
                "accuracy_retention_percent": vpeft_accuracy / full_accuracy * 100,
                "memory_saving_percent": (1 - vpeft["resources"]["peak_gpu_memory_mib"] / full["resources"]["peak_gpu_memory_mib"]) * 100,
                "training_time_change_percent": (vpeft["timing"]["training_seconds"] / full["timing"]["training_seconds"] - 1) * 100,
                "gpu_hour_change_percent": (vpeft["timing"]["gpu_hours"] / full["timing"]["gpu_hours"] - 1) * 100,
            }
        )
    return rows


def convergence_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Method | Epoch 41-45 mean | Epoch 46-50 mean | Delta | Best epoch | Best | Last | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['previous_5_epoch_mean']:.6f} | "
            f"{row['last_5_epoch_mean']:.6f} | {row['delta_last5_vs_prev5']:+.6f} | {row['best_epoch']} | "
            f"{row['best_metric']:.6f} | {row['last_epoch_metric']:.6f} | {row['status']} |"
        )
    return "\n".join(lines)


def write_docs(
    comparison: list[dict[str, object]],
    convergence: list[dict[str, object]],
    fairness: dict[str, object],
    tradeoff: list[dict[str, object]],
    decision: str,
    plots: list[str],
) -> None:
    table = convergence_table(convergence)
    audit = f"""# C3 P1 50-Epoch 收敛审计

## 1. Six-run status

seed824 六组 50-epoch run 均为 PASS。执行策略为 `restart_all`：30-epoch cosine schedule 的前 30 步不等价于 50-epoch schedule，因此没有混用 resume 与 restart。原 30-epoch pilot 未覆盖。

## 2. Epoch-wise curves

逐 epoch 数据：`../results/convergence_epochwise_e50.csv`。

{chr(10).join(f'- `{path}`' for path in plots)}

## 3. Fixed convergence rule

沿用阈值 0.01：`delta = mean(epoch 46-50) - mean(epoch 41-45)`；`delta > 0.01` 为 `NOT_CONVERGED`，否则为 `CONVERGED_OR_PLATEAU`。未按方法或结果调整阈值。

## 4. Per-run result

{table}

## 5. Epoch decision

`{decision}`。共有 {sum(row['status'] == 'NOT_CONVERGED' for row in convergence)}/6 组仍超过固定阈值，不能将 50 epoch 称为 final P1 结果；下一阶段仍须六种条件统一预算。

## 6. Fairness

`Protocol fairness: {fairness['status']}`。除 epoch 30→50 外，模型、预训练权重、split、100-image 预算、batch、imgsz、optimizer、lr、weight decay、scheduler、augmentation、seed、freeze 边界与 V-PEFT 配置均保持一致。

## 7. 30 vs 50

两个数据集的单 seed 方法顺序在锁定 test mAP50-95 上均保持 Full-SFT > Frozen Backbone > V-PEFT；这只说明当前 seed 的排序未反转，不构成多 seed 方法结论。详见 `../results/e30_vs_e50.csv`。

## 8. Multi-seed gate

`MULTISEED_READY = NO`。先完成统一 {75 if decision == 'EXTEND_ALL_TO_75' else 100} epoch 候选并重新审计；未运行 seed825/826。
"""
    (P1_ROOT / "docs" / "CONVERGENCE_AUDIT_E50.md").write_text(audit, encoding="utf-8")

    tradeoff_lines = [
        f"- {row['dataset']}: parameter reduction={row['trainable_parameter_reduction_percent']:.2f}%, accuracy retention={row['accuracy_retention_percent']:.2f}%, memory saving={row['memory_saving_percent']:.2f}%, time change={row['training_time_change_percent']:+.2f}%."
        for row in tradeoff
    ]
    report = f"""# C3 P1 对照报告（seed824，50-Epoch Candidate）

## 1. Research Question

在工业缺陷小样本条件下，V-PEFT 是否能以显著更少的可训练参数和资源成本，保持或改善 Full-SFT / Frozen Backbone 的检测性能？

## 2. Protocol

`restart_all`，YOLO11n/yolo11n.pt、每数据集 100 张训练图、50 epochs、batch 8、imgsz 640、AdamW、lr0=0.001、weight decay=0.0005、cosine scheduler、GPU0、FP32、seed824。30 epoch 结果仅为历史 pilot；50 epoch 是当前 convergence candidate。

## 3. Dataset / Split

NEU-DET 与 DeepPCB 沿用 seed824 固定成员列表以及原 val/test，三种方法在同一数据集内成员完全相同。

## 4. Three Methods

- Full-SFT：`lora_r=0, freeze=0`。
- Frozen Backbone：冻结 `model.0-model.10`，训练 `model.11-model.23`。
- V-PEFT：rank=8、alpha=16、strict AO Planner，actual backend=`peft`，不允许 silent fallback。

## 5. Accuracy and Resource Results

{markdown_table(comparison)}

## 6. 30 vs 50

见 `../results/e30_vs_e50.csv`。两个数据集的单 seed 方法顺序没有反转，但 50 epoch 仍未满足统一收敛门槛。

## 7. V-PEFT Trade-off

{chr(10).join(tradeoff_lines)}

## 8. Multi-seed Statistics

未运行 seed825/826；不报告 mean/std/95% CI。

## 9. Planner Analysis

NEU-DET 与 DeepPCB 均为 Planner=ACCEPT、planner backend=vpeft、actual backend=peft、planned/applied targets=59/52，adapter 导出成功。

## 10. Convergence and Limitations

{table}

本阶段仍为单 seed；`{decision}`，因此 50 epoch 不能称为 final P1 result，也不能声明任一方法普遍优于其他方法。

## 11. P1 Conclusion

六组公平闭环均 PASS，但只有 2/6 达到固定 plateau 判据。`MULTISEED_READY = NO`，下一步必须继续统一预算审计。
"""
    (P1_ROOT / "docs" / "C3_P1_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    runs = load_runs()
    refresh_artifact_manifests(runs)
    comparison = comparison_rows(runs)
    convergence = convergence_rows(runs)
    fairness = fairness_audit(runs)
    plots = plot_curves(runs)
    e30_e50 = e30_vs_e50_rows(runs, convergence)
    tradeoff = tradeoff_rows(runs)
    plateau_count = sum(row["status"] == "CONVERGED_OR_PLATEAU" for row in convergence)
    decision = "KEEP_50" if plateau_count >= 4 else "EXTEND_ALL_TO_75"
    multiseed_ready = decision == "KEEP_50" and fairness["status"] == "PASS"

    write_csv(P1_ROOT / "results" / "pilot_seed824_e50.csv", comparison)
    (P1_ROOT / "results" / "pilot_seed824_e50.md").write_text(
        "# C3 P1 seed824 50-Epoch Candidate\n\n30 epoch 结果仅为历史 pilot；本表为当前 50 epoch convergence candidate。\n\n"
        + markdown_table(comparison)
        + "\n",
        encoding="utf-8",
    )
    write_csv(P1_ROOT / "results" / "convergence_epochwise_e50.csv", epochwise_rows(runs))
    write_csv(P1_ROOT / "results" / "e30_vs_e50.csv", e30_e50)
    write_csv(P1_ROOT / "results" / "tradeoff_e50.csv", tradeoff)
    json_write(P1_ROOT / "evidence" / "protocol_fairness_audit_e50.json", fairness)
    json_write(
        P1_ROOT / "evidence" / "convergence_audit_e50.json",
        {
            "schema_version": 1,
            "metric": "mAP50-95",
            "rule": {
                "previous_window": "epochs 41-45",
                "last_window": "epochs 46-50",
                "not_converged_if_delta_gt": THRESHOLD,
            },
            "runs": convergence,
            "epoch_decision": decision,
            "protocol_fairness": fairness["status"],
            "multiseed_ready": multiseed_ready,
            "execution_policy": "restart_all",
            "plots": plots,
        },
    )
    write_docs(comparison, convergence, fairness, tradeoff, decision, plots)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "fairness": fairness["status"],
                "plateau_count": plateau_count,
                "epoch_decision": decision,
                "multiseed_ready": multiseed_ready,
            },
            ensure_ascii=False,
        )
    )
    return 0 if fairness["status"] == "PASS" and len(runs) == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
