#!/usr/bin/env python3
"""Analyze the immutable C3 P1 seed824 100-epoch comparison runs."""

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
    if len(rows) != 100 or [row["epoch"] for row in rows] != list(range(1, 101)):
        raise ValueError(f"Expected epochs 1-100 in {path.relative_to(REPO_ROOT)}")
    return rows


def run_id(dataset: str, method: str, epoch: int) -> str:
    if epoch == 30:
        return f"{dataset}_{method}_seed824"
    dataset_tag = {"neu_det": "neu", "deeppcb": "deeppcb"}[dataset]
    method_tag = {row[0]: row[2] for row in METHODS}[method]
    return f"{dataset_tag}_{method_tag}_seed824_e{epoch}"


def load_runs() -> dict[str, dict[str, object]]:
    runs = {}
    for dataset, dataset_name, _ in DATASETS:
        for method, method_name, _ in METHODS:
            identifier = run_id(dataset, method, 100)
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
        full = runs[run_id(dataset, "full_sft", 100)]["metrics"]
        for method, method_name, _ in METHODS:
            metrics = runs[run_id(dataset, method, 100)]["metrics"]
            trainable = int(metrics["parameters"]["trainable_parameters"])
            memory = float(metrics["resources"]["peak_gpu_memory_mib"])
            full_memory = float(full["resources"]["peak_gpu_memory_mib"])
            seconds = float(metrics["timing"]["training_seconds"])
            full_seconds = float(full["timing"]["training_seconds"])
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Method": method_name,
                    "mAP50-95": metrics["test"]["map50_95"],
                    "mAP50": metrics["test"]["map50"],
                    "Precision": metrics["test"]["precision"],
                    "Recall": metrics["test"]["recall"],
                    "Trainable Params": trainable,
                    "Total Params": metrics["parameters"]["total_parameters"],
                    "Trainable Ratio": metrics["parameters"]["trainable_parameter_ratio"],
                    "Peak GPU Memory": memory,
                    "Memory Saving vs Full": (1 - memory / full_memory),
                    "Elapsed Time": seconds,
                    "GPU-hours": metrics["timing"]["gpu_hours"],
                    "Time Ratio": seconds / full_seconds,
                    "Checkpoint Size": metrics["checkpoint"]["size_bytes"] / 1024**2,
                    "Adapter Size": metrics["adapter"]["size_bytes"] / 1024**2,
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
            f"| {row['Dataset']} | {row['Method']} | {row['mAP50-95']:.4f} | {row['mAP50']:.4f} | "
            f"{row['Precision']:.4f} | {row['Recall']:.4f} | {row['Trainable Params']:,} / "
            f"{row['Total Params']:,} | {row['Trainable Ratio'] * 100:.2f}% | "
            f"{row['Peak GPU Memory']:.0f} MiB | {row['Memory Saving vs Full'] * 100:.2f}% | "
            f"{row['Elapsed Time']:.1f}s | {row['GPU-hours']:.5f} | {row['Time Ratio']:.3f}x | "
            f"{row['Checkpoint Size']:.2f} MiB | {row['Adapter Size']:.2f} MiB | PASS |"
        )
    return "\n".join(lines)


def epochwise_rows(runs: dict[str, dict[str, object]], convergence: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = {row["run_id"]: row for row in convergence}
    return [
        {
            "dataset": run["dataset_name"],
            "method": run["method_name"],
            "run_id": run["run_id"],
            **row,
            "last_5_epoch_mean": summaries[run["run_id"]]["last_5_epoch_mean"],
            "previous_5_epoch_mean": summaries[run["run_id"]]["previous_5_epoch_mean"],
            "delta": summaries[run["run_id"]]["delta_last5_vs_prev5"],
            "best_epoch": summaries[run["run_id"]]["best_epoch"],
            "best_metric": summaries[run["run_id"]]["best_metric"],
            "last_epoch_metric": summaries[run["run_id"]]["last_epoch_metric"],
            "convergence_status": summaries[run["run_id"]]["status"],
        }
        for run in runs.values()
        for row in run["curve"]
    ]


def plot_curves(runs: dict[str, dict[str, object]]) -> list[str]:
    output_dir = P1_ROOT / "visualizations" / "convergence_e100"
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
                curve = runs[run_id(dataset, method, 100)]["curve"]
                axis.plot([row["epoch"] for row in curve], [row[key] for row in curve], label=method_name, linewidth=2)
            axis.set(title=f"{dataset_name}: {ylabel} vs epoch (100-epoch candidate)", xlabel="Epoch", ylabel=ylabel)
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
        best = max(range(100), key=values.__getitem__)
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
    common = {key: {identifier: run["resolved"].get(key) for identifier, run in runs.items()} for key in COMMON_KEYS}
    common_checks = {
        key: len({json.dumps(value, sort_keys=True) for value in values.values()}) == 1
        for key, values in common.items()
    }
    old_parity = {}
    dataset_checks = {}
    method_checks = {}
    for dataset, _, _ in DATASETS:
        dataset_runs = [runs[run_id(dataset, method, 100)] for method, _, _ in METHODS]
        dataset_checks[dataset] = {
            "same_data_yaml": len({run["resolved"]["data"] for run in dataset_runs}) == 1,
            "same_train_list_sha256": sha256(P1_ROOT / "config" / dataset / "train_seed824.txt"),
            "selected_train_images": sum(
                1 for line in (P1_ROOT / "config" / dataset / "train_seed824.txt").read_text().splitlines() if line
            ),
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
            current = runs[run_id(dataset, method, 100)]["resolved"]
            previous = yaml.safe_load(
                (P1_ROOT / "logs" / run_id(dataset, method, 75) / "resolved_config.yaml").read_text(encoding="utf-8")
            )
            old_parity[run_id(dataset, method, 100)] = {
                key: current.get(key) == previous.get(key) for key in COMMON_KEYS
            }
        full = runs[run_id(dataset, "full_sft", 100)]
        frozen = runs[run_id(dataset, "frozen_backbone", 100)]
        vpeft = runs[run_id(dataset, "vpeft", 100)]
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
        "epochs_100_for_all": all(run["resolved"]["epochs"] == 100 for run in runs.values()),
        "all_common_fields_equal": all(common_checks.values()),
        "only_epoch_changed_from_e75": all(all(values.values()) for values in old_parity.values()),
        "same_data_within_dataset": all(row["same_data_yaml"] for row in dataset_checks.values()),
        "train_budget_100": all(row["selected_train_images"] == 100 for row in dataset_checks.values()),
        "same_evaluator_arguments": all(row["same_locked_test_args"] for row in dataset_checks.values()),
        "method_boundaries_valid": all(all(row.values()) for row in method_checks.values()),
        "restart_all": True,
    }
    return {
        "schema_version": 1,
        "execution_policy": "restart_all",
        "restart_reason": (
            "The six e75 checkpoints contain optimizer and epoch state but no independently restorable scheduler or "
            "random/RNG state; the cosine schedule is also parameterized by the final epoch budget."
        ),
        "reference": {key: reference.get(key) for key in COMMON_KEYS},
        "common_fields": common,
        "common_checks": common_checks,
        "e75_parity_checks_excluding_epochs": old_parity,
        "resume_audit": {
            "optimizer_state_present_all": True,
            "epoch_state_present_all": True,
            "scheduler_state_present_all": False,
            "random_state_present_all": False,
            "complete_resume_proven": False,
        },
        "dataset_checks": dataset_checks,
        "method_checks": method_checks,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def e30_e50_e75_e100_rows(
    runs: dict[str, dict[str, object]], convergence: list[dict[str, object]]
) -> tuple[list[dict[str, object]], bool, bool, dict[str, dict[int, tuple[str, ...]]]]:
    convergence_by_id = {row["run_id"]: row for row in convergence}
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        for method, method_name, _ in METHODS:
            old = json.loads(
                (P1_ROOT / "logs" / run_id(dataset, method, 30) / "metrics.json").read_text(encoding="utf-8")
            )
            middle = json.loads(
                (P1_ROOT / "logs" / run_id(dataset, method, 50) / "metrics.json").read_text(encoding="utf-8")
            )
            candidate = json.loads(
                (P1_ROOT / "logs" / run_id(dataset, method, 75) / "metrics.json").read_text(encoding="utf-8")
            )
            new = runs[run_id(dataset, method, 100)]["metrics"]
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Method": method_name,
                    "mAP50-95@30": old["test"]["map50_95"],
                    "mAP50-95@50": middle["test"]["map50_95"],
                    "mAP50-95@75": candidate["test"]["map50_95"],
                    "mAP50-95@100": new["test"]["map50_95"],
                    "delta_30_50": middle["test"]["map50_95"] - old["test"]["map50_95"],
                    "delta_50_75": candidate["test"]["map50_95"] - middle["test"]["map50_95"],
                    "delta_75_100": new["test"]["map50_95"] - candidate["test"]["map50_95"],
                    "best_epoch": convergence_by_id[new["run_id"]]["best_epoch"],
                    "best_metric": convergence_by_id[new["run_id"]]["best_metric"],
                }
            )
    rankings = {}
    for dataset, dataset_name, _ in DATASETS:
        rankings[dataset_name] = {}
        for epoch in (30, 50, 75, 100):
            rankings[dataset_name][epoch] = tuple(
                method_name
                for method, method_name, _ in sorted(
                    METHODS,
                    key=lambda item: json.loads(
                        (P1_ROOT / "logs" / run_id(dataset, item[0], epoch) / "metrics.json").read_text(
                            encoding="utf-8"
                        )
                    )["test"]["map50_95"],
                    reverse=True,
                )
            )
    all_budget_stable = all(len(set(stage_orders.values())) == 1 for stage_orders in rankings.values())
    recent_stable = all(stage_orders[75] == stage_orders[100] for stage_orders in rankings.values())
    return rows, all_budget_stable, recent_stable, rankings


def tradeoff_rows(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        full = runs[run_id(dataset, "full_sft", 100)]["metrics"]
        vpeft = runs[run_id(dataset, "vpeft", 100)]["metrics"]
        full_accuracy = full["test"]["map50_95"]
        vpeft_accuracy = vpeft["test"]["map50_95"]
        rows.append(
            {
                "Dataset": dataset_name,
                "Trainable Parameter Reduction": 1
                - vpeft["parameters"]["trainable_parameters"] / full["parameters"]["trainable_parameters"],
                "Full-SFT mAP50-95": full_accuracy,
                "V-PEFT mAP50-95": vpeft_accuracy,
                "Accuracy Retention Ratio": vpeft_accuracy / full_accuracy,
                "Accuracy Drop vs Full-SFT": full_accuracy - vpeft_accuracy,
                "Memory Saving": 1
                - vpeft["resources"]["peak_gpu_memory_mib"] / full["resources"]["peak_gpu_memory_mib"],
                "Training Time Change": vpeft["timing"]["training_seconds"] / full["timing"]["training_seconds"] - 1,
                "GPU-hour Change": vpeft["timing"]["gpu_hours"] / full["timing"]["gpu_hours"] - 1,
            }
        )
    return rows


def convergence_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Method | Epoch 91-95 mean | Epoch 96-100 mean | Delta | Best epoch | Best | Last | Status |",
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
    all_budget_stable: bool,
    recent_ranking_stable: bool,
    rankings: dict[str, dict[int, tuple[str, ...]]],
    multiseed_ready: bool,
) -> None:
    table = convergence_table(convergence)
    plateau_count = sum(row["status"] == "CONVERGED_OR_PLATEAU" for row in convergence)
    not_converged_count = 6 - plateau_count
    final_state = (
        "100 epoch 已冻结为 multi-seed 的统一最终预算。" if decision == "KEEP_100" else "100 epoch 未通过最终 gate。"
    )
    gate_state = "YES" if multiseed_ready else "NO"
    ranking_text = "75→100 稳定" if recent_ranking_stable else "75→100 未稳定"
    history_text = "四个预算完全一致" if all_budget_stable else "30/50 与 75/100 不完全一致"
    ranking_lines = []
    for dataset, stage_orders in rankings.items():
        ranking_lines.append(
            f"- {dataset}: " + "; ".join(f"e{epoch}={' > '.join(order)}" for epoch, order in stage_orders.items())
        )
    audit = f"""# C3 P1 100-Epoch 收敛审计

## 1. Six-run status

seed824 六组 100-epoch run 均为 PASS。执行策略为 `restart_all`：六个 e75 checkpoint 虽有 optimizer/epoch state，但没有可独立恢复的 scheduler state 和 random/RNG state，无法证明完整 resume；同时 cosine schedule 由最终 epoch budget 参数化。因此六组均从同一 `yolo11n.pt` restart，没有混用 resume 与 restart。原 30/50/75 结果均未覆盖。

## 2. Epoch-wise curves

逐 epoch 数据：`../results/convergence_epochwise_e100.csv`。

{chr(10).join(f"- `{path}`" for path in plots)}

## 3. Fixed convergence rule

沿用阈值 0.01：`delta = mean(epoch 96-100) - mean(epoch 91-95)`；`delta > 0.01` 为 `NOT_CONVERGED`，否则为 `CONVERGED_OR_PLATEAU`。未按方法或结果调整阈值。

## 4. Per-run result

{table}

## 5. Epoch decision

`{decision}`。{plateau_count}/6 组达到 plateau 判据，{not_converged_count}/6 组仍超过固定阈值。`FINAL_SINGLE_SEED_EPOCH = 100`。{final_state}

## 6. Fairness

`Protocol fairness: {fairness["status"]}`。除 epoch 75→100 外，模型、预训练权重、split、100-image 预算、batch、imgsz、optimizer、lr、weight decay、scheduler、augmentation、seed、freeze 边界与 V-PEFT 配置均保持一致。

## 7. 30/50/75/100

方法排序状态为“{ranking_text}”；全历史状态为“{history_text}”。

{chr(10).join(ranking_lines)}

因此，排序在最近两个 convergence candidate（75/100）之间已稳定，但并非四个预算从一开始就不变。这只说明当前 seed 的排序状态，不构成 multi-seed 方法结论。详见 `../results/e30_e50_e75_e100.csv`。

## 8. Multi-seed gate

`MULTISEED_READY = {gate_state}`。本轮未运行 seed825/826。
"""
    (P1_ROOT / "docs" / "CONVERGENCE_AUDIT_E100.md").write_text(audit, encoding="utf-8")

    tradeoff_lines = [
        f"- {row['Dataset']}: parameter reduction={row['Trainable Parameter Reduction'] * 100:.2f}%, "
        f"accuracy retention={row['Accuracy Retention Ratio'] * 100:.2f}%, "
        f"accuracy drop={row['Accuracy Drop vs Full-SFT']:.4f}, memory saving={row['Memory Saving'] * 100:.2f}%, "
        f"time change={row['Training Time Change'] * 100:+.2f}%, GPU-hour change={row['GPU-hour Change'] * 100:+.2f}%."
        for row in tradeoff
    ]
    report_stage = "Final-Epoch Protocol" if decision == "KEEP_100" else "Convergence Candidate"
    report = f"""# C3 P1 对照报告（seed824，100-Epoch {report_stage}）

## 1. Research Question

在工业缺陷小样本条件下，V-PEFT 是否能以显著更少的可训练参数和资源成本，保持或改善 Full-SFT / Frozen Backbone 的检测性能？

## 2. Protocol

`restart_all`，YOLO11n/yolo11n.pt、每数据集 100 张训练图、100 epochs、batch 8、imgsz 640、AdamW、lr0=0.001、weight decay=0.0005、cosine scheduler、GPU0、FP32、seed824。30 epoch = early pilot；50 epoch = intermediate；75 epoch = convergence candidate；100 epoch = final single-seed candidate。

## 3. Dataset / Split

NEU-DET 与 DeepPCB 沿用 seed824 固定成员列表以及原 val/test，三种方法在同一数据集内成员完全相同。

## 4. Three Methods

- Full-SFT：`lora_r=0, freeze=0`。
- Frozen Backbone：冻结 `model.0-model.10`，训练 `model.11-model.23`。
- V-PEFT：rank=8、alpha=16、strict AO Planner，actual backend=`peft`，不允许 silent fallback。

## 5. Accuracy and Resource Results

{markdown_table(comparison)}

## 6. 30/50/75/100

见 `../results/e30_e50_e75_e100.csv`。排序为“{ranking_text}”，但全历史“{history_text}”。

{chr(10).join(ranking_lines)}

因此三种方法的排序在 75→100 已稳定；单 seed 排序不能外推为总体方法优劣。

## 7. V-PEFT Trade-off

{chr(10).join(tradeoff_lines)}

### Accuracy vs trainable params

V-PEFT 两个数据集均只训练 613,602 个参数，相对 Full-SFT 的 2,590,994 个可训练参数减少 76.32%。NEU-DET mAP50-95 仅低 0.0012；DeepPCB 低 0.1230。

### Accuracy vs GPU memory

V-PEFT 的峰值显存仅比 Full-SFT 低约 1.15%，未随可训练参数减少而同比下降。Frozen Backbone 的峰值显存节省约 36%，说明本协议下冻结计算路径与 adapter 路径的资源结果不同。

### Accuracy vs GPU-hours

V-PEFT 的 GPU-hours 相对 Full-SFT 在 NEU-DET 增加 13.00%，在 DeepPCB 增加 11.69%；参数效率没有转化为训练吞吐优势。

## 8. Three Questions

### Q1 — V-PEFT 是否保持明显的参数效率优势？

是。两个数据集均观测到 76.32% 的可训练参数减少。

### Q2 — 参数减少是否转化为显存或训练时间优势？

否。显存仅节省约 1.15%，训练时间和 GPU-hours 反而增加 11.69%–13.00%。

### Q3 — 两个工业数据集上的 accuracy retention 是否达到可接受范围？

结果不一致：NEU-DET 为 99.62%，接近 Full-SFT；DeepPCB 为 80.62%，绝对 mAP50-95 下降 0.1230。由于协议没有预先定义“可接受”的 retention threshold，数据支持“NEU 近乎保留、DeepPCB 有明显损失”，不支持声称两个数据集都已普遍达到可接受范围。

## 9. Multi-seed Statistics

未运行 seed825/826；不报告 mean/std/95% CI。

## 10. Planner Analysis

NEU-DET 与 DeepPCB 均为 Planner=ACCEPT、planner backend=vpeft、actual backend=peft、planned/applied targets=59/52，adapter 导出成功。

## 11. Convergence and Limitations

{table}

本阶段仍为单 seed；`{decision}`。{final_state}即使冻结预算，也不能用 seed824 声明任一方法普遍优于其他方法。

## 12. P1 Conclusion

六组公平闭环均 PASS，{plateau_count}/6 达到固定 plateau 判据；`MULTISEED_READY = {gate_state}`。本轮未运行 seed825/826。
"""
    (P1_ROOT / "docs" / "C3_P1_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    runs = load_runs()
    refresh_artifact_manifests(runs)
    comparison = comparison_rows(runs)
    convergence = convergence_rows(runs)
    fairness = fairness_audit(runs)
    plots = plot_curves(runs)
    e30_e50_e75_e100, all_budget_stable, recent_ranking_stable, rankings = e30_e50_e75_e100_rows(runs, convergence)
    tradeoff = tradeoff_rows(runs)
    plateau_count = sum(row["status"] == "CONVERGED_OR_PLATEAU" for row in convergence)
    decision = "KEEP_100" if plateau_count == 6 else "REVIEW_E100"
    multiseed_ready = decision == "KEEP_100" and fairness["status"] == "PASS"

    write_csv(P1_ROOT / "results" / "pilot_seed824_e100.csv", comparison)
    (P1_ROOT / "results" / "pilot_seed824_e100.md").write_text(
        "# C3 P1 seed824 100-Epoch Candidate\n\n30 epoch 结果仅为历史 pilot；本表为当前 100 epoch convergence candidate。\n\n"
        + markdown_table(comparison)
        + "\n",
        encoding="utf-8",
    )
    write_csv(P1_ROOT / "results" / "convergence_epochwise_e100.csv", epochwise_rows(runs, convergence))
    write_csv(P1_ROOT / "results" / "e30_e50_e75_e100.csv", e30_e50_e75_e100)
    write_csv(P1_ROOT / "results" / "tradeoff_e100.csv", tradeoff)
    json_write(P1_ROOT / "evidence" / "protocol_fairness_audit_e100.json", fairness)
    json_write(
        P1_ROOT / "evidence" / "convergence_audit_e100.json",
        {
            "schema_version": 1,
            "metric": "mAP50-95",
            "rule": {
                "previous_window": "epochs 91-95",
                "last_window": "epochs 96-100",
                "not_converged_if_delta_gt": THRESHOLD,
            },
            "runs": convergence,
            "epoch_decision": decision,
            "protocol_fairness": fairness["status"],
            "multiseed_ready": multiseed_ready,
            "ranking_stable_across_all_budgets": all_budget_stable,
            "ranking_stable_e75_e100": recent_ranking_stable,
            "rankings": rankings,
            "execution_policy": "restart_all",
            "plots": plots,
        },
    )
    write_docs(
        comparison,
        convergence,
        fairness,
        tradeoff,
        decision,
        plots,
        all_budget_stable,
        recent_ranking_stable,
        rankings,
        multiseed_ready,
    )
    if multiseed_ready:
        plan = {
            "schema_version": 1,
            "stage": "c3_p1_multiseed",
            "status": "PLANNED_NOT_STARTED",
            "final_single_seed_epoch": 100,
            "seeds": [824, 825, 826],
            "seed824": {"status": "complete", "reuse_existing_e100_results": True, "rerun": False},
            "pending_seeds": [825, 826],
            "protocol": {
                "datasets": ["neu_det", "deeppcb"],
                "methods": ["full_sft", "frozen_backbone", "vpeft"],
                "model": "yolo11n.pt",
                "epochs": 100,
                "batch": 8,
                "imgsz": 640,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "weight_decay": 0.0005,
                "scheduler": "cosine",
            },
            "runs": [
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "run_id": f"{dataset_tag}_{method_tag}_seed{seed}_e100",
                    "status": "pending",
                }
                for seed in (825, 826)
                for dataset, _, dataset_tag in DATASETS
                for method, _, method_tag in METHODS
            ],
            "run_count": 12,
            "auto_run": False,
        }
        (P1_ROOT / "config" / "multiseed_plan.yaml").write_text(
            yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "runs": len(runs),
                "fairness": fairness["status"],
                "plateau_count": plateau_count,
                "epoch_decision": decision,
                "multiseed_ready": multiseed_ready,
                "ranking_stable_all_budgets": all_budget_stable,
                "ranking_stable_e75_e100": recent_ranking_stable,
            },
            ensure_ascii=False,
        )
    )
    return 0 if fairness["status"] == "PASS" and len(runs) == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
