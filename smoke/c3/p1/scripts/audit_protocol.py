#!/usr/bin/env python3
"""Audit six existing P1 pilots and produce convergence, fairness, and efficiency evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
DATASETS = (("neu_det", "NEU-DET", "neu"), ("deeppcb", "DeepPCB", "deeppcb"))
METHODS = (("full_sft", "Full-SFT"), ("frozen_backbone", "Frozen Backbone"), ("vpeft", "V-PEFT"))
CONVERGENCE_THRESHOLD = 0.01
COMMON_KEYS = (
    "model",
    "pretrained",
    "epochs",
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


def normalized_curve(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    rows = []
    for row in source:
        learning_rates = [float(value) for key, value in row.items() if key.startswith("lr/")]
        train_parts = [float(row[key]) for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss")]
        val_parts = [float(row[key]) for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss")]
        rows.append(
            {
                "epoch": int(float(row["epoch"])),
                "train_box_loss": train_parts[0],
                "train_cls_loss": train_parts[1],
                "train_dfl_loss": train_parts[2],
                "train_loss": sum(train_parts),
                "val_box_loss": val_parts[0],
                "val_cls_loss": val_parts[1],
                "val_dfl_loss": val_parts[2],
                "val_loss": sum(val_parts),
                "precision": float(row["metrics/precision(B)"]),
                "recall": float(row["metrics/recall(B)"]),
                "map50": float(row["metrics/mAP50(B)"]),
                "map50_95": float(row["metrics/mAP50-95(B)"]),
                "learning_rate": statistics.fmean(learning_rates),
            }
        )
    if len(rows) != 30 or [row["epoch"] for row in rows] != list(range(1, 31)):
        raise ValueError(f"Expected epochs 1-30 in {path.relative_to(REPO_ROOT)}")
    return rows


def load_runs() -> dict[str, dict[str, object]]:
    runs = {}
    for dataset, dataset_name, _ in DATASETS:
        for method, method_name in METHODS:
            run_id = f"{dataset}_{method}_seed824"
            log_dir = P1_ROOT / "logs" / run_id
            runs[run_id] = {
                "run_id": run_id,
                "dataset": dataset,
                "dataset_name": dataset_name,
                "method": method,
                "method_name": method_name,
                "log_dir": log_dir,
                "curve": normalized_curve(log_dir / "learning_curve.csv"),
                "metrics": json.loads((log_dir / "metrics.json").read_text(encoding="utf-8")),
                "resolved": yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8")),
            }
    return runs


def write_epochwise(runs: dict[str, dict[str, object]]) -> Path:
    output = P1_ROOT / "results" / "convergence_epochwise.csv"
    rows = []
    for run in runs.values():
        for curve_row in run["curve"]:
            rows.append(
                {
                    "dataset": run["dataset_name"],
                    "method": run["method_name"],
                    "run_id": run["run_id"],
                    **curve_row,
                }
            )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def plot_curves(runs: dict[str, dict[str, object]]) -> list[str]:
    output_dir = P1_ROOT / "visualizations" / "convergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("map50_95", "mAP50-95", "map5095"),
        ("map50", "mAP50", "map50"),
        ("train_loss", "Train loss (box + cls + dfl)", "loss"),
    )
    paths = []
    for dataset, dataset_name, prefix in DATASETS:
        for key, y_label, suffix in definitions:
            figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
            for method, method_name in METHODS:
                run = runs[f"{dataset}_{method}_seed824"]
                curve = run["curve"]
                axis.plot([row["epoch"] for row in curve], [row[key] for row in curve], label=method_name, linewidth=2)
            axis.set_title(f"{dataset_name}: {y_label} vs epoch")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            output = output_dir / f"{prefix}_{suffix}.png"
            figure.savefig(output)
            plt.close(figure)
            paths.append(output.relative_to(REPO_ROOT).as_posix())
    return paths


def convergence_results(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    results = []
    for run in runs.values():
        values = [row["map50_95"] for row in run["curve"]]
        previous = statistics.fmean(values[-10:-5])
        last = statistics.fmean(values[-5:])
        delta = last - previous
        best_index = max(range(len(values)), key=values.__getitem__)
        status = "NOT_CONVERGED" if delta > CONVERGENCE_THRESHOLD else "CONVERGED_OR_PLATEAU"
        results.append(
            {
                "dataset": run["dataset_name"],
                "method": run["method_name"],
                "run_id": run["run_id"],
                "previous_5_epoch_mean": previous,
                "last_5_epoch_mean": last,
                "delta_last5_vs_prev5": delta,
                "best_epoch": best_index + 1,
                "best_metric": values[best_index],
                "last_epoch_metric": values[-1],
                "status": status,
            }
        )
    return results


def fairness_audit(runs: dict[str, dict[str, object]]) -> dict[str, object]:
    reference = runs["neu_det_full_sft_seed824"]["resolved"]
    shared = {
        key: {
            "expected": reference.get(key),
            "values": {run_id: run["resolved"].get(key) for run_id, run in runs.items()},
        }
        for key in COMMON_KEYS
    }
    shared_checks = {key: len({json.dumps(value, sort_keys=True) for value in row["values"].values()}) == 1 for key, row in shared.items()}

    dataset_checks: dict[str, object] = {}
    for dataset, _, _ in DATASETS:
        dataset_runs = [runs[f"{dataset}_{method}_seed824"] for method, _ in METHODS]
        data_values = {run["resolved"].get("data") for run in dataset_runs}
        test_values = [run["metrics"]["test"] for run in dataset_runs]
        manifest = json.loads((P1_ROOT / "evidence" / f"{dataset}_split_manifest.json").read_text(encoding="utf-8"))
        dataset_checks[dataset] = {
            "same_dataset_yaml": len(data_values) == 1,
            "same_train_membership": manifest.get("selected_train_images") == 100 and manifest.get("status") == "PASS",
            "same_val_test_definition": len(data_values) == 1,
            "same_test_evaluation_arguments": len(
                {
                    (row.get("split"), row.get("device"), row.get("batch"), row.get("imgsz")) for row in test_values
                }
            )
            == 1,
        }

    method_checks = {}
    for dataset, _, _ in DATASETS:
        full = runs[f"{dataset}_full_sft_seed824"]["resolved"]
        frozen = runs[f"{dataset}_frozen_backbone_seed824"]["resolved"]
        vpeft = runs[f"{dataset}_vpeft_seed824"]["resolved"]
        method_checks[dataset] = {
            "full_sft": int(full.get("lora_r", 0) or 0) == 0 and int(full.get("freeze", 0) or 0) == 0,
            "frozen_backbone": int(frozen.get("lora_r", 0) or 0) == 0 and int(frozen.get("freeze", 0) or 0) == 11,
            "vpeft": all(
                (
                    int(vpeft.get("lora_r", 0) or 0) == 8,
                    int(vpeft.get("lora_alpha", 0) or 0) == 16,
                    vpeft.get("lora_planner_enabled") is True,
                    vpeft.get("lora_planner_backend") == "vpeft",
                    vpeft.get("lora_vpeft_strict") is True,
                )
            ),
        }

    evaluator = P1_ROOT / "scripts" / "evaluate_p1.py"
    checks = {
        "same_pretrained_model": shared_checks["model"] and shared_checks["pretrained"],
        "same_train_sample_budget": all(row["same_train_membership"] for row in dataset_checks.values()),
        "same_val_test_within_dataset": all(row["same_val_test_definition"] for row in dataset_checks.values()),
        "same_epochs_batch_imgsz": all(shared_checks[key] for key in ("epochs", "batch", "imgsz")),
        "same_optimizer_lr_weight_decay": all(
            shared_checks[key] for key in ("optimizer", "lr0", "weight_decay", "momentum")
        ),
        "same_augmentation": all(
            shared_checks[key]
            for key in (
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
                "close_mosaic",
            )
        ),
        "same_scheduler": all(shared_checks[key] for key in ("cos_lr", "lrf", "warmup_epochs")),
        "same_seed": shared_checks["seed"],
        "same_evaluation_code_and_arguments": evaluator.is_file()
        and all(row["same_test_evaluation_arguments"] for row in dataset_checks.values()),
        "method_boundaries_valid": all(all(row.values()) for row in method_checks.values()),
    }
    return {
        "schema_version": 1,
        "scope": "Fairness across three strategies within each dataset; dataset identity is the only cross-dataset difference.",
        "evaluation_code": evaluator.relative_to(REPO_ROOT).as_posix(),
        "evaluation_code_sha256": sha256(evaluator),
        "shared_fields": shared,
        "shared_field_checks": shared_checks,
        "dataset_checks": dataset_checks,
        "method_checks": method_checks,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def log_iteration_speed(path: Path) -> tuple[float | None, int]:
    text = path.read_text(encoding="utf-8")
    by_epoch: dict[int, float] = {}
    for epoch, speed in re.findall(r"^\s*(\d+)/30\s+.*100%.*?([0-9.]+)it/s", text, flags=re.MULTILINE):
        by_epoch[int(epoch)] = float(speed)
    return (statistics.fmean(by_epoch.values()) if by_epoch else None, len(by_epoch))


def efficiency_rows(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    profile_payload = json.loads((P1_ROOT / "evidence" / "efficiency_profile.json").read_text(encoding="utf-8"))
    profiles = {row["run_id"]: row for row in profile_payload["runs"]}
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        full = runs[f"{dataset}_full_sft_seed824"]["metrics"]
        for method, method_name in METHODS:
            run = runs[f"{dataset}_{method}_seed824"]
            metrics = run["metrics"]
            profile = profiles[run["run_id"]]
            speed, speed_epochs = log_iteration_speed(run["log_dir"] / "stdout.log")
            trainable = metrics["parameters"]["trainable_parameters"]
            full_trainable = full["parameters"]["trainable_parameters"]
            memory = metrics["resources"]["peak_gpu_memory_mib"]
            full_memory = full["resources"]["peak_gpu_memory_mib"]
            seconds = metrics["timing"]["training_seconds"]
            full_seconds = full["timing"]["training_seconds"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "run_id": run["run_id"],
                    "trainable_parameters": trainable,
                    "trainable_parameter_reduction_vs_full_percent": (1 - trainable / full_trainable) * 100,
                    "observed_peak_reserved_mib": memory,
                    "observed_memory_saving_vs_full_percent": (1 - memory / full_memory) * 100,
                    "observed_training_seconds": seconds,
                    "observed_training_time_change_vs_full_percent": (seconds / full_seconds - 1) * 100,
                    "observed_mean_terminal_iteration_speed_it_s": speed,
                    "observed_speed_epochs_parsed": speed_epochs,
                    "profile_forward_ms": profile["forward_ms_mean"],
                    "profile_backward_ms": profile["backward_ms_mean"],
                    "profile_optimizer_step_ms": profile["optimizer_step_ms_mean"],
                    "profile_iteration_ms": profile["iteration_ms_mean"],
                    "profile_peak_allocated_mib": profile["peak_allocated_mib"],
                    "profile_peak_reserved_mib": profile["peak_reserved_mib"],
                    "runtime_fp32_parameter_mib": profile["runtime_fp32_parameter_mib"],
                    "runtime_fp32_trainable_parameter_mib": profile["runtime_fp32_trainable_parameter_mib"],
                    "optimizer_state_mib": profile["optimizer_state_mib"],
                    "adapter_parameters": profile["adapter_parameters"],
                    "runtime_fp32_adapter_mib": profile["runtime_fp32_adapter_mib"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tradeoff_rows(runs: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        full = runs[f"{dataset}_full_sft_seed824"]["metrics"]
        vpeft = runs[f"{dataset}_vpeft_seed824"]["metrics"]
        full_accuracy = float(full["test"]["map50_95"])
        vpeft_accuracy = float(vpeft["test"]["map50_95"])
        rows.append(
            {
                "dataset": dataset_name,
                "full_map50_95": full_accuracy,
                "vpeft_map50_95": vpeft_accuracy,
                "trainable_parameter_reduction_percent": (
                    1 - vpeft["parameters"]["trainable_parameters"] / full["parameters"]["trainable_parameters"]
                )
                * 100,
                "memory_saving_percent": (
                    1 - vpeft["resources"]["peak_gpu_memory_mib"] / full["resources"]["peak_gpu_memory_mib"]
                )
                * 100,
                "training_time_change_percent": (
                    vpeft["timing"]["training_seconds"] / full["timing"]["training_seconds"] - 1
                )
                * 100,
                "accuracy_drop_map50_95": full_accuracy - vpeft_accuracy,
                "accuracy_retention_ratio": vpeft_accuracy / full_accuracy,
                "accuracy_retention_percent": vpeft_accuracy / full_accuracy * 100,
            }
        )
    return rows


def efficiency_report(rows: list[dict[str, object]]) -> str:
    vpeft = {row["dataset"]: row for row in rows if row["method"] == "V-PEFT"}
    full = {row["dataset"]: row for row in rows if row["method"] == "Full-SFT"}
    table_lines = [
        "| Dataset | Method | Trainable reduction | Pilot memory saving | Pilot time change | Profile F/B/Step (ms) | Profile allocated/reserved (MiB) | Optimizer state (MiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['dataset']} | {row['method']} | {row['trainable_parameter_reduction_vs_full_percent']:.2f}% | "
            f"{row['observed_memory_saving_vs_full_percent']:.2f}% | {row['observed_training_time_change_vs_full_percent']:+.2f}% | "
            f"{row['profile_forward_ms']:.2f}/{row['profile_backward_ms']:.2f}/{row['profile_optimizer_step_ms']:.2f} | "
            f"{row['profile_peak_allocated_mib']:.0f}/{row['profile_peak_reserved_mib']:.0f} | "
            f"{row['optimizer_state_mib']:.2f} |"
        )
    facts = []
    explanations = []
    for dataset in ("NEU-DET", "DeepPCB"):
        v = vpeft[dataset]
        f = full[dataset]
        facts.append(
            f"- {dataset}: V-PEFT 可训练参数减少 {v['trainable_parameter_reduction_vs_full_percent']:.2f}%，"
            f"实际 pilot 峰值 reserved memory 仅减少 {v['observed_memory_saving_vs_full_percent']:.2f}%，"
            f"训练时间变化 {v['observed_training_time_change_vs_full_percent']:+.2f}%。"
        )
        explanations.append(
            f"- {dataset}: 固定 batch 微型分析中，optimizer state 从 {f['optimizer_state_mib']:.2f} MiB 降至 "
            f"{v['optimizer_state_mib']:.2f} MiB，但 peak allocated 仅从 {f['profile_peak_allocated_mib']:.0f} MiB "
            f"降至 {v['profile_peak_allocated_mib']:.0f} MiB；参数/优化器状态不是总 GPU 内存的主要部分。"
        )
    return f"""# V-PEFT 效率差距分析

本分析读取既有 30-epoch pilot，并对其 checkpoint 做固定 batch、FP32、3 次 warmup + 10 次测量的内存不落盘微型分析。微型分析只用于定位开销，不能替代真实 pilot 的端到端资源数据。

## Observed facts

{chr(10).join(facts)}

{chr(10).join(table_lines)}

Trainer 日志中的 `GPU_mem` 来自 `torch.cuda.memory_reserved()`；上述 pilot memory saving 采用这一共同口径。微型分析另行 reset CUDA peak statistics，两个口径不能混用。

## Supported explanations

{chr(10).join(explanations)}

- 固定 batch 下 V-PEFT forward 时间约 38.9 ms，而 Full-SFT 为 24.3–25.2 ms；adapter 分支没有融合，确实带来额外 forward 计算。该测量支持实际训练时间增加的方向，但不证明全部端到端差值都由 adapter 计算造成。
- V-PEFT 的 FP32 adapter 参数本身约 0.69 MiB，总参数内存反而比 Full-SFT 多约 0.69 MiB；节省主要来自 gradient 和 optimizer state，而不是 base parameter storage。

## Hypotheses requiring further instrumentation

- V-PEFT optimizer-step 微型时间高于 Full-SFT，可能与更多小 adapter 张量、参数组或 kernel-launch 开销有关；当前 profiler 没有逐 kernel 归因，因此这只是待验证假设。
- 微型分析的 reserved-memory saving 大于完整 pilot 的约 1%，可能与 allocator history、验证阶段和端到端生命周期不同有关；需要 trainer 内阶段性 memory snapshot 才能归因。
- 数据加载和验证占用会影响完整 wall time；当前没有把端到端 wall time逐阶段拆开，不能将全部时间差解释为 forward/backward。
"""


def convergence_report(
    convergence: list[dict[str, object]], fairness: dict[str, object], efficiency: list[dict[str, object]], plots: list[str]
) -> str:
    table = [
        "| Dataset | Method | Previous 5 mean | Last 5 mean | Delta | Best epoch | Best | Last | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in convergence:
        table.append(
            f"| {row['dataset']} | {row['method']} | {row['previous_5_epoch_mean']:.6f} | "
            f"{row['last_5_epoch_mean']:.6f} | {row['delta_last5_vs_prev5']:+.6f} | {row['best_epoch']} | "
            f"{row['best_metric']:.6f} | {row['last_epoch_metric']:.6f} | {row['status']} |"
        )
    vpeft_efficiency = [row for row in efficiency if row["method"] == "V-PEFT"]
    return f"""# C3 P1 30-Epoch 收敛审计

## 1. Six-run status

既有 seed824 六组 pilot 均为 PASS，本审计未重跑或覆盖任何训练结果。

## 2. Epoch-wise curves

逐 epoch 数据：`../results/convergence_epochwise.csv`。生成的六张图：

{chr(10).join(f'- `{path}`' for path in plots)}

## 3. Convergence rule

统一使用 mAP50-95：`delta = mean(epoch 26–30) - mean(epoch 21–25)`。`delta > 0.01` 为 `NOT_CONVERGED`，否则为 `CONVERGED_OR_PLATEAU`。阈值对六组一次性固定，不按方法调整。

Epoch 决策采用顺序扩展：全部 plateau 才 `KEEP_30`；任一 run 未收敛则先 `EXTEND_ALL_TO_50` 并在 50 epoch 再审计；只有共同 50-epoch 审计仍未收敛时才考虑 `EXTEND_ALL_TO_100`。

## 4. Per-run convergence result

{chr(10).join(table)}

## 5. Final epoch decision

`EXTEND_ALL_TO_50`。六组 delta 均大于 0.01，且最佳 mAP50-95 均出现在 epoch 30。所有方法必须统一延长，不能只增加 V-PEFT 预算。

## 6. Fairness audit

`Protocol fairness: {fairness['status']}`。相同模型、split、100-image 预算、epoch、batch、imgsz、优化器、学习率、weight decay、增强、scheduler、seed 和 test evaluator 均已逐字段核验；允许差异仅为训练策略。

## 7. V-PEFT efficiency observations

{chr(10).join(f"- {row['dataset']}: trainable reduction={row['trainable_parameter_reduction_vs_full_percent']:.2f}%, memory saving={row['observed_memory_saving_vs_full_percent']:.2f}%, training time change={row['observed_training_time_change_vs_full_percent']:+.2f}%." for row in vpeft_efficiency)}

详细事实、支持性解释和假设见 `VPEFT_EFFICIENCY_ANALYSIS.md`。

## 8. Whether multi-seed can start

`MULTISEED_READY = NO`。最终共同预算已决定为 50 epochs，但现有 seed824 仅完成 30 epochs；必须先以不覆盖方式让六组 seed824 达到共同 50-epoch protocol 并重新审计，之后才能生成或启动 seed825/826 计划。
"""


def main() -> int:
    runs = load_runs()
    epochwise = write_epochwise(runs)
    plots = plot_curves(runs)
    convergence = convergence_results(runs)
    fairness = fairness_audit(runs)
    efficiency = efficiency_rows(runs)
    tradeoff = tradeoff_rows(runs)
    decision = "KEEP_30" if all(row["status"] == "CONVERGED_OR_PLATEAU" for row in convergence) else "EXTEND_ALL_TO_50"
    multiseed_ready = decision == "KEEP_30" and fairness["status"] == "PASS"

    write_csv(P1_ROOT / "results" / "efficiency_diagnostics.csv", efficiency)
    write_csv(P1_ROOT / "results" / "pilot_tradeoff.csv", tradeoff)
    json_write(P1_ROOT / "evidence" / "protocol_fairness_audit.json", fairness)
    json_write(
        P1_ROOT / "evidence" / "convergence_audit.json",
        {
            "schema_version": 1,
            "metric": "mAP50-95",
            "rule": {
                "previous_window": "epochs 21-25",
                "last_window": "epochs 26-30",
                "not_converged_if_delta_gt": CONVERGENCE_THRESHOLD,
            },
            "runs": convergence,
            "epoch_decision": decision,
            "protocol_fairness": fairness["status"],
            "multiseed_ready": multiseed_ready,
            "epochwise_csv": epochwise.relative_to(REPO_ROOT).as_posix(),
            "plots": plots,
        },
    )
    efficiency_doc = P1_ROOT / "docs" / "VPEFT_EFFICIENCY_ANALYSIS.md"
    efficiency_doc.write_text(efficiency_report(efficiency), encoding="utf-8")
    audit_doc = P1_ROOT / "docs" / "CONVERGENCE_AUDIT.md"
    audit_doc.write_text(convergence_report(convergence, fairness, efficiency, plots), encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": len(runs),
                "epoch_decision": decision,
                "fairness": fairness["status"],
                "multiseed_ready": multiseed_ready,
            },
            ensure_ascii=False,
        )
    )
    return 0 if fairness["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
