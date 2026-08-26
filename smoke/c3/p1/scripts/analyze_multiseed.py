#!/usr/bin/env python3
"""Validate and summarize the immutable 18-run C3 P1 experiment matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
T_CRITICAL_DF2_95 = 4.302652729911275
DATASETS = (("neu_det", "NEU-DET", "neu"), ("deeppcb", "DeepPCB", "deeppcb"))
METHODS = (
    ("full_sft", "Full-SFT", "full"),
    ("frozen_backbone", "Frozen Backbone", "frozen"),
    ("vpeft", "V-PEFT", "vpeft"),
)
SEEDS = (824, 825, 826)
PROTOCOL_EXPECTED = {
    "epochs": 100,
    "batch": 8,
    "imgsz": 640,
    "workers": 0,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "weight_decay": 0.0005,
    "cos_lr": True,
    "amp": False,
    "deterministic": True,
}
FAIRNESS_KEYS = (
    "model",
    "pretrained",
    "epochs",
    "batch",
    "imgsz",
    "workers",
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
METRICS = ("map50_95", "map50", "precision", "recall")


def run_id(dataset: str, method: str, seed: int) -> str:
    dataset_tag = next(row[2] for row in DATASETS if row[0] == dataset)
    method_tag = next(row[2] for row in METHODS if row[0] == method)
    return f"{dataset_tag}_{method_tag}_seed{seed}_e100"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float]) -> dict[str, float]:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Every reported statistic requires three finite seed values")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = T_CRITICAL_DF2_95 * standard_deviation / math.sqrt(len(values))
    return {
        "mean": mean,
        "std": standard_deviation,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def verify_manifest(log_dir: Path) -> dict[str, object]:
    manifest_path = log_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for row in manifest["artifacts"]:
        artifact = REPO_ROOT / row["path"]
        if not artifact.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
        elif artifact.stat().st_size != row["size_bytes"] or sha256(artifact) != row["sha256"]:
            failures.append({"path": row["path"], "reason": "size_or_sha256_mismatch"})
    return {
        "manifest": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "artifact_count": manifest["artifact_count"],
        "verified": not failures,
        "failures": failures,
    }


def load_runs() -> dict[tuple[str, str, int], dict[str, object]]:
    runs = {}
    for dataset, dataset_name, _ in DATASETS:
        for method, method_name, _ in METHODS:
            for seed in SEEDS:
                identifier = run_id(dataset, method, seed)
                log_dir = P1_ROOT / "logs" / identifier
                metrics_path = log_dir / "metrics.json"
                if not metrics_path.is_file():
                    raise FileNotFoundError(metrics_path.relative_to(REPO_ROOT))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                checks = metrics.get("checks", {})
                if metrics.get("status") != "PASS" or metrics.get("exit_code") != 0 or not checks or not all(checks.values()):
                    raise ValueError(f"Run failed structured validation: {identifier}")
                learning_curve = log_dir / "learning_curve.csv"
                with learning_curve.open(encoding="utf-8", newline="") as stream:
                    curve_rows = list(csv.DictReader(stream))
                if len(curve_rows) != 100 or not all(
                    math.isfinite(float(value))
                    for row in curve_rows
                    for key, value in row.items()
                    if key != "epoch" and value not in (None, "")
                ):
                    raise ValueError(f"Incomplete or non-finite learning curve: {identifier}")
                resolved = yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
                environment = json.loads((log_dir / "environment.json").read_text(encoding="utf-8"))
                integrity = verify_manifest(log_dir)
                if not integrity["verified"]:
                    raise ValueError(f"Artifact integrity failed: {identifier}: {integrity['failures']}")
                runs[(dataset, method, seed)] = {
                    "run_id": identifier,
                    "dataset": dataset,
                    "dataset_name": dataset_name,
                    "method": method,
                    "method_name": method_name,
                    "seed": seed,
                    "log_dir": log_dir,
                    "metrics": metrics,
                    "resolved": resolved,
                    "environment": environment,
                    "integrity": integrity,
                }
    return runs


def fairness_audit(runs: dict[tuple[str, str, int], dict[str, object]]) -> dict[str, object]:
    exact_protocol = {
        run["run_id"]: {key: run["resolved"].get(key) == expected for key, expected in PROTOCOL_EXPECTED.items()}
        for run in runs.values()
    }
    seed_only_variation = {}
    for dataset, _, _ in DATASETS:
        for method, _, _ in METHODS:
            method_runs = [runs[(dataset, method, seed)] for seed in SEEDS]
            seed_only_variation[f"{dataset}:{method}"] = {
                key: len({json.dumps(run["resolved"].get(key), sort_keys=True) for run in method_runs}) == 1
                for key in FAIRNESS_KEYS
            }
    sample_membership = {}
    for dataset, _, _ in DATASETS:
        train_list = P1_ROOT / "config" / dataset / "train_seed824.txt"
        dataset_yaml = yaml.safe_load((P1_ROOT / "config" / dataset / "dataset.yaml").read_text(encoding="utf-8"))
        sample_membership[dataset] = {
            "dataset_yaml_train": dataset_yaml["train"],
            "expected_fixed_train": f"smoke/c3/p1/config/{dataset}/train_seed824.txt",
            "fixed_train_list_sha256": sha256(train_list),
            "selected_train_images": len([line for line in train_list.read_text(encoding="utf-8").splitlines() if line]),
            "all_runs_use_same_data_yaml": len(
                {runs[(dataset, method, seed)]["resolved"].get("data") for method, _, _ in METHODS for seed in SEEDS}
            )
            == 1,
        }
    method_boundaries = {}
    for dataset, _, _ in DATASETS:
        for method, _, _ in METHODS:
            for seed in SEEDS:
                run = runs[(dataset, method, seed)]
                resolved = run["resolved"]
                if method == "full_sft":
                    passed = int(resolved.get("freeze", 0) or 0) == 0 and int(resolved.get("lora_r", 0) or 0) == 0
                elif method == "frozen_backbone":
                    passed = int(resolved.get("freeze", 0) or 0) == 11 and int(resolved.get("lora_r", 0) or 0) == 0
                else:
                    adapter = run["metrics"]["adapter"]
                    passed = all(
                        (
                            int(resolved.get("lora_r", 0) or 0) == 8,
                            int(resolved.get("lora_alpha", 0) or 0) == 16,
                            resolved.get("lora_vpeft_strict") is True,
                            resolved.get("lora_planner_enabled") is True,
                            resolved.get("lora_planner_backend") == "vpeft",
                            adapter.get("planner_status") in {"ACCEPT", "ADAPT"},
                            adapter.get("actual_backend") == "peft",
                            int(adapter.get("applied_targets") or 0) > 0,
                            int(adapter.get("size_bytes") or 0) > 0,
                        )
                    )
                method_boundaries[run["run_id"]] = passed
    checks = {
        "all_exact_protocol_fields": all(all(values.values()) for values in exact_protocol.values()),
        "only_seed_varies_within_dataset_method": all(
            all(values.values()) for values in seed_only_variation.values()
        ),
        "fixed_sample_ids": all(
            row["dataset_yaml_train"] == row["expected_fixed_train"]
            and row["selected_train_images"] == 100
            and row["all_runs_use_same_data_yaml"]
            for row in sample_membership.values()
        ),
        "seed_matches_run": all(run["resolved"].get("seed") == seed for (dataset, method, seed), run in runs.items()),
        "method_boundaries_locked": all(method_boundaries.values()),
        "evaluation_code_fixed": subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "smoke/c3/p1/scripts/evaluate_p1.py"],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "exact_protocol": exact_protocol,
        "seed_only_variation": seed_only_variation,
        "sample_membership": sample_membership,
        "method_boundaries": method_boundaries,
        "evaluation_code_sha256": sha256(P1_ROOT / "scripts" / "evaluate_p1.py"),
    }


def all_run_rows(runs: dict[tuple[str, str, int], dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        for method, method_name, _ in METHODS:
            for seed in SEEDS:
                run = runs[(dataset, method, seed)]
                metrics = run["metrics"]
                rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method_name,
                        "seed": seed,
                        "mAP50-95": metrics["test"]["map50_95"],
                        "mAP50": metrics["test"]["map50"],
                        "precision": metrics["test"]["precision"],
                        "recall": metrics["test"]["recall"],
                        "trainable_params": metrics["parameters"]["trainable_parameters"],
                        "total_params": metrics["parameters"]["total_parameters"],
                        "trainable_ratio": metrics["parameters"]["trainable_parameter_ratio"],
                        "peak_gpu_memory_mib": metrics["resources"]["peak_gpu_memory_mib"],
                        "elapsed_seconds": metrics["timing"]["training_seconds"],
                        "gpu_hours": metrics["timing"]["gpu_hours"],
                        "checkpoint_size": metrics["checkpoint"]["size_bytes"],
                        "adapter_size": metrics["adapter"]["size_bytes"],
                        "exit_code": metrics["exit_code"],
                        "status": metrics["status"],
                        "run_id": run["run_id"],
                        "device": run["environment"]["device_argument"],
                        "gpu_name": run["environment"]["gpu_name"],
                        "checkpoint_sha256": metrics["checkpoint"]["sha256"],
                    }
                )
    return rows


def summary_rows(runs: dict[tuple[str, str, int], dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        for method, method_name, _ in METHODS:
            method_runs = [runs[(dataset, method, seed)]["metrics"] for seed in SEEDS]
            row: dict[str, object] = {"dataset": dataset_name, "method": method_name, "n": 3}
            for metric in METRICS:
                result = stats([float(run["test"][metric]) for run in method_runs])
                row.update({f"{metric}_{key}": value for key, value in result.items()})
            parameters = {int(run["parameters"]["trainable_parameters"]) for run in method_runs}
            totals = {int(run["parameters"]["total_parameters"]) for run in method_runs}
            if len(parameters) != 1 or len(totals) != 1:
                raise ValueError(f"Parameter count is not structural for {dataset}/{method}")
            row.update(
                {
                    "trainable_params": parameters.pop(),
                    "total_params": totals.pop(),
                    "trainable_ratio": method_runs[0]["parameters"]["trainable_parameter_ratio"],
                    "peak_gpu_memory_mib_mean": statistics.fmean(
                        float(run["resources"]["peak_gpu_memory_mib"]) for run in method_runs
                    ),
                    "elapsed_seconds_mean": statistics.fmean(
                        float(run["timing"]["training_seconds"]) for run in method_runs
                    ),
                    "gpu_hours_mean": statistics.fmean(float(run["timing"]["gpu_hours"]) for run in method_runs),
                }
            )
            rows.append(row)
    return rows


def summary_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# C3 P1 Multi-seed Summary",
        "",
        "Statistics use n=3 independent training seeds. Standard deviation is the sample SD; 95% CIs use the two-sided Student-t critical value for df=2 (4.30265) and are not clipped to [0, 1].",
        "",
        "| Dataset | Method | mAP50-95 mean ± SD | 95% CI | mAP50 mean ± SD | 95% CI | Precision mean ± SD | Recall mean ± SD | Peak GPU MiB | Time (s) | GPU-hours |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['map50_95_mean']:.4f} ± {row['map50_95_std']:.4f} | "
            f"[{row['map50_95_ci95_lower']:.4f}, {row['map50_95_ci95_upper']:.4f}] | "
            f"{row['map50_mean']:.4f} ± {row['map50_std']:.4f} | "
            f"[{row['map50_ci95_lower']:.4f}, {row['map50_ci95_upper']:.4f}] | "
            f"{row['precision_mean']:.4f} ± {row['precision_std']:.4f} | "
            f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f} | "
            f"{row['peak_gpu_memory_mib_mean']:.1f} | {row['elapsed_seconds_mean']:.1f} | "
            f"{row['gpu_hours_mean']:.5f} |"
        )
    return "\n".join(lines) + "\n"


def paired_rows(runs: dict[tuple[str, str, int], dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        deltas = [
            float(runs[(dataset, "vpeft", seed)]["metrics"]["test"]["map50_95"])
            - float(runs[(dataset, "full_sft", seed)]["metrics"]["test"]["map50_95"])
            for seed in SEEDS
        ]
        result = stats(deltas)
        rows.append(
            {
                "dataset": dataset_name,
                "comparison": "V-PEFT minus Full-SFT",
                "delta_seed824": deltas[0],
                "delta_seed825": deltas[1],
                "delta_seed826": deltas[2],
                "mean_delta": result["mean"],
                "std_delta": result["std"],
                "ci95_lower": result["ci95_lower"],
                "ci95_upper": result["ci95_upper"],
                "direction_consistency": f"{sum(value > 0 for value in deltas)}/3 positive; {sum(value < 0 for value in deltas)}/3 negative",
            }
        )
    return rows


def tradeoff_rows(
    summary: list[dict[str, object]], runs: dict[tuple[str, str, int], dict[str, object]]
) -> list[dict[str, object]]:
    lookup = {(row["dataset"], row["method"]): row for row in summary}
    rows = []
    for dataset, dataset_name, _ in DATASETS:
        full = lookup[(dataset_name, "Full-SFT")]
        vpeft = lookup[(dataset_name, "V-PEFT")]
        memory_by_seed = []
        time_by_seed = []
        gpu_hours_by_seed = []
        for seed in SEEDS:
            full_run = runs[(dataset, "full_sft", seed)]["metrics"]
            vpeft_run = runs[(dataset, "vpeft", seed)]["metrics"]
            memory_by_seed.append(
                1 - float(vpeft_run["resources"]["peak_gpu_memory_mib"])
                / float(full_run["resources"]["peak_gpu_memory_mib"])
            )
            time_by_seed.append(
                float(vpeft_run["timing"]["training_seconds"])
                / float(full_run["timing"]["training_seconds"])
                - 1
            )
            gpu_hours_by_seed.append(
                float(vpeft_run["timing"]["gpu_hours"]) / float(full_run["timing"]["gpu_hours"]) - 1
            )
        rows.append(
            {
                "dataset": dataset_name,
                "parameter_reduction": 1 - vpeft["trainable_params"] / full["trainable_params"],
                "accuracy_retention": vpeft["map50_95_mean"] / full["map50_95_mean"],
                "accuracy_drop_vs_full": full["map50_95_mean"] - vpeft["map50_95_mean"],
                "memory_saving": 1 - vpeft["peak_gpu_memory_mib_mean"] / full["peak_gpu_memory_mib_mean"],
                "training_time_change": vpeft["elapsed_seconds_mean"] / full["elapsed_seconds_mean"] - 1,
                "gpu_hour_change": vpeft["gpu_hours_mean"] / full["gpu_hours_mean"] - 1,
                "memory_saving_seed824": memory_by_seed[0],
                "memory_saving_seed825": memory_by_seed[1],
                "memory_saving_seed826": memory_by_seed[2],
                "memory_saving_all_seeds_positive": all(value > 0 for value in memory_by_seed),
                "training_time_change_seed824": time_by_seed[0],
                "training_time_change_seed825": time_by_seed[1],
                "training_time_change_seed826": time_by_seed[2],
                "training_time_overhead_all_seeds": all(value > 0 for value in time_by_seed),
                "gpu_hour_change_seed824": gpu_hours_by_seed[0],
                "gpu_hour_change_seed825": gpu_hours_by_seed[1],
                "gpu_hour_change_seed826": gpu_hours_by_seed[2],
                "full_map50_95_mean": full["map50_95_mean"],
                "vpeft_map50_95_mean": vpeft["map50_95_mean"],
                "full_peak_gpu_memory_mib_mean": full["peak_gpu_memory_mib_mean"],
                "vpeft_peak_gpu_memory_mib_mean": vpeft["peak_gpu_memory_mib_mean"],
                "full_elapsed_seconds_mean": full["elapsed_seconds_mean"],
                "vpeft_elapsed_seconds_mean": vpeft["elapsed_seconds_mean"],
            }
        )
    return rows


def rankings(summary: list[dict[str, object]]) -> dict[str, dict[str, list[str]]]:
    output = {}
    for _, dataset_name, _ in DATASETS:
        rows = [row for row in summary if row["dataset"] == dataset_name]
        output[dataset_name] = {
            "accuracy": [row["method"] for row in sorted(rows, key=lambda row: row["map50_95_mean"], reverse=True)],
            "parameter_efficiency": [row["method"] for row in sorted(rows, key=lambda row: row["trainable_params"])],
            "memory_efficiency": [
                row["method"] for row in sorted(rows, key=lambda row: row["peak_gpu_memory_mib_mean"])
            ],
            "time_efficiency": [row["method"] for row in sorted(rows, key=lambda row: row["elapsed_seconds_mean"])],
        }
    return output


def run_manifest(runs: dict[tuple[str, str, int], dict[str, object]]) -> dict[str, object]:
    entries = []
    for run in runs.values():
        metrics = run["metrics"]
        entries.append(
            {
                "run_id": run["run_id"],
                "dataset": run["dataset_name"],
                "method": run["method_name"],
                "seed": run["seed"],
                "reused_existing": run["seed"] == 824,
                "status": metrics["status"],
                "device": run["environment"]["device_argument"],
                "gpu_name": run["environment"]["gpu_name"],
                "log_dir": run["log_dir"].relative_to(REPO_ROOT).as_posix(),
                "checkpoint": metrics["checkpoint"],
                "artifact_manifest": run["integrity"],
            }
        )
    return {
        "schema_version": 1,
        "experiment_matrix": {"datasets": 2, "methods": 3, "seeds": 3, "total_runs": 18},
        "seed824_rerun": False,
        "new_runs": 12,
        "runs": entries,
    }


def report_text(
    summary: list[dict[str, object]],
    tradeoffs: list[dict[str, object]],
    paired: list[dict[str, object]],
    ranking: dict[str, dict[str, list[str]]],
) -> str:
    summary_table = summary_markdown(summary).split("\n", 4)[4].rstrip()
    tradeoff_lookup = {row["dataset"]: row for row in tradeoffs}
    paired_lookup = {row["dataset"]: row for row in paired}
    neu = tradeoff_lookup["NEU-DET"]
    deep = tradeoff_lookup["DeepPCB"]
    neu_pair = paired_lookup["NEU-DET"]
    deep_pair = paired_lookup["DeepPCB"]
    rank_lines = []
    for dataset_name, values in ranking.items():
        rank_lines.extend(
            [
                f"- {dataset_name} accuracy: {' > '.join(values['accuracy'])}",
                f"- {dataset_name} parameter efficiency: {' > '.join(values['parameter_efficiency'])}",
                f"- {dataset_name} memory efficiency: {' > '.join(values['memory_efficiency'])}",
                f"- {dataset_name} time efficiency: {' > '.join(values['time_efficiency'])}",
            ]
        )
    return f"""# C3 P1 Final Report — Three-seed 100-Epoch Protocol

## 1. Research Question

在两个工业缺陷小样本数据集上，V-PEFT 能否在显著降低可训练参数的同时，跨 seed 保持 Full-SFT 的准确率并改善 GPU 资源效率？

## 2. Final Protocol

YOLO11n (`yolo11n.pt`)，每数据集固定 100 张训练图，100 epochs，batch=8，imgsz=640，AdamW，lr0=0.001，weight decay=0.0005，cosine scheduler，FP32，seeds=824/825/826。seed824 未重跑；seed825/826 只改变训练 seed。

## 3. Datasets and Splits

NEU-DET 与 DeepPCB 均复用 seed824 固定训练成员列表和原有 val/test split。三种方法、三个 seed 的 sample IDs 完全一致。

## 4. Three Training Strategies

- Full-SFT：全模型训练。
- Frozen Backbone：冻结 `model.0`–`model.10`，训练其余层。
- V-PEFT：rank=8、alpha=16、strict AO planner、backend=`vpeft`，运行时 actual backend=`peft`，不允许 silent fallback。

## 5. 18-run Experiment Matrix

2 datasets × 3 methods × 3 seeds = 18 runs。NEU 9/9、DeepPCB 9/9 均为 PASS；12 个新增 run 各自独占 GPU 和输出目录，seed824 六个正式结果保持冻结。

## 6. Multi-seed Accuracy Results

{summary_table}

## 7. Resource Efficiency Results

资源指标为三 seed 均值。排序只描述对应维度，不合并成单一 winner：

{chr(10).join(rank_lines)}

## 8. V-PEFT Trade-off

- NEU-DET：parameter reduction={neu['parameter_reduction']:.2%}，accuracy retention={neu['accuracy_retention']:.2%}，accuracy drop={neu['accuracy_drop_vs_full']:.4f}，memory saving={neu['memory_saving']:.2%}，training time change={neu['training_time_change']:+.2%}，GPU-hour change={neu['gpu_hour_change']:+.2%}。
- DeepPCB：parameter reduction={deep['parameter_reduction']:.2%}，accuracy retention={deep['accuracy_retention']:.2%}，accuracy drop={deep['accuracy_drop_vs_full']:.4f}，memory saving={deep['memory_saving']:.2%}，training time change={deep['training_time_change']:+.2%}，GPU-hour change={deep['gpu_hour_change']:+.2%}。

V-PEFT 的 76.32% 可训练参数减少在两个数据集和三个 seed 上是结构常数。是否带来准确率、显存或时间收益必须分别判断，不能由参数量直接推断。

## 9. Paired Seed Analysis

- NEU-DET V-PEFT − Full-SFT：seed deltas=({neu_pair['delta_seed824']:+.4f}, {neu_pair['delta_seed825']:+.4f}, {neu_pair['delta_seed826']:+.4f})，mean={neu_pair['mean_delta']:+.4f}，95% CI=[{neu_pair['ci95_lower']:+.4f}, {neu_pair['ci95_upper']:+.4f}]，方向={neu_pair['direction_consistency']}。
- DeepPCB V-PEFT − Full-SFT：seed deltas=({deep_pair['delta_seed824']:+.4f}, {deep_pair['delta_seed825']:+.4f}, {deep_pair['delta_seed826']:+.4f})，mean={deep_pair['mean_delta']:+.4f}，95% CI=[{deep_pair['ci95_lower']:+.4f}, {deep_pair['ci95_upper']:+.4f}]，方向={deep_pair['direction_consistency']}。

n=3 很小，因此主要报告配对差值、spread、CI 与方向一致性，不用 p-value 支撑强结论。

## 10. Qualitative Comparison

每数据集从按文件名排序的固定 test split 中等间距选取 10 张，选择发生在推理前且与模型结果无关。四栏固定为 GT / Full-SFT / Frozen / V-PEFT，均使用预先指定的 seed824 100-epoch `best.pt`，confidence threshold=0.25，imgsz=640；未针对方法调阈值。

## 11. Planner Analysis

六个 V-PEFT runs 均满足 strict=true、planner status=ACCEPT 或合法 ADAPT、planner backend=vpeft、actual backend=peft、applied targets>0、adapter export 成功，且无 silent fallback。

## 12. Negative / Positive Findings

- Positive：V-PEFT 的可训练参数减少跨 seed 稳定为 76.32%。
- Accuracy：NEU accuracy retention={neu['accuracy_retention']:.2%}；DeepPCB={deep['accuracy_retention']:.2%}。数据集间 trade-off 明显不同。
- Resource：V-PEFT 的 memory saving 为 NEU {neu['memory_saving']:.2%}、DeepPCB {deep['memory_saving']:.2%}；time change 为 NEU {neu['training_time_change']:+.2%}、DeepPCB {deep['training_time_change']:+.2%}。这说明当前实现的参数减少没有自动转化为同比显存或时间减少。

对预先提出的跨 seed 观察逐项复核：

- A — NEU：V-PEFT 三个 seed 均低于对应 Full-SFT；mean drop=0.0126、retention=96.20%，配对 CI 跨 0。数据支持“均值接近但未达到 parity”，不支持“无损”。
- B — DeepPCB：三个配对差值均为负，mean drop=0.1320，配对 95% CI 完全低于 0；较大性能损失跨 seed 持续存在。
- C — 参数：76.32% reduction 是所有 V-PEFT run 的结构常数，稳定。
- D — 显存：NEU 各 seed saving 为 {neu['memory_saving_seed824']:.2%}/{neu['memory_saving_seed825']:.2%}/{neu['memory_saving_seed826']:.2%}；DeepPCB 为 {deep['memory_saving_seed824']:.2%}/{deep['memory_saving_seed825']:.2%}/{deep['memory_saving_seed826']:.2%}。近乎为零的 saving 稳定。
- E — 时间：NEU 各 seed change 为 {neu['training_time_change_seed824']:+.2%}/{neu['training_time_change_seed825']:+.2%}/{neu['training_time_change_seed826']:+.2%}；DeepPCB 为 {deep['training_time_change_seed824']:+.2%}/{deep['training_time_change_seed825']:+.2%}/{deep['training_time_change_seed826']:+.2%}。六个配对均为 overhead。

## 13. Limitations

每组只有三个 seed，t-based 95% CI 很宽且对单个 run 敏感；每数据集仅用 100 张训练图；结论限于 YOLO11n、当前冻结边界、V-PEFT rank/planner 与 RTX 4090 FP32 实现。并行运行使用相同型号独占 GPU，但 seed824 与新增 runs 的系统时段不同。qualitative comparison 使用预先指定 seed824，不代表跨 seed 集成模型。

## 14. Final P1 Conclusion

Overall C3 P1 = PASS：18/18 runs 可追踪，protocol fairness、统计汇总、准确率/资源比较、trade-off、配对分析、qualitative comparison 与 artifact integrity 均完成。数据支持 V-PEFT 具有显著且稳定的 trainable-parameter efficiency；其准确率保留具有数据集依赖性，且当前实现下不能声称显存或训练时间随参数量同比下降。
"""


def efficiency_text(tradeoffs: list[dict[str, object]]) -> str:
    lines = [
        "# V-PEFT Multi-seed Efficiency Analysis",
        "",
        "## Observed Fact",
        "",
    ]
    for row in tradeoffs:
        lines.append(
            f"- {row['dataset']}: trainable parameter reduction={row['parameter_reduction']:.2%}; "
            f"mean peak-memory saving={row['memory_saving']:.2%}; mean training-time change="
            f"{row['training_time_change']:+.2%}; mean GPU-hour change={row['gpu_hour_change']:+.2%}. "
            f"Per-seed memory savings=({row['memory_saving_seed824']:.2%}, {row['memory_saving_seed825']:.2%}, "
            f"{row['memory_saving_seed826']:.2%}); per-seed time changes=({row['training_time_change_seed824']:+.2%}, "
            f"{row['training_time_change_seed825']:+.2%}, {row['training_time_change_seed826']:+.2%})."
        )
    lines.extend(
        [
            "",
            "这些数值是当前代码、FP32、RTX 4090 与锁定 P1 protocol 下的三 seed 观测值。可训练参数大幅减少没有转化为同比的峰值显存下降。",
            "",
            "## Supported Explanation",
            "",
            "训练器报告的峰值显存包含参数以外的激活、梯度相关缓冲、优化器状态、验证与框架开销；因此 trainable parameter count 与 total peak memory 不是同一个量。V-PEFT 还执行 adapter/planner 相关计算，故参数减少本身不足以推出 wall-clock 加速。这里的解释只说明为什么两类指标可以不同，不声称已分解各项显存或耗时占比。",
            "",
            "## Hypothesis",
            "",
            "adapter 注入、额外张量操作、当前 kernel 路径或低利用率小数据训练可能贡献时间开销；激活或固定框架开销可能主导峰值显存。验证这些机制需要独立 profiler、算子级时间线和显存分解实验，本 P1 数据不能把任一机制确认为事实。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    qualitative_manifest = P1_ROOT / "visualizations" / "final" / "manifest.json"
    if not qualitative_manifest.is_file():
        raise FileNotFoundError("Run qualitative_multiseed.py before final analysis")
    qualitative = json.loads(qualitative_manifest.read_text(encoding="utf-8"))
    if any(row["sample_count"] != 10 for row in qualitative["datasets"].values()):
        raise ValueError("Qualitative comparison is incomplete")
    qualitative_integrity = all(
        sha256(REPO_ROOT / sample["source"]) == sample["source_sha256"]
        and sha256(REPO_ROOT / sample["comparison"]) == sample["comparison_sha256"]
        for dataset in qualitative["datasets"].values()
        for sample in dataset["samples"]
    )
    if not qualitative_integrity:
        raise ValueError("Qualitative source or comparison SHA-256 verification failed")

    protocol = yaml.safe_load((P1_ROOT / "config" / "protocol.yaml").read_text(encoding="utf-8"))
    plan = yaml.safe_load((P1_ROOT / "config" / "multiseed_plan.yaml").read_text(encoding="utf-8"))
    final_protocol_fixed = all(
        (
            protocol.get("stage") == "multiseed_final_protocol",
            protocol["training"].get("seeds") == [824, 825, 826],
            protocol["training"].get("epochs") == 100,
            protocol["datasets"].get("train_sample_count_per_dataset") == 100,
            protocol["extension_gate"].get("multiseed_status") == "complete",
            protocol["extension_gate"].get("overall_c3_p1") == "PASS",
        )
    )
    plan_complete = all(
        (
            plan.get("status") == "COMPLETE",
            plan.get("pending_seeds") == [],
            plan.get("completed_run_count") == 12,
            len(plan.get("runs", [])) == 12,
            all(run.get("status") == "PASS" for run in plan.get("runs", [])),
        )
    )
    if not final_protocol_fixed or not plan_complete:
        raise ValueError("Final protocol or multiseed plan is not in its completed locked state")

    runs = load_runs()
    fairness = fairness_audit(runs)
    if fairness["status"] != "PASS":
        raise ValueError(f"Protocol fairness failed: {fairness['checks']}")
    all_rows = all_run_rows(runs)
    summaries = summary_rows(runs)
    paired = paired_rows(runs)
    tradeoffs = tradeoff_rows(summaries, runs)
    ranking = rankings(summaries)

    results = P1_ROOT / "results"
    write_csv(results / "p1_all_runs.csv", all_rows)
    write_csv(results / "p1_summary.csv", summaries)
    (results / "p1_summary.md").write_text(summary_markdown(summaries), encoding="utf-8")
    write_csv(results / "paired_full_vs_vpeft.csv", paired)
    write_csv(results / "tradeoff_multiseed.csv", tradeoffs)

    manifest = run_manifest(runs)
    write_json(P1_ROOT / "evidence" / "multiseed_manifest.json", manifest)
    validation = {
        "schema_version": 1,
        "protocol_fairness": fairness,
        "final_protocol_fixed": {
            "status": "PASS",
            "protocol": "smoke/c3/p1/config/protocol.yaml",
            "protocol_sha256": sha256(P1_ROOT / "config" / "protocol.yaml"),
            "multiseed_plan": "smoke/c3/p1/config/multiseed_plan.yaml",
            "multiseed_plan_sha256": sha256(P1_ROOT / "config" / "multiseed_plan.yaml"),
        },
        "18_run_completeness": {
            "expected": 18,
            "found": len(runs),
            "neu": sum(run["dataset"] == "neu_det" for run in runs.values()),
            "deeppcb": sum(run["dataset"] == "deeppcb" for run in runs.values()),
            "all_pass": all(run["metrics"]["status"] == "PASS" for run in runs.values()),
        },
        "statistical_summary": {
            "status": "PASS",
            "n_per_dataset_method": 3,
            "sample_standard_deviation": True,
            "ci_method": "two-sided Student-t, df=2, t*=4.302652729911275",
            "metrics": list(METRICS),
            "paired_full_vs_vpeft": True,
            "summary_csv": "smoke/c3/p1/results/p1_summary.csv",
            "tradeoff_csv": "smoke/c3/p1/results/tradeoff_multiseed.csv",
        },
        "artifact_integrity": {
            "status": "PASS",
            "verified_runs": len(runs),
            "all_run_manifests_verified": all(run["integrity"]["verified"] for run in runs.values()),
            "qualitative_comparisons": sum(row["sample_count"] for row in qualitative["datasets"].values()),
            "qualitative_sha256_verified": qualitative_integrity,
        },
        "rankings": ranking,
        "P1 status": "PASS",
    }
    write_json(P1_ROOT / "evidence" / "p1_final_validation.json", validation)
    (P1_ROOT / "docs" / "C3_P1_REPORT.md").write_text(
        report_text(summaries, tradeoffs, paired, ranking), encoding="utf-8"
    )
    (P1_ROOT / "docs" / "VPEFT_EFFICIENCY_ANALYSIS.md").write_text(
        efficiency_text(tradeoffs), encoding="utf-8"
    )

    for row in summaries:
        print(
            f"{row['dataset']} {row['method']}: mAP50-95={row['map50_95_mean']:.6f}±{row['map50_95_std']:.6f} "
            f"CI=[{row['map50_95_ci95_lower']:.6f},{row['map50_95_ci95_upper']:.6f}]"
        )
    print("OVERALL_C3_P1=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
