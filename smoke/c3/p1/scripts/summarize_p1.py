#!/usr/bin/env python3
"""Summarize the six seed-824 P1 pilots without ranking methods by accuracy alone."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
P1_ROOT = REPO_ROOT / "smoke" / "c3" / "p1"
DATASETS = (("neu_det", "NEU-DET"), ("deeppcb", "DeepPCB"))
METHODS = (("full_sft", "Full-SFT"), ("frozen_backbone", "Frozen Backbone"), ("vpeft", "V-PEFT"))


def load_runs() -> list[dict[str, object]]:
    runs = []
    for dataset, dataset_name in DATASETS:
        for method, method_name in METHODS:
            run_id = f"{dataset}_{method}_seed824"
            path = P1_ROOT / "logs" / run_id / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path.relative_to(REPO_ROOT))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["dataset_name"] = dataset_name
            payload["method_name"] = method_name
            runs.append(payload)
    return runs


def adapter_mib(run: dict[str, object]) -> float:
    return float(run["adapter"]["size_bytes"]) / 1024**2


def checkpoint_mib(run: dict[str, object]) -> float:
    return float(run["checkpoint"]["size_bytes"]) / 1024**2


def rows_with_ratios(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    full_by_dataset = {run["dataset"]: run for run in runs if run["method"] == "full_sft"}
    rows = []
    for run in runs:
        full = full_by_dataset[run["dataset"]]
        trainable = int(run["parameters"]["trainable_parameters"])
        full_trainable = int(full["parameters"]["trainable_parameters"])
        memory = float(run["resources"]["peak_gpu_memory_mib"])
        full_memory = float(full["resources"]["peak_gpu_memory_mib"])
        seconds = float(run["timing"]["training_seconds"])
        full_seconds = float(full["timing"]["training_seconds"])
        test = run["test"]
        rows.append(
            {
                "dataset": run["dataset_name"],
                "method": run["method_name"],
                "seed": run["seed"],
                "map50_95": float(test["map50_95"]),
                "map50": float(test["map50"]),
                "precision": float(test["precision"]),
                "recall": float(test["recall"]),
                "trainable_parameters": trainable,
                "total_parameters": int(run["parameters"]["total_parameters"]),
                "trainable_ratio_percent": float(run["parameters"]["trainable_parameter_ratio"]) * 100,
                "trainable_vs_full_percent": trainable / full_trainable * 100,
                "peak_gpu_memory_mib": memory,
                "memory_saving_vs_full_percent": (1 - memory / full_memory) * 100,
                "training_seconds": seconds,
                "training_time_ratio_vs_full": seconds / full_seconds,
                "gpu_hours": float(run["timing"]["gpu_hours"]),
                "checkpoint_mib": checkpoint_mib(run),
                "adapter_mib": adapter_mib(run),
                "status": run["status"],
            }
        )
    return rows


def best_epoch(run: dict[str, object]) -> tuple[int, float]:
    curve = P1_ROOT / "logs" / str(run["run_id"]) / "learning_curve.csv"
    with curve.open(encoding="utf-8", newline="") as stream:
        rows = [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    key = "metrics/mAP50-95(B)"
    values = [float(row[key]) for row in rows]
    index = max(range(len(values)), key=values.__getitem__)
    epoch_value = int(float(rows[index].get("epoch", index + 1))) + (0 if float(rows[index].get("epoch", index + 1)) >= 1 else 1)
    return epoch_value, values[index]


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Method | mAP50-95 | mAP50 | Precision | Recall | Trainable / Total | Trainable vs Full | Peak GPU Mem | Memory Saving | Time | GPU-hours | Time Ratio | Checkpoint | Adapter | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['map50_95']:.4f} | {row['map50']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['trainable_parameters']:,} / "
            f"{row['total_parameters']:,} | {row['trainable_vs_full_percent']:.2f}% | "
            f"{row['peak_gpu_memory_mib']:.0f} MiB | {row['memory_saving_vs_full_percent']:.2f}% | "
            f"{row['training_seconds']:.1f}s | {row['gpu_hours']:.5f} | {row['training_time_ratio_vs_full']:.3f}× | "
            f"{row['checkpoint_mib']:.2f} MiB | {row['adapter_mib']:.2f} MiB | {row['status']} |"
        )
    return "\n".join(lines)


def write_report(runs: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    table = markdown_table(rows)
    convergence = []
    for run in runs:
        epoch, value = best_epoch(run)
        convergence.append(
            f"- {run['dataset_name']} / {run['method_name']}: best validation mAP50-95={value:.4f}, epoch={epoch}."
        )
    vpeft_lines = []
    for run in runs:
        if run["method"] != "vpeft":
            continue
        adapter = run["adapter"]
        vpeft_lines.append(
            f"- {run['dataset_name']}: Planner={adapter['planner_status']}, planner backend={adapter['planner_backend']}, "
            f"actual backend={adapter['actual_backend']}, planned/applied={adapter['planned_targets']}/{adapter['applied_targets']}."
        )
    all_pass = all(run["status"] == "PASS" for run in runs)
    late_peak = any(best_epoch(run)[0] >= 27 for run in runs)
    conclusion = (
        "六组 seed824 pilot 均通过闭环验收；本表仅验证公平协议可运行并给出单 seed 初步测量，不声明任一方法优于其他方法。"
        if all_pass
        else "至少一组 seed824 pilot 未通过，当前不能进入多 seed 阶段。"
    )
    epoch_note = (
        "至少一组曲线在最后四个 epoch 内达到最佳值，正式 multi-seed epoch 预算仍需统一收敛审计后锁定。"
        if late_peak
        else "六组最佳值均未集中在最后四个 epoch；仍须在进入 multi-seed 前统一确认最终 epoch。"
    )
    report = f"""# C3 P1 Pilot 报告（seed 824）

## 1. Research Question

在工业缺陷小样本条件下，V-PEFT 是否能以显著更少的可训练参数和资源成本，保持或改善 Full-SFT / Frozen Backbone 的检测性能？

## 2. Protocol

两个数据集统一使用 YOLO11n 预训练权重、100 张训练图、30 epochs、batch 8、imgsz 640、AdamW、lr0=0.001、weight decay=0.0005、cosine scheduler、GPU 0、FP32、seed 824。三个方法仅训练策略不同，最终精度统一在锁定 test split 上评测。

## 3. Dataset / Split

NEU-DET 按来源类别确定性分层抽样；DeepPCB 按 seed824 确定性无放回抽样并审计多标签分布。三种方法共享同一成员列表，原有 val/test 不变且无重叠。详见 `../evidence/P1_DATA_PLAN.md`。

## 4. Three Methods

- Full-SFT：`lora_r=0, freeze=0`。
- Frozen Backbone：依据 `yolo11.yaml` 的结构冻结 `model.0-model.10`，训练 `model.11-model.23`。
- V-PEFT：rank=8、alpha=16、strict V-PEFT AO Planner，不允许 fallback，实际 backend 必须为 PEFT。

## 5. Accuracy Results

{table}

表内为固定 test split 的单 seed 指标；不按 mAP 单独排名。

## 6. Resource Results

同表记录 trainable/total parameters、GPU 峰值、训练耗时、GPU-hours、checkpoint 与 adapter 大小。Memory Saving 和 Time Ratio 均以同数据集 Full-SFT 为基准。

## 7. Multi-seed Statistics

本阶段仅执行 seed=824 pilot。seed=825/826 尚未运行，因此不报告 mean/std/95% CI，也不把当前结果作为正式 P1 多 seed 结论。

## 8. Qualitative Results

正式四栏 GT / Full-SFT / Frozen Backbone / V-PEFT 可视化留到 multi-seed protocol 确认后生成；本阶段没有挑选或替换样本。

## 9. Planner Analysis

{chr(10).join(vpeft_lines)}

## 10. Limitations

当前只有一个 seed，30 epochs 是统一 pilot 预算。收敛记录如下：

{chr(10).join(convergence)}

{epoch_note} 当前差异可能包含随机波动，不能外推为方法普遍优劣。

## 11. P1 Conclusion

{conclusion}
"""
    output = P1_ROOT / "docs" / "C3_P1_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")


def main() -> int:
    runs = load_runs()
    rows = rows_with_ratios(runs)
    csv_path = P1_ROOT / "results" / "pilot_seed824.csv"
    md_path = P1_ROOT / "results" / "pilot_seed824.md"
    write_csv(rows, csv_path)
    table = markdown_table(rows)
    md_path.write_text(
        "# C3 P1 Pilot 三方对照（seed 824）\n\n"
        "以下为统一协议、固定 test split 的单 seed pilot，不构成方法优劣或多 seed 结论。\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )
    write_report(runs, rows)
    status = "PASS" if all(run["status"] == "PASS" for run in runs) else "FAIL"
    print(json.dumps({"status": status, "runs": len(runs)}, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
