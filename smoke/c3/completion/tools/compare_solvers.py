#!/usr/bin/env python3
"""Run AO, DCO, and native OR-Tools MIP on identical C3 model constraints."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.lora.api import _build_vpeft_placement_plan
from ultralytics.utils.lora.config import LoRAConfig

MODEL_CONFIG = "ultralytics/cfg/models/11/yolo11n.yaml"
DATASETS = ("neu", "deeppcb")
SOLVERS = ("ao", "dco", "mip")


def run_solver(dataset: str, solver: str, budget: int) -> dict:
    """Compile one solver plan from an independently constructed real model."""
    model = DetectionModel(MODEL_CONFIG, ch=3, nc=6, verbose=False)
    config = LoRAConfig(
        r=8,
        alpha=16,
        backend="fallback",
        planner_enabled=True,
        planner_backend="vpeft",
        planner_solver=solver,
        adapter_budget=budget,
        vpeft_strict=True,
        exclude_modules=["0.conv"],
    )
    started = time.perf_counter()
    plan = _build_vpeft_placement_plan(model, config)
    elapsed = time.perf_counter() - started
    diagnostics = dict(plan.metadata.get("solver_diagnostics") or {})
    record = {
        "dataset": dataset,
        "solver": solver,
        "requested_solver": plan.metadata["requested_solver"],
        "effective_solver": plan.metadata["effective_solver"],
        "fallback": bool(plan.metadata["fallback"]),
        "status": plan.status,
        "native_mip_status": diagnostics.get("native_mip_status"),
        "objective": diagnostics.get("objective_value", plan.metadata.get("utility")),
        "utility": plan.metadata.get("utility"),
        "runtime_seconds": diagnostics.get("runtime_seconds", elapsed),
        "measured_wall_seconds": elapsed,
        "budget": plan.budget["max_adapter_params"],
        "planned_parameters": plan.budget["used_adapter_params"],
        "target_module_count": len(plan.targets),
        "target_modules": [target.name for target in plan.targets],
        "rank_pattern": {target.name: target.rank for target in plan.targets},
        "ranks": sorted({target.rank for target in plan.targets}),
        "solver_diagnostics": diagnostics,
    }
    if solver == "mip":
        required = {
            "requested_solver": "mip",
            "effective_solver": "mip",
            "fallback": False,
            "native_mip_status": diagnostics.get("native_mip_status"),
        }
        if required["native_mip_status"] not in {"OPTIMAL", "FEASIBLE"} or any(
            record[key] != value for key, value in required.items() if key != "native_mip_status"
        ):
            raise RuntimeError(f"native MIP contract failed: {record}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=2_100_000)
    parser.add_argument("--output-dir", type=Path, default=Path("smoke/c3/completion/evidence/solvers"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [run_solver(dataset, solver, args.budget) for dataset in DATASETS for solver in SOLVERS]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/compare_solvers.py",
        "constraint_identity": {
            "model_config": MODEL_CONFIG,
            "num_classes": 6,
            "variant": "lora",
            "rank": 8,
            "budget": args.budget,
            "exclude_modules": ["0.conv"],
        },
        "runtime_dependencies": {
            package: importlib.metadata.version(package) for package in ("ortools", "protobuf", "numpy", "pandas")
        },
        "dataset_note": (
            "The architecture and class count are identical across C3 datasets; dataset labels are retained "
            "because these are the plans used for each dataset's training protocol."
        ),
        "runs": records,
    }
    json_path = args.output_dir / "solver_comparison.json"
    csv_path = args.output_dir / "solver_comparison.csv"
    log_path = args.output_dir / "solver_comparison.log"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "solver_suite_start",
                        "generated_at": payload["generated_at"],
                        "command": payload["command"],
                        "constraint_identity": payload["constraint_identity"],
                        "runtime_dependencies": payload["runtime_dependencies"],
                    },
                    sort_keys=True,
                ),
                *(json.dumps({"event": "solver_finish", **record}, sort_keys=True) for record in records),
                json.dumps({"event": "solver_suite_finish", "run_count": len(records), "status": "PASS"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fields = [
        "dataset",
        "solver",
        "requested_solver",
        "effective_solver",
        "fallback",
        "status",
        "native_mip_status",
        "objective",
        "utility",
        "runtime_seconds",
        "measured_wall_seconds",
        "budget",
        "planned_parameters",
        "target_module_count",
        "ranks",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields}
            row["ranks"] = ";".join(str(rank) for rank in record["ranks"])
            writer.writerow(row)
    print(
        json.dumps(
            {"json": json_path.as_posix(), "csv": csv_path.as_posix(), "log": log_path.as_posix(), "runs": len(records)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
