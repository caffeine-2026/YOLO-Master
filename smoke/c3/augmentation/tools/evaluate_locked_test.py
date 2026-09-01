#!/usr/bin/env python3
"""Evaluate one frozen checkpoint on the locked test exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke.c3.p1.scripts import run_p1 as common
from ultralytics import YOLO

ROOT = REPO_ROOT / "smoke" / "c3" / "augmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("neu", "deeppcb"), required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--sample-size", type=int, choices=(10, 50, 100, 500), default=100)
    parser.add_argument("--seed", type=int, choices=(824, 825, 826), required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen_path = ROOT / "results" / "frozen_selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("test_access_before_selection") is not False:
        raise ValueError("Policy was not frozen without test access")
    allowed = {"baseline", frozen["datasets"][args.dataset]["frozen_policy"]}
    if args.policy not in allowed:
        raise ValueError(f"Locked test policy must be one of {sorted(allowed)}")

    if args.sample_size != 100 and not frozen["datasets"][args.dataset]["scaling_trigger_passed"]:
        raise ValueError("Scaling test evaluation is prohibited when the validation trigger did not pass")
    phase = ("search" if args.seed == 824 else "confirm") if args.sample_size == 100 else "scaling"
    run_id = f"{phase}_{args.dataset}_{args.sample_size}_{args.policy}_seed{args.seed}_e100"
    training_metrics_path = ROOT / "logs" / phase / run_id / "metrics.json"
    training = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    if training.get("status") != "PASS" or training.get("test") is not None:
        raise ValueError(f"Invalid source training evidence: {run_id}")
    checkpoint_value = training["checkpoints"]["best"]
    checkpoint = REPO_ROOT / checkpoint_value["path"]
    if common.sha256(checkpoint) != checkpoint_value["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {run_id}")
    if training["parameters"]["trainable_parameters"] != 613_602:
        raise ValueError(f"Wrong accuracy-first V-PEFT parameter count: {run_id}")

    eval_id = f"test_{args.dataset}_{args.sample_size}_{args.policy}_seed{args.seed}"
    eval_dir = ROOT / "artifacts" / "locked_test" / eval_id
    output = ROOT / "evaluations" / f"{eval_id}.json"
    if eval_dir.exists() or output.exists():
        raise FileExistsError(f"Refusing to overwrite locked test evaluation: {eval_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = REPO_ROOT / "smoke" / "c3" / "p2" / "config" / "splits" / f"{args.dataset}_{args.sample_size}.yaml"
    model = YOLO(checkpoint)
    metrics = model.val(
        data=data.as_posix(),
        split="test",
        device=args.device,
        batch=8,
        imgsz=640,
        workers=0,
        quantize=32,
        plots=True,
        save_json=False,
        save_dir=eval_dir.as_posix(),
    )
    values = {key: float(value) for key, value in metrics.results_dict.items()}
    names = metrics.names
    per_class = []
    for position, class_index in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(position)
        class_id = int(class_index)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "precision": float(precision),
                "recall": float(recall),
                "ap50": float(ap50),
                "ap50_95": float(ap),
            }
        )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "split": "locked_test",
        "selection_frozen_before_test": True,
        "test_metrics_used_for_selection": False,
        "dataset": args.dataset,
        "sample_size": args.sample_size,
        "policy": args.policy,
        "seed": args.seed,
        "source_run_id": run_id,
        "source_training_metrics": common.relative(training_metrics_path),
        "source_resolved_config": common.relative(training_metrics_path.with_name("resolved_config.yaml")),
        "source_checkpoint": {
            **checkpoint_value,
            "hash_verified_before_load": True,
            "load_succeeded": True,
            "source_best_epoch": training["validation"]["best_epoch"],
            "source_seed": training["seed"],
        },
        "data": {"path": common.relative(data), "sha256": common.sha256(data), "split": "test"},
        "overall": {
            "precision": values["metrics/precision(B)"],
            "recall": values["metrics/recall(B)"],
            "map50": values["metrics/mAP50(B)"],
            "map50_95": values["metrics/mAP50-95(B)"],
        },
        "per_class": per_class,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "save_dir": common.relative(eval_dir),
    }
    common.json_write(output, payload)
    print(json.dumps({"eval_id": eval_id, "status": "PASS", "overall": payload["overall"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
