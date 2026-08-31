#!/usr/bin/env python3
"""Calibrate LOVO from independent dataset/shot units and validate on held-out units."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smoke.c3.completion.tools.run_efficiency import ROOT, SEEDS, selected_candidate
from smoke.c3.p2.tools.summarize_multiseed import run_location
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.utils.lora.planner import (
    ArchitectureFingerprint,
    LOVODataCollector,
    LOVODataPoint,
    LOVOValidator,
    PEFTPlanner,
)

DATASETS = ("neu", "deeppcb")
CALIBRATION_SIZES = (10, 50, 100)
HELDOUT_SIZES = (500,)
MODEL_CONFIG = "ultralytics/cfg/models/11/yolo11n.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def last_validation(path: Path) -> float:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100:
        raise ValueError(f"Expected 100 validation epochs: {path.relative_to(REPO_ROOT)}")
    value = float(rows[-1]["metrics/mAP50-95(B)"])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite validation metric: {path.relative_to(REPO_ROOT)}")
    return value


def unit_observation(
    dataset: str, size: int, candidate: str, fingerprint: ArchitectureFingerprint
) -> tuple[LOVODataPoint, dict]:
    paired = []
    sources = []
    for seed in SEEDS:
        full_run_id, full_dir, _ = run_location(dataset, size, "full", seed)
        new_run_id = f"final_{dataset}_{size}_vpeft_{candidate}_seed{seed}_e100"
        new_dir = ROOT / "logs" / "final" / new_run_id
        full_curve = full_dir / "learning_curve.csv"
        new_curve = new_dir / "learning_curve.csv"
        full_value = last_validation(full_curve)
        new_value = last_validation(new_curve)
        paired.append(
            {
                "seed": seed,
                "full_run_id": full_run_id,
                "vpeft_run_id": new_run_id,
                "full_last_validation_map50_95": full_value,
                "vpeft_last_validation_map50_95": new_value,
                "delta_map50_95": new_value - full_value,
                "full_curve_path": full_curve.relative_to(REPO_ROOT).as_posix(),
                "full_curve_sha256": sha256(full_curve),
                "vpeft_curve_path": new_curve.relative_to(REPO_ROOT).as_posix(),
                "vpeft_curve_sha256": sha256(new_curve),
            }
        )
        sources.extend([full_run_id, new_run_id])
    deltas = [row["delta_map50_95"] for row in paired]
    observation_id = f"{dataset}-{size}shot-selected-vpeft-vs-full-validation-last-epoch"
    point = LOVODataPoint(
        fingerprint=fingerprint,
        variant="lora",
        delta_mAP=statistics.fmean(deltas),
        model_name="yolo11n-nc6",
        dataset=f"{dataset}-{size}shot",
        epochs=100,
        rank=8,
        notes="Mean paired delta across seeds 824/825/826; one dataset-shot configuration is one observation.",
        observation_id=observation_id,
        metric_split="validation_last_epoch",
        source_run_ids=sources,
    )
    detail = {
        "observation_id": observation_id,
        "dataset": dataset,
        "sample_size": size,
        "aggregation_unit": "dataset_x_sample_size",
        "seeds_are_replicates_not_observations": True,
        "delta_definition": "selected_vpeft_minus_full_sft",
        "metric_split": "validation_last_epoch",
        "mean_delta_map50_95": statistics.fmean(deltas),
        "sample_std_delta_map50_95": statistics.stdev(deltas),
        "paired_seed_runs": paired,
    }
    return point, detail


def main() -> int:
    output_dir = ROOT / "evidence" / "lovo"
    report_path = output_dir / "lovo_calibration_report.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite formal LOVO evidence")
    candidate = selected_candidate()
    model = DetectionModel(MODEL_CONFIG, ch=3, nc=6, verbose=False)
    # Match ``PEFTPlanner.predict_advisory`` exactly: it fingerprints the
    # unwrapped detection graph rather than the DetectionModel wrapper.
    fingerprint = ArchitectureFingerprint.compute(getattr(model, "model", model))
    calibration_pairs = [
        unit_observation(dataset, size, candidate, fingerprint) for dataset in DATASETS for size in CALIBRATION_SIZES
    ]
    heldout_pairs = [
        unit_observation(dataset, size, candidate, fingerprint) for dataset in DATASETS for size in HELDOUT_SIZES
    ]
    collector = LOVODataCollector()
    for point, _ in calibration_pairs:
        collector.add(point)
    calibration_ids = {point.observation_id for point, _ in calibration_pairs}
    heldout_ids = {point.observation_id for point, _ in heldout_pairs}
    calibration_source_ids = [run_id for point, _ in calibration_pairs for run_id in point.source_run_ids]
    heldout_source_ids = [run_id for point, _ in heldout_pairs for run_id in point.source_run_ids]
    if (
        len(collector) < 5
        or calibration_ids & heldout_ids
        or len(calibration_source_ids) != len(set(calibration_source_ids))
        or len(heldout_source_ids) != len(set(heldout_source_ids))
        or set(calibration_source_ids) & set(heldout_source_ids)
    ):
        raise RuntimeError("LOVO calibration/held-out separation contract failed")

    validator = LOVOValidator()
    planner = PEFTPlanner(lovo_collector=collector, lovo_validator=validator)
    config = LoRAConfig(r=8, alpha=16, planner_enabled=True, planner_backend="vpeft")
    advisory = planner.predict_advisory(model, config)
    if not (
        advisory["evidence_observation_count"] >= 5
        and advisory["uses_learned_evidence"] is True
        and advisory["evidence_source"] != "default_prior"
        and advisory["confidence_score"] > 0
    ):
        raise RuntimeError(f"Learned LOVO evidence contract failed: {advisory}")
    cross_validation = validator.full_report(collector)
    heldout_results = []
    for point, detail in heldout_pairs:
        predicted, std_error = planner.predict_with_uncertainty(point.fingerprint, point.variant, point.rank)
        lower = predicted - 1.96 * std_error
        upper = predicted + 1.96 * std_error
        actual = point.delta_mAP
        heldout_results.append(
            {
                **detail,
                "predicted_delta_map50_95": predicted,
                "actual_delta_map50_95": actual,
                "error": predicted - actual,
                "absolute_error": abs(predicted - actual),
                "prediction_std_error": std_error,
                "prediction_interval_95": [lower, upper],
                "covered_by_interval_95": lower <= actual <= upper,
            }
        )
    errors = [row["error"] for row in heldout_results]
    heldout_metrics = {
        "n": len(heldout_results),
        "mae": statistics.fmean(abs(value) for value in errors),
        "rmse": math.sqrt(statistics.fmean(value**2 for value in errors)),
        "mean_error": statistics.fmean(errors),
        "coverage_95": statistics.fmean(float(row["covered_by_interval_95"]) for row in heldout_results),
    }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/calibrate_lovo.py",
        "status": "LEARNED_CALIBRATION_WITH_HELDOUT_VALIDATION",
        "candidate": candidate,
        "metric_policy": {
            "split": "validation",
            "epoch": "last epoch (100), avoiding best-epoch selection leakage",
            "locked_test_read": False,
            "delta": "selected V-PEFT minus Full-SFT under paired seed/protocol",
            "seed_replicates_aggregated": True,
            "observation_unit": "unique dataset x sample-size configuration",
        },
        "calibration_observation_ids": sorted(calibration_ids),
        "heldout_observation_ids": sorted(heldout_ids),
        "sets_disjoint": not bool(calibration_ids & heldout_ids),
        "source_run_sets_disjoint": not bool(set(calibration_source_ids) & set(heldout_source_ids)),
        "calibration_source_run_count": len(calibration_source_ids),
        "heldout_source_run_count": len(heldout_source_ids),
        "calibration_observations": [detail for _, detail in calibration_pairs],
        "heldout_results": heldout_results,
        "advisory": advisory,
        "calibration_cross_validation": cross_validation,
        "heldout_metrics": heldout_metrics,
        "cold_start_comparison": {
            "prior_predicted_delta": 0.066,
            "prior_source": "default_prior",
            "prior_confidence": 0,
            "prior_is_measurement": False,
            "learned_predicted_delta": advisory["predicted_delta"],
        },
        "limitations": [
            "All observations use one YOLO11n architecture and one LoRA variant, so the learned design matrix is rank deficient.",
            "Shot subsets are nested and each dataset reuses its fixed validation images; dataset-shot configurations are unique experimental units but not disjoint image cohorts.",
            "The sample count is small, therefore the computed confidence remains low and must not be presented as strong evidence.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    collector.save(output_dir / "calibration_observations.json")
    (output_dir / "heldout_observations.json").write_text(
        json.dumps([point.to_dict() for point, _ in heldout_pairs], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "observation_count": advisory["evidence_observation_count"],
                "source": advisory["evidence_source"],
                "confidence": advisory["confidence_score"],
                "heldout_rmse": heldout_metrics["rmse"],
                "coverage_95": heldout_metrics["coverage_95"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
