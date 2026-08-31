#!/usr/bin/env python3
"""Generate real ACCEPT/ADAPT/REFUSE planner evidence on the C3 YOLO11n model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.utils.lora.planner import PEFTPlanner

MODEL_CONFIG = "ultralytics/cfg/models/11/yolo11n.yaml"


def branch_cases() -> dict[str, LoRAConfig]:
    """Return constraints that naturally exercise each planner branch."""
    common = {
        "r": 8,
        "alpha": 16,
        "backend": "fallback",
        "planner_enabled": True,
        "planner_backend": "vpeft",
        "planner_solver": "ao",
        "vpeft_strict": True,
    }
    return {
        "ACCEPT": LoRAConfig(**common, adapter_budget=2_100_000, include_attention=False),
        # YOLO11n has no attention blocks, so requesting attention targets invokes
        # the architecture guardrail and produces a genuine ADAPT decision.
        "ADAPT": LoRAConfig(**common, adapter_budget=2_100_000, include_attention=True),
        # One parameter cannot fund even the smallest feasible rank-4 adapter.
        "REFUSE": LoRAConfig(**common, adapter_budget=1, include_attention=False),
    }


def _config_dict(config: LoRAConfig) -> dict:
    return {
        "model_config": MODEL_CONFIG,
        "num_classes": 6,
        "variant": config.peft_type,
        "requested_rank": config.r,
        "adapter_budget": config.adapter_budget,
        "requested_solver": config.planner_solver,
        "include_attention": config.include_attention,
        "include_head": config.include_head,
        "only_backbone": config.only_backbone,
        "exclude_modules": list(config.exclude_modules or []),
    }


def run_case(expected_status: str, config: LoRAConfig, audit_dir: Path) -> dict:
    """Run one real planning case and return its traceable evidence payload."""
    model = DetectionModel(MODEL_CONFIG, ch=3, nc=6, verbose=False)
    planner = PEFTPlanner(audit_dir=audit_dir)
    candidate_targets = planner.detect_targets(model, config)
    decision = planner.plan(model, config)
    selected = list(decision.target_modules_hint or [])
    selected_set = set(selected)
    rank_pattern = dict(decision.metadata.get("budget_rank_pattern") or {})
    payload = {
        "expected_status": expected_status,
        "input": _config_dict(config),
        "decision": decision.to_dict(),
        "candidate_modules": candidate_targets,
        "candidate_module_count": len(candidate_targets),
        "selected_modules": selected,
        "selected_module_count": len(selected),
        "rejected_modules": [name for name in candidate_targets if name not in selected_set],
        "rejected_module_count": len(candidate_targets) - len(selected),
        "effective_rank_pattern": rank_pattern,
        "effective_ranks": sorted(set(rank_pattern.values())),
        "guardrail_result": {
            "triggered": bool(decision.evidence.get("guardrails")),
            "guardrails": list(decision.evidence.get("guardrails") or []),
            "safety_overrides": dict(decision.safety_overrides),
            "refusal_reason": decision.refusal_reason,
        },
    }
    if decision.status != expected_status:
        raise RuntimeError(f"expected {expected_status}, got {decision.status}: {decision.to_dict()}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("smoke/c3/completion/evidence/planner_branches"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.output_dir / "audits"
    log_path = args.output_dir / "planner_branches.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    try:
        cases = [run_case(name, config, audit_dir) for name, config in branch_cases().items()]
    finally:
        LOGGER.removeHandler(handler)
        handler.close()

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": "../YOLO-Master/.venv/bin/python smoke/c3/completion/tools/generate_planner_branch_evidence.py",
        "model_source": MODEL_CONFIG,
        "cases": cases,
    }
    output = args.output_dir / "planner_branches.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": output.as_posix(), "statuses": [c["decision"]["status"] for c in cases]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
