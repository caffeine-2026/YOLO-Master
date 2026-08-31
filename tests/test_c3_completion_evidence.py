"""Regression tests for the remaining C3 completion evidence contracts."""

import pytest

from smoke.c3.completion.tools.generate_planner_branch_evidence import branch_cases, run_case


@pytest.mark.parametrize("expected_status", ["ACCEPT", "ADAPT", "REFUSE"])
def test_real_yolo11n_planner_branch(expected_status, tmp_path):
    payload = run_case(expected_status, branch_cases()[expected_status], tmp_path / "audits")

    assert payload["decision"]["status"] == expected_status
    assert payload["input"]["adapter_budget"] > 0
    if expected_status == "ACCEPT":
        assert payload["selected_module_count"] > 0
        assert not payload["guardrail_result"]["triggered"]
    elif expected_status == "ADAPT":
        assert payload["selected_module_count"] > 0
        assert "attention_target_policy" in payload["guardrail_result"]["guardrails"]
        assert payload["guardrail_result"]["safety_overrides"]["include_attention"] is False
    else:
        assert payload["selected_module_count"] == 0
        assert "adapter_budget" in payload["guardrail_result"]["guardrails"]
        assert "budget" in payload["guardrail_result"]["refusal_reason"].lower()
