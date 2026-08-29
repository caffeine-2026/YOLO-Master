"""CPU-safe tests for C3 Studio data, plots, provenance, and app construction."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ["C3_STUDIO_FORCE_CPU"] = "1"
os.environ["MPLBACKEND"] = "Agg"

STUDIO_ROOT = Path(__file__).resolve().parents[1]
if str(STUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDIO_ROOT))

import matplotlib.pyplot as plt
from utils.load_results import (
    DATASETS,
    METHODS,
    SAMPLE_SIZES,
    canonical_checkpoint,
    comparison_table,
    evidence_catalog,
    p1_all_runs,
    p1_summary,
    p2_all_runs,
    p2_summary,
    planner_data,
    scaling_data,
    stage_status,
    validate_matrix,
)
from utils.plotting import comparison_figure, scaling_figure

from app import create_demo


class StudioSmokeTests(unittest.TestCase):
    def test_p1_csvs_load(self) -> None:
        self.assertEqual(len(p1_all_runs()), 18)
        self.assertEqual(len(p1_summary()), 6)

    def test_p2_csvs_load_and_matrix_passes(self) -> None:
        self.assertEqual(len(p2_all_runs()), 72)
        self.assertEqual(len(p2_summary()), 24)
        self.assertEqual(validate_matrix(), {"expected": 72, "passed": 72, "status": "PASS"})

    def test_stage_statuses(self) -> None:
        status = stage_status()
        self.assertEqual(status["P0"], "PASS")
        self.assertEqual(status["P1"], "PASS")
        self.assertEqual(status["P2"], "PASS")

    def test_all_comparison_cells(self) -> None:
        for dataset in DATASETS:
            for sample_size in SAMPLE_SIZES:
                table = comparison_table(dataset, sample_size)
                self.assertEqual(table["Method"].tolist(), list(METHODS))
                self.assertEqual(len(table), 3)

    def test_comparison_chart(self) -> None:
        figure = comparison_figure("NEU-DET", 100)
        self.assertEqual(len(figure.axes), 4)
        plt.close(figure)

    def test_all_scaling_options(self) -> None:
        metrics = ("mAP50-95", "mAP50", "Accuracy Retention", "Peak GPU Memory", "GPU-hours")
        for dataset in DATASETS:
            for metric in metrics:
                frame = scaling_data(dataset, metric)
                self.assertEqual(len(frame), 12)
                self.assertTrue(frame[["mean", "lower", "upper"]].notna().all().all())
        figure = scaling_figure("DeepPCB", "Accuracy Retention")
        self.assertEqual(len(figure.axes), 1)
        plt.close(figure)

    def test_planner_is_read_from_existing_metrics(self) -> None:
        data = planner_data()
        self.assertEqual(data["status"], "ACCEPT")
        self.assertEqual(data["planner_backend"], "vpeft")
        self.assertEqual(data["actual_backend"], "peft")
        self.assertEqual(data["planned_targets"], 59)
        self.assertEqual(data["applied_targets"], 52)

    def test_final_checkpoints_resolve_and_verify(self) -> None:
        for dataset in DATASETS:
            for method in METHODS:
                checkpoint, row = canonical_checkpoint(dataset, method)
                self.assertTrue(checkpoint.is_file())
                self.assertIn("e100", str(row["run_id"]))

    def test_evidence_paths_are_repository_relative(self) -> None:
        catalog = evidence_catalog()
        self.assertTrue((catalog["Status"] == "Available").all())
        self.assertTrue(catalog["Repository path"].str.startswith("smoke/c3/").all())

    def test_six_tab_app_builds_without_model_load(self) -> None:
        demo = create_demo()
        config = demo.get_config_file()
        labels = {component.get("props", {}).get("label") for component in config.get("components", [])}
        for expected in (
            "Overview",
            "3-Way Comparison",
            "Few-shot Scaling",
            "Live Inference",
            "V-PEFT Planner",
            "Evidence / Reproduction",
        ):
            self.assertIn(expected, labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
