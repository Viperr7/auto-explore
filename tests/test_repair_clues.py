"""Tests for STG-eval repair clues integration in Auto Explore."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from harness.cli import build_parser, config_from_args
from harness.config import ExplorerConfig
from harness.contracts import ExplorationAction, PageState, ScreenCapture
from harness.memory import GraphMemory
from harness.planning import MemoryGuidedPlanner
from harness.repair_clues import load_repair_clues
from harness.state import ExplorationState


def sample_clues_payload(package="com.demo.app"):
    return {
        "format": "stg_eval_repair_clues",
        "schema_version": 1,
        "package": package,
        "stg_root": "/tmp/stg",
        "summary": {"total_targets": 2, "by_priority": {"P0": 1, "P1": 1}},
        "prioritized_activities": [
            {"activity": f"{package}/.SettingsActivity", "score": 12, "unexercised_controls": 2},
            {"activity": f"{package}/.HomeActivity", "score": 4, "one_hop_missing_screens": 1},
        ],
        "targets": [
            {
                "target_id": "closure:no_reaction:n1:0",
                "priority": 0,
                "kind": "closure_control_gap",
                "gap_type": "no_reaction",
                "activity": f"{package}/.SettingsActivity",
                "control": {"resource_id": "btn_privacy", "label": "隐私设置", "class": "TextView"},
                "guidance": "验证控件 隐私设置 的真实行为",
            },
            {
                "target_id": "srm_1hop:CouponActivity",
                "priority": 1,
                "kind": "missing_srm_screen_1hop",
                "gap_type": "missing_screen",
                "activity": f"{package}/.HomeActivity",
                "from_title": "首页",
                "target_title": "优惠券中心",
                "expected_labels": ["领券", "优惠券"],
                "guidance": "在首页寻找进入优惠券中心的入口",
            },
        ],
    }


class RepairCluesTests(unittest.TestCase):
    def test_load_validation_and_priority_lookup(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "repair_clues.json"
            p.write_text(json.dumps(sample_clues_payload(), ensure_ascii=False), encoding="utf-8")
            clues = load_repair_clues(p, "com.demo.app")
            self.assertEqual(12, clues.activity_priority("com.demo.app/.SettingsActivity"))
            self.assertEqual(12, clues.activity_priority("com.demo.app.SettingsActivity"))
            self.assertEqual(4, clues.activity_priority("HomeActivity"))
            self.assertEqual(0, clues.activity_priority("UnknownActivity"))
            with self.assertRaises(ValueError):
                load_repair_clues(p, "com.other.app")

    def test_planner_injects_context_and_prioritizes_clue_matched_action(self):
        with TemporaryDirectory() as tmp:
            clues_path = Path(tmp) / "repair_clues.json"
            clues_path.write_text(json.dumps(sample_clues_payload(), ensure_ascii=False), encoding="utf-8")
            clues = load_repair_clues(clues_path, "com.demo.app")
            cfg = ExplorerConfig(
                app_package="com.demo.app",
                output_dir=str(Path(tmp) / "out"),
                graph_memory=False,
                summary_model="",
            )
            memory = GraphMemory(cfg)
            state = ExplorationState(cfg)
            captured = {}

            class DummyAnalyzer:
                def choose_next_action(self, img, size, candidates, context):
                    captured["candidates"] = candidates
                    captured["context"] = context
                    # Return None so deterministic fallback runs and we verify clue-priority sorting too
                    return None

                def locate_replay_action(self, img, size, step):
                    return {"found": True, "bounds": {"x": 100, "y": 200, "width": 50, "height": 50}}

            planner = MemoryGuidedPlanner(memory, state, DummyAnalyzer(), repair_clues=clues)
            page = PageState(
                page_id="page_1",
                screenshot_path="",
                screenshot_hash="h1",
                activity="com.demo.app/.SettingsActivity",
                page_type="settings",
                elements=[],
                recommended_actions=[],
                planned_actions=[],
                action_sequence=[],
                timestamp="2026-11-19T00:00:00Z",
                app_package="com.demo.app",
                page_description="设置页",
            )
            capture = ScreenCapture(image_b64="ZmFrZQ==", screenshot_hash="h1", screen_size=(1080, 2400))
            a0 = ExplorationAction(
                action_type="click", element_id="e0", element_text="关于我们",
                target_description="点击关于我们", control_key="btn_about",
                bounds={"x": 10, "y": 10, "width": 100, "height": 50},
            )
            a1 = ExplorationAction(
                action_type="click", element_id="e1", element_text="隐私设置",
                target_description="点击隐私设置", control_key="com.demo.app:id/btn_privacy",
                bounds={"x": 10, "y": 100, "width": 100, "height": 50},
            )
            selected = planner.select(page, capture, [a0, a1])
            self.assertIsNotNone(selected)
            # Clue-matched a1 (index 1) must be chosen ahead of a0 (index 0)!
            self.assertEqual(1, selected[0])
            self.assertIn("repair_clues", captured["context"])
            self.assertEqual(12, captured["context"]["repair_clues"]["activity_priority_score"])
            self.assertIn("repair_clue_score", captured["candidates"][1])
            self.assertNotIn("repair_clue_score", captured["candidates"][0])

    def test_cli_parses_repair_clues_and_rejects_mix_with_repair_plan(self):
        parser = build_parser()
        args = parser.parse_args([
            "--device", "127.0.0.1:5555",
            "--package", "com.demo.app",
            "--repair-clues", "/tmp/repair_clues.json",
            "--summary-model", "",
            "--no-semantic-dedup",
        ])
        cfg = config_from_args(args, parser)
        self.assertEqual("/tmp/repair_clues.json", cfg.repair_clues)


if __name__ == "__main__":
    unittest.main()
