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

    def test_batch_task_spec_supports_repair_clues_and_gym_seed(self):
        from orchestration.models import TaskSpec
        spec = TaskSpec(
            package="com.demo.app",
            options={
                "gym_seed": "/tmp/auto_explore_seed",
                "gym_seed_verification": "adaptive",
                "repair_clues": "/tmp/repair_clues.json",
                "max_steps": 30,
            },
        )
        self.assertEqual("/tmp/repair_clues.json", spec.options["repair_clues"])
        self.assertEqual("/tmp/auto_explore_seed", spec.options["gym_seed"])
        with self.assertRaises(ValueError):
            TaskSpec(package="com.demo.app", options={"gym_seed_verification": "loose"})
        with self.assertRaises(ValueError):
            TaskSpec(package="com.demo.app", options={"gym_seed": "/tmp/seed", "page_exploration": True})

    def test_hierarchy_capture_and_eval_compatible_graph_export(self):
        import base64
        from io import BytesIO
        from PIL import Image
        from harness.lifecycle import ExplorationLifecycle
        from harness.tools import AndroidController

        buf = BytesIO()
        Image.new("RGB", (100, 200), "blue").save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        home_xml = (
            '<hierarchy rotation="0">'
            '<node package="com.demo.app" class="android.widget.FrameLayout" bounds="[0,0][100,200]">'
            '<node package="com.demo.app" class="android.widget.TextView" text="设置" '
            'resource-id="com.demo.app:id/btn_settings" clickable="true" enabled="true" bounds="[10,20][90,80]"/>'
            '<node package="com.demo.app" class="android.widget.TextView" text="关于" '
            'resource-id="com.demo.app:id/btn_about" clickable="true" enabled="true" bounds="[10,100][90,160]"/>'
            '</node></hierarchy>'
        )
        settings_xml = (
            '<hierarchy rotation="0">'
            '<node package="com.demo.app" class="android.widget.FrameLayout" bounds="[0,0][100,200]">'
            '<node package="com.demo.app" class="android.widget.TextView" text="设置中心" '
            'resource-id="com.demo.app:id/settings_title" clickable="false" enabled="true" bounds="[10,10][90,50]"/>'
            '<node package="com.demo.app" class="android.widget.TextView" text="返回" '
            'resource-id="com.demo.app:id/btn_back" clickable="true" enabled="true" bounds="[10,60][90,110]"/>'
            '</node></hierarchy>'
        )

        # 1. AndroidController.capture_hierarchy parses /dev/tty and falls back cleanly
        ctrl = AndroidController(ExplorerConfig(app_package="com.demo.app"))
        ctrl._adb_bytes = lambda *args: b"UI hierchary dumped to: /dev/tty\n" + home_xml.encode("utf-8")
        self.assertEqual(home_xml, ctrl.capture_hierarchy())
        calls = []
        def fallback_bytes(*args):
            calls.append(args)
            if "/dev/tty" in args:
                raise RuntimeError("tty unsupported")
            return settings_xml.encode("utf-8")
        ctrl._adb_bytes = fallback_bytes
        ctrl._adb = lambda *args, **kwargs: ""
        self.assertEqual(settings_xml, ctrl.capture_hierarchy())
        self.assertEqual(2, len(calls))

        # 2. ScreenCapture preserves equality regardless of hierarchy_xml
        c1 = ScreenCapture(png_b64, "h1", (100, 200), hierarchy_xml=home_xml)
        c2 = ScreenCapture(png_b64, "h1", (100, 200))
        self.assertEqual(c1, c2)

        # 3. ExplorationLifecycle attaches captured XML
        class DummyCtrl:
            def capture_screenshot(self):
                return png_b64, "h1", (100, 200)
            def capture_hierarchy(self):
                return home_xml
        life = ExplorationLifecycle(ExplorerConfig(app_package="com.demo.app"), DummyCtrl(), None, None)
        cap = life.capture_screen()
        self.assertEqual(home_xml, cap.hierarchy_xml)

        # 4. ExplorationState saves XML + canonical edge_id + action_details.tap_x/tap_y
        with TemporaryDirectory() as tmp:
            cfg = ExplorerConfig(
                app_package="com.demo.app",
                output_dir=str(Path(tmp) / "run_out"),
                graph_memory=False,
                summary_model="",
                save_screenshots=True,
            )
            state = ExplorationState(cfg)
            cap_home = ScreenCapture(png_b64, "h_home", (100, 200), hierarchy_xml=home_xml)
            buf2 = BytesIO()
            Image.new("RGB", (100, 200), "green").save(buf2, format="PNG")
            cap_set = ScreenCapture(base64.b64encode(buf2.getvalue()).decode("ascii"), "h_set", (100, 200), hierarchy_xml=settings_xml)

            p0_png = state.save_page_screenshot("page_0", cap_home)
            p0_xml = state.save_page_hierarchy("page_0", cap_home)
            p1_png = state.save_page_screenshot("page_1", cap_set)
            p1_xml = state.save_page_hierarchy("page_1", cap_set)
            p0 = PageState("page_0", p0_png, "h_home", "com.demo.app/.HomeActivity", "home",
                           [], [], [], [], 1.0, app_package="com.demo.app", hierarchy_path=p0_xml)
            p1 = PageState("page_1", p1_png, "h_set", "com.demo.app/.SettingsActivity", "settings",
                           [], [], [], [], 2.0, app_package="com.demo.app", hierarchy_path=p1_xml)
            state.register_page(p0, cap_home)
            state.register_page(p1, cap_set)
            state.graph.root_page_id = "page_0"
            act = ExplorationAction(
                action_type="click", target_description="点击设置", element_text="设置",
                bounds={"x": 10, "y": 20, "width": 80, "height": 60},
            )
            b_png, a_png = state.save_transition_screens(cap_home, cap_set)
            b_xml, a_xml = state.save_transition_hierarchies(cap_home, cap_set, b_png, a_png)
            state.record_transition(
                from_page="page_0", to_page="page_1", action=act, success=True, state_changed=True,
                before_screenshot=b_png, after_screenshot=a_png, recovery="continued",
                before_hierarchy_path=b_xml, after_hierarchy_path=a_xml,
                visual_changed=True, destination_verified=True,
            )
            state.save_graph(announce=False)

            saved = json.loads((Path(cfg.output_dir) / "app_graph.json").read_text(encoding="utf-8"))
            self.assertTrue(Path(saved["pages"]["page_0"]["hierarchy_path"]).is_file())
            self.assertTrue(Path(saved["pages"]["page_1"]["hierarchy_path"]).is_file())
            tr = saved["transitions"][0]
            self.assertEqual("edge_00000", tr["edge_id"])
            self.assertEqual(50, tr["action_details"]["tap_x"])
            self.assertEqual(50, tr["action_details"]["tap_y"])
            self.assertTrue(Path(tr["before_hierarchy_path"]).is_file())
            self.assertTrue(Path(tr["after_hierarchy_path"]).is_file())
            self.assertTrue(tr["visual_changed"])

    def test_v3_global_core_template_matching_prompt_and_live_hit_decay(self):
        from harness.context import ExplorationContextBuilder
        from harness.repair_plan import candidates as repair_plan_candidates

        payload = {
            "format": "stg_eval_repair_clues",
            "schema_version": 1,
            "source_mode": "canonical_core_v3",
            "package": "com.demo.app",
            "stg_root": "/tmp/stg",
            "summary": {"total_targets": 2, "by_priority": {"P1": 2}},
            "prioritized_activities": [
                {"activity": "com.demo.app/.WeightCurveActivity", "score": 6, "core_missing_templates": 1},
            ],
            "targets": [
                {
                    "target_id": "core_missing:cft_curve",
                    "priority": 1,
                    "kind": "missing_core_template",
                    "gap_type": "missing_core_template",
                    "canonical_name": "体重曲线",
                    "member_count": 3,
                    "activity": "com.demo.app/.WeightCurveActivity",
                    "target_title": "体重曲线",
                    "target_activities": ["com.demo.app/.WeightCurveActivity"],
                    "target_purpose": "查看体重变化折线趋势",
                    "expected_labels": ["体重曲线", "趋势图"],
                    "guidance": "优先探索核心功能模板「体重曲线」",
                },
                *[
                    {
                        "target_id": f"core_missing:cft_extra_{i}",
                        "priority": 1,
                        "kind": "missing_core_template",
                        "gap_type": "missing_core_template",
                        "canonical_name": f"核心页{i}",
                        "member_count": 1,
                        "activity": "",
                        "target_title": f"核心页{i}",
                        "target_activities": [f"com.demo.app/.Core{i}Activity"],
                        "expected_labels": [f"核心功能{i}"],
                        "guidance": f"探索核心页{i}",
                    }
                    for i in range(12)
                ],
            ],
        }
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "repair_clues.json"
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            clues = load_repair_clues(p, "com.demo.app")

            # 1. Global missing_core_template matches even on MainActivity (where local activity_targets is empty)
            btn_curve = ExplorationAction(
                action_type="click", element_id="e1", element_text="体重曲线",
                target_description="查看体重曲线入口", control_key="btn_curve",
                bounds={"x": 0, "y": 0, "width": 100, "height": 50},
            )
            btn_noise = ExplorationAction(
                action_type="click", element_id="e2", element_text="关于我们",
                target_description="查看关于我们", control_key="btn_about",
                bounds={"x": 0, "y": 60, "width": 100, "height": 50},
            )
            sc_hit, ms_hit = clues.match_action("com.demo.app/.MainActivity", btn_curve)
            sc_noise, _ = clues.match_action("com.demo.app/.MainActivity", btn_noise)
            self.assertGreater(sc_hit, 0)
            self.assertEqual(0, sc_noise)
            self.assertTrue(any("体重曲线" in m for m in ms_hit))

            # 2. Expanded top_missing_screens shows >6 targets (up to 15) and VLM prompt mentions repair_clues
            ctx = clues.prompt_context("com.demo.app/.MainActivity")
            self.assertEqual(13, len(ctx["top_missing_screens"]))
            self.assertEqual(0, ctx["progress"]["hit_in_run"])
            prompt = ExplorationContextBuilder.next_action_prompt([], {"repair_clues": ctx}, 1080, 2400)
            self.assertIn("repair_clues", prompt)
            self.assertIn("repair_clue_score", prompt)

            # 3. Generic MainActivity observation must NOT falsely retire targets; real WeightCurveActivity DOES
            self.assertEqual([], clues.mark_observed("com.demo.app/.MainActivity", page_description="应用主页"))
            hits = clues.mark_observed("com.demo.app/.WeightCurveActivity", page_description="体重趋势页面")
            self.assertEqual(1, len(hits))
            self.assertEqual("core_missing:cft_curve", hits[0]["target_id"])
            self.assertEqual(1, clues.progress()["hit_in_run"])
            self.assertEqual(12, clues.progress()["remaining_missing_targets"])
            # After hit: match_action drops to 0, activity priority decays to 0, and retired target leaves top_missing_screens
            sc_after, _ = clues.match_action("com.demo.app/.MainActivity", btn_curve)
            self.assertEqual(0, sc_after)
            self.assertEqual(0, clues.activity_priority("com.demo.app/.WeightCurveActivity"))
            titles_after = [x["title"] for x in clues.prompt_context("com.demo.app/.MainActivity")["top_missing_screens"]]
            self.assertNotIn("体重曲线", titles_after)

        # 4. repair_plan.candidates drops aux_excluded screens and marks canonical_core
        fake_report = {
            "coverage": {
                "functional_templates": {
                    "canonical": {
                        "core": {"templates": [{"members": ["CoreA"], "covered": False}]},
                        "excluded": [{"screen": "AuxDialog", "reason": "aux_component"}],
                    }
                }
            },
            "findings": [
                {"type": "missing_screen", "screen": "CoreA"},
                {"type": "missing_screen", "screen": "ExtendedB"},
                {"type": "missing_screen", "screen": "AuxDialog"},
            ],
        }
        cands = list(repair_plan_candidates(fake_report, []).values())
        by_scr = {r["screen"]: r for r in cands}
        self.assertNotIn("AuxDialog", by_scr)
        self.assertTrue(by_scr["CoreA"]["canonical_core"])
        self.assertFalse(by_scr["ExtendedB"]["canonical_core"])

    def test_local_ngram_embedding_cli_delays_and_short_chinese_clue_matching(self):
        from harness.cli import build_parser, config_from_args
        from harness.semantic import cosine_similarity
        from harness.tools import VLMAnalyzer

        # 1. local-ngram embedding requires no network and cleanly separates screen templates
        cfg = ExplorerConfig(vlm_api_key="test-key", embedding_model="local-ngram")
        analyzer = VLMAnalyzer(cfg)
        v_login1 = analyzer.embed_description("QQ Music login screen")
        v_login2 = analyzer.embed_description("QQ Music login dialog")
        v_player = analyzer.embed_description("exercise weight tracking home dashboard")
        self.assertEqual(256, len(v_login1))
        self.assertEqual(v_login1, analyzer.embed_description("QQ Music login screen"))
        sim_same = cosine_similarity(v_login1, v_login2)
        sim_diff = cosine_similarity(v_login1, v_player)
        self.assertIsNotNone(sim_same)
        self.assertIsNotNone(sim_diff)
        self.assertGreater(sim_same, 0.55)
        self.assertLess(sim_diff, 0.25)

        # 2. CLI exposes --launch-delay and --vlm-timeout
        parser = build_parser()
        args = parser.parse_args([
            "--package", "com.demo.app", "--device", "127.0.0.1:16384",
            "--launch-delay", "9.5", "--vlm-timeout", "60",
            "--embedding-model", "local-ngram",
        ])
        parsed_cfg = config_from_args(args, parser)
        self.assertEqual(9.5, parsed_cfg.launch_delay)
        self.assertEqual(60.0, parsed_cfg.vlm_timeout)
        self.assertEqual("local-ngram", parsed_cfg.embedding_model)

        # 3. Short 2-char Chinese nav buttons match compound missing core templates, while generics do not
        payload = {
            "format": "stg_eval_repair_clues",
            "schema_version": 1,
            "source_mode": "canonical_core_v3",
            "package": "com.demo.app",
            "stg_root": "/tmp/stg",
            "summary": {"total_targets": 2, "by_priority": {"P1": 2}},
            "prioritized_activities": [],
            "targets": [
                {
                    "target_id": "core_missing:cft_singer",
                    "priority": 1,
                    "kind": "missing_core_template",
                    "canonical_name": "歌手主页",
                    "target_title": "歌手主页",
                    "expected_labels": ["歌手主页", "歌手专辑"],
                    "guidance": "探索歌手主页",
                },
                {
                    "target_id": "core_missing:cft_mall",
                    "priority": 1,
                    "kind": "missing_core_template",
                    "canonical_name": "商城首页",
                    "target_title": "商城首页",
                    "expected_labels": ["商城首页"],
                    "guidance": "探索商城首页",
                },
            ],
        }
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "repair_clues.json"
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            clues = load_repair_clues(p, "com.demo.app")
            btn_singer = ExplorationAction(action_type="click", element_id="e1", element_text="歌手", target_description="Open 歌手", bounds={"x": 0, "y": 0, "width": 50, "height": 50})
            btn_home = ExplorationAction(action_type="click", element_id="e2", element_text="首页", target_description="Open 首页", bounds={"x": 0, "y": 0, "width": 50, "height": 50})
            sc_singer, _ = clues.match_action("com.demo.app/.MainActivity", btn_singer)
            sc_home, _ = clues.match_action("com.demo.app/.MainActivity", btn_home)
            self.assertGreater(sc_singer, 0)
            self.assertEqual(0, sc_home)

    def test_avoid_list_bfs_priority_queue_analyze_injection_and_bottom_tab_survival(self):
        from harness.loop import AndroidExplorer

        payload = {
            "format": "stg_eval_repair_clues",
            "schema_version": 1,
            "source_mode": "canonical_core_v3",
            "package": "com.demo.app",
            "stg_root": "/tmp/stg",
            "summary": {"total_targets": 1, "by_priority": {"P1": 1}},
            "covered_core_templates": [
                {"template_id": "cft_water", "name": "饮水记录", "labels": ["饮水记录", "喝水打卡"],
                 "activities": ["com.demo.app/.WaterActivity"]},
                {"template_id": "cft_diet", "name": "饮食记录", "labels": ["饮食记录", "记一餐"],
                 "activities": ["com.demo.app/.DietActivity"]},
            ],
            "prioritized_activities": [
                {"activity": "com.demo.app/.ShopHubActivity", "score": 10, "core_missing_templates": 1},
            ],
            "targets": [
                {
                    "target_id": "core_missing:cft_mall",
                    "priority": 1,
                    "kind": "missing_core_template",
                    "canonical_name": "商城首页",
                    "target_title": "商城首页",
                    "expected_labels": ["商城首页", "薄荷商城"],
                    "guidance": "探索商城首页",
                },
            ],
        }
        with TemporaryDirectory() as tmp:
            clues_path = Path(tmp) / "repair_clues.json"
            clues_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cfg = ExplorerConfig(
                app_package="com.demo.app",
                output_dir=str(Path(tmp) / "out"),
                repair_clues=str(clues_path),
                graph_memory=False,
                summary_model="",
                max_actions_per_page=3,
                state_exploration=False,
            )
            seen_mem = {}

            class DummyAnalyzer:
                model = "dummy/v1"
                def analyze_screen_with_memory(self, img, size, memory):
                    seen_mem.update(memory)
                    return {"elements": [], "actions": [], "page_type": "home"}

            explorer = AndroidExplorer(cfg, controller=object(), analyzer=DummyAnalyzer())
            clues = explorer.repair_clues
            assert clues is not None

            # 1. Covered-only template gets negative avoid score; missing target gets positive score
            act_water = ExplorationAction(action_type="click", element_id="e0", element_text="饮水记录",
                                          target_description="打开饮水记录", bounds={"x": 10, "y": 500, "width": 100, "height": 50})
            act_diet = ExplorationAction(action_type="click", element_id="e1", element_text="饮食记录",
                                         target_description="打开饮食记录", bounds={"x": 10, "y": 600, "width": 100, "height": 50})
            act_mall = ExplorationAction(action_type="click", element_id="e2", element_text="商城",
                                         target_description="打开商城", bounds={"x": 300, "y": 2200, "width": 100, "height": 80}, element_type="tab")
            act_mine = ExplorationAction(action_type="click", element_id="e3", element_text="我的",
                                         target_description="打开我的", bounds={"x": 900, "y": 2200, "width": 100, "height": 80}, element_type="tab")
            act_other = ExplorationAction(action_type="click", element_id="e4", element_text="通知",
                                          target_description="打开通知", bounds={"x": 900, "y": 100, "width": 80, "height": 80})

            sc_water, ms_water = clues.match_action("com.demo.app/.MainActivity", act_water)
            sc_mall, _ = clues.match_action("com.demo.app/.MainActivity", act_mall)
            self.assertLess(sc_water, 0)
            self.assertTrue(any("AVOID" in m for m in ms_water))
            self.assertGreater(sc_mall, 0)

            # 2. _analyze_screen injects repair_clues + repair_clues_guidance even when graph_memory=False
            cap = ScreenCapture(image_b64="ZmFrZQ==", screenshot_hash="h0", screen_size=(1080, 2400))
            explorer._analyze_screen(cap, [], "com.demo.app/.MainActivity")
            self.assertIn("repair_clues_guidance", seen_mem)
            self.assertIn("商城首页", seen_mem["repair_clues_guidance"])
            self.assertIn("饮水记录", seen_mem["repair_clues_guidance"])

            # 3. _select_actions preserves missing-clue + bottom tabs ahead of avoid-list under budget=3
            p_home = PageState("page_0", "", "h0", "com.demo.app/.MainActivity", "home",
                               [], [], [act_water, act_diet, act_other, act_mine, act_mall], [], 1.0, app_package="com.demo.app")
            explorer.state.register_page(p_home, cap)
            chosen = explorer._select_actions(p_home)
            chosen_texts = [a.element_text for a in chosen]
            self.assertEqual(3, len(chosen))
            self.assertIn("商城", chosen_texts)
            self.assertIn("我的", chosen_texts)
            self.assertNotIn("饮水记录", chosen_texts)
            self.assertNotIn("饮食记录", chosen_texts)

            # 4. BFS priority queue pops high-value missing-clue page ahead of earlier FIFO covered-only page
            p_cov = PageState("page_1", "", "h1", "com.demo.app/.WaterActivity", "detail",
                              [], [], [act_other], [], 2.0, app_package="com.demo.app")
            p_hub = PageState("page_2", "", "h2", "com.demo.app/.ShopHubActivity", "list",
                              [], [], [act_mall], [], 3.0, app_package="com.demo.app")
            explorer.state.register_page(p_cov, cap)
            explorer.state.register_page(p_hub, cap)
            explorer._pending_pages.clear()
            explorer._pending_pages.extend(["page_1", "page_2"])
            self.assertEqual("page_2", explorer._pop_next_pending_page())
            self.assertEqual("page_1", explorer._pop_next_pending_page())


if __name__ == "__main__":
    unittest.main()
