"""Offline termination checks for the real state scheduler, without device/model calls."""

import base64
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from harness.config import ExplorerConfig
from harness.contracts import PageState, ScreenCapture
from harness.state import ExplorationState
from harness.state_loop import StateExplorer


PACKAGE = "com.example.app"
ACTIVITY = PACKAGE + "/.Main"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (100, 100), "blue").save(buffer, format="PNG")
    data = buffer.getvalue()
    shot = ScreenCapture(base64.b64encode(data).decode(), sha256(data).hexdigest(), (100, 100))
    config = ExplorerConfig(app_package=PACKAGE, output_dir=str(tmp_path / "auto"),
                            save_screenshots=False)
    state = ExplorationState(config)
    root = PageState("root", "", shot.screenshot_hash, ACTIVITY, "list", [], [], [], [], 0)
    state.register_page(root, shot)
    failure = {"error": ""}

    def profile(self, page, capture, path, **options):
        guard = {"page_id": page.page_id, "layout_id": "root-layout", "state": {}, "values": {},
                 "context": {}, "surface": page.page_id, "module": page.page_id,
                 "activity": ACTIVITY, "conflicts": [], "analysis_error": failure["error"]}
        return guard, [], [], {}

    monkeypatch.setattr(StateExplorer, "profile", profile)
    explorer = SimpleNamespace(
        state=state, graph=state.graph, config=config,
        controller=SimpleNamespace(get_current_activity=lambda: ACTIVITY),
        _is_target_activity=lambda activity: activity == ACTIVITY,
        _step_budget_exhausted=lambda: False,
        capability_service=SimpleNamespace(update=lambda **options: None),
    )
    return StateExplorer(explorer), root, shot, failure


def test_analysis_failure_without_work_is_blocked(runner):
    driver, root, shot, failure = runner
    failure["error"] = "screen_analysis_unavailable"

    driver.run(root, shot)

    assert driver.graph.work_items == {}
    assert root.exploration_status == "blocked"
    assert driver.graph.termination_reason == "blocked"


def test_unanalyzed_page_without_work_is_blocked(runner):
    driver, root, shot, _ = runner
    other = PageState("unprofiled", "", shot.screenshot_hash, ACTIVITY, "list", [], [], [], [], 0)
    driver.state.register_page(other, shot)

    driver.run(root, shot)

    assert {work.page_id for work in driver.graph.work_items.values()} == {"root"}
    assert other.exploration_status == "blocked"
    assert driver.graph.termination_reason == "blocked"


def test_successfully_analyzed_page_with_no_actions_is_exhausted(runner):
    driver, root, shot, _ = runner

    driver.run(root, shot)

    assert {work.status for work in driver.graph.work_items.values()} == {"complete"}
    assert root.exploration_status == "complete"
    assert driver.graph.termination_reason == "frontier_exhausted"


def test_budget_limit_takes_precedence_over_unanalyzed_page(runner):
    driver, root, shot, failure = runner
    failure["error"] = "screen_analysis_unavailable"
    driver.graph.termination_reason = "budget_limited"

    driver.run(root, shot)

    assert driver.graph.termination_reason == "budget_limited"


def test_app_return_failure_takes_precedence_over_budget_limit(runner):
    driver, root, shot, failure = runner
    failure["error"] = "screen_analysis_unavailable"
    driver.graph.termination_reason = "budget_limited"
    driver.app_return_failed = True

    driver.run(root, shot)

    assert driver.graph.termination_reason == "blocked"


@pytest.mark.parametrize("status, expected", [
    ("planning_blocked", "blocked"),
    ("recovery_failed", "blocked"),
    ("state_unresolved", "blocked"),
    ("budget_limited", "budget_limited"),
])
def test_existing_work_status_keeps_termination_semantics(runner, monkeypatch, status, expected):
    driver, root, shot, _ = runner

    def stop(burst):
        work, = driver.graph.work_items.values()
        work.status = status
        return None

    monkeypatch.setattr(driver, "pick", stop)
    driver.run(root, shot)

    assert driver.graph.termination_reason == expected
