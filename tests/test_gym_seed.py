import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from harness import cli
from harness.contracts import ExplorationAction, ScreenCapture
from harness.grounding import with_source


PACKAGE = "com.example.app"
ACTIVITY = PACKAGE + "/.Main"


@pytest.fixture
def seed(tmp_path):
    root = tmp_path / "auto_explore_seed"
    (root / "assets").mkdir(parents=True)
    (root / "graphs").mkdir()
    for name, color in (("root", "blue"), ("detail", "green")):
        Image.new("RGB", (100, 100), color).save(root / "assets" / (name + ".png"))
    graph = {
        "app_package": PACKAGE,
        "pages": {name: {"activity": ACTIVITY, "screenshot_path": f"assets/{name}.png"}
                  for name in ("root", "detail")},
        "task_runs": [{"task_id": "t1", "reset_policy": "restart", "start_page_id": "root",
                       "edge_ids": ["edge_0"]}],
        "transitions": [{"edge_id": "edge_0", "from_page": "root", "to_page": "detail",
                         "success": True, "action_details": {"action_type": "click", "tap_x": 40, "tap_y": 40}}],
    }
    graph_path = root / "graphs/graph_000.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    files = {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file()}
    (root / "manifest.json").write_text(json.dumps({
        "format": "gym_explore_seed", "schema_version": 1, "app_package": PACKAGE,
        "graphs": [{"path": "graphs/graph_000.json", "sha256": files["graphs/graph_000.json"]}],
        "files": files,
    }), encoding="utf-8")
    return root


def test_cli_accepts_published_directory_and_loads_route(seed):
    parser = cli.build_parser()
    args = parser.parse_args(["--device", "127.0.0.1:5555", "--package", PACKAGE, "--gym-seed", str(seed.parent), "--summary-model", ""])
    config = cli.config_from_args(args, parser)
    from harness.gym_seed import load_seed_routes
    routes = load_seed_routes(config.gym_seed, config.app_package, max_depth=3)
    assert [(r[0].x, r[0].y) for r in routes] == [(40, 40)]


def test_seed_rejects_wrong_package_and_tampered_graph(seed):
    from harness.gym_seed import load_seed_routes
    with pytest.raises(ValueError, match="package"):
        load_seed_routes(seed, "wrong.app", max_depth=3)
    (seed / "graphs/graph_000.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_seed_routes(seed, PACKAGE, max_depth=3)


def test_reads_seed_written_by_research_exporter(seed, tmp_path, monkeypatch):
    research = Path(__file__).resolve().parents[2] / "research-explore"
    if not research.is_dir():
        pytest.skip("Research Explore checkout is not adjacent")
    monkeypatch.syspath_prepend(str(research))
    from gym_explore.handoff import export_handoff
    from harness.gym_seed import load_seed_routes
    source = tmp_path / "app_graph.json"
    graph = json.loads((seed / "graphs/graph_000.json").read_text())
    for page in graph["pages"].values():
        page["screenshot_path"] = str(seed / page["screenshot_path"])
    source.write_text(json.dumps(graph), encoding="utf-8")
    exported = tmp_path / "actual-export"
    export_handoff(source, exported)
    route, = load_seed_routes(exported, PACKAGE, max_depth=3)
    assert (route[0].x, route[0].y, route[0].edge) == (40, 40, "edge_0")


def capture(data):
    return ScreenCapture(base64.b64encode(data).decode(), sha256(data).hexdigest(), (100, 100))


@pytest.mark.parametrize("mode", ["match", "chrome_drift", "source_mismatch", "target_mismatch", "unsafe"])
def test_seed_bootstrap_registers_only_live_verified_destinations(seed, mode):
    from harness.gym_seed import bootstrap, load_seed_routes
    from harness.state_loop import Position

    images = [capture((seed / "assets" / (name + ".png")).read_bytes()) for name in ("root", "detail")]
    buffer = BytesIO()
    Image.new("RGB", (100, 100), "red").save(buffer, format="PNG")
    wrong = capture(buffer.getvalue())
    chrome = Image.open(BytesIO(base64.b64decode(images[0].image_b64))).convert("RGB")
    chrome.putpixel((50, 3), (255, 0, 0))
    chrome_buffer = BytesIO()
    chrome.save(chrome_buffer, format="PNG")
    changed_chrome = capture(chrome_buffer.getvalue())
    queue, edges, logs, executions = [], [], [], []
    graph = SimpleNamespace(completed_steps=0)
    root = SimpleNamespace(page_id="root")
    detail = SimpleNamespace(page_id="detail")
    live = images[0]
    shot_count = 0

    def shot():
        nonlocal shot_count
        shot_count += 1
        return changed_chrome if mode == "chrome_drift" and shot_count > 1 and live == images[0] else live

    def restart():
        nonlocal live
        live = wrong if mode == "source_mismatch" else images[0]

    def execute(action, size):
        nonlocal live
        executions.append(action)
        live = wrong if mode == "target_mismatch" else images[1]
        return True

    def observe(page, shot, path, **options):
        if options.get("finish_step"):
            from dataclasses import replace
            path = [*path[:-1], replace(path[-1], target_guard={"page_id": page.page_id})]
        action = with_source(ExplorationAction("click", "detail", bounds={"x": 30, "y": 30, "width": 20, "height": 20},
                             intent="mutation" if mode == "unsafe" else "navigate", replay_safe=True), shot, "test")
        work = SimpleNamespace(work_id=page.page_id, planned_actions=[action])
        if page is detail:
            queue.append(work.work_id)
        return Position(page, SimpleNamespace(state_id=page.page_id), work, shot, {"layout_id": action.source_layout_id}, path)

    def record_transition(**edge):
        edges.append(edge)
        return {"observation_id": "obs1"}

    state = SimpleNamespace(append_observation=logs.append, record_transition=record_transition)
    controller = SimpleNamespace(restart_app=restart, execute_action=execute, get_current_activity=lambda: ACTIVITY)
    explorer = SimpleNamespace(controller=controller, _capture_screen=shot,
                               _is_target_activity=lambda _: True,
                               _capture_or_get_page=lambda shot, path, **_: (detail if shot == images[1] else root, False))
    runner = SimpleNamespace(explorer=explorer, state=state, graph=graph, observe=observe,
                             registry=SimpleNamespace(record_result=lambda *_: None),
                             config=SimpleNamespace(max_steps=4, save_screenshots=False))
    runner.current = observe(root, images[0], [])
    bootstrap(runner, load_seed_routes(seed, PACKAGE, max_depth=3))
    assert bool(queue) == (mode in {"match", "chrome_drift"})
    assert bool(executions) == (mode in {"match", "chrome_drift", "target_mismatch"})
    assert graph.completed_steps == len(executions)
    assert any(e["destination_verified"] for e in edges) == (mode in {"match", "chrome_drift"})
    if mode in {"match", "chrome_drift"}:
        assert runner.current.work.work_id == "detail"
        assert runner.current.path[-1].replay_safe and runner.current.path[-1].arrival_verified
        assert edges[0]["to_work_id"] == "detail"


def test_scheduler_expands_new_edge_from_seed_work(seed, tmp_path, monkeypatch):
    from harness.config import ExplorerConfig
    from harness.contracts import PageState
    from harness.gym_seed import load_seed_routes
    from harness.grounding import layout_id
    from harness.state import ExplorationState
    from harness.state_loop import StateExplorer
    from harness.knowledge import valid_observation

    Image.new("RGB", (100, 100), "red").save(tmp_path / "next.png")
    images = {name: capture(path.read_bytes()) for name, path in
              (("root", seed / "assets/root.png"), ("detail", seed / "assets/detail.png"),
               ("next", tmp_path / "next.png"))}
    config = ExplorerConfig(app_package=PACKAGE, output_dir=str(tmp_path / "auto"),
                            max_steps=2, save_screenshots=False)
    state = ExplorationState(config)
    live = "root"
    calls = []

    def restart():
        nonlocal live
        live = "root"

    def execute(action, size):
        nonlocal live
        calls.append((live, action.target_description))
        live = action.target_description
        return True

    def register(shot, path, **_):
        name = next(n for n, image in images.items() if image == shot)
        if name in state.graph.pages:
            return state.graph.pages[name], False
        page = PageState(name, "", shot.screenshot_hash, ACTIVITY, "list", [], [], [], list(path), 0)
        state.register_page(page, shot)
        return page, True

    def profile(self, page, shot, path, **_):
        guard = {"page_id": page.page_id, "layout_id": layout_id(shot), "state": {}, "values": {},
                 "context": {}, "surface": page.page_id, "module": page.page_id, "activity": ACTIVITY,
                 "conflicts": []}
        target = {"root": "detail", "detail": "next"}.get(page.page_id)
        actions = [with_source(ExplorationAction("click", target, bounds={"x": 30, "y": 30, "width": 20, "height": 20},
                   intent="navigate", replay_safe=True), shot, "fixture")] if target else []
        return guard, actions, [], {}

    monkeypatch.setattr(StateExplorer, "profile", profile)
    controller = SimpleNamespace(restart_app=restart, execute_action=execute, get_current_activity=lambda: ACTIVITY)
    explorer = SimpleNamespace(
        controller=controller, state=state, graph=state.graph, config=config,
        _capture_screen=lambda: images[live], _capture_or_get_page=register,
        _is_target_activity=lambda _: True, _has_executable_action=lambda *_: True,
        _step_budget_exhausted=lambda: state.graph.completed_steps >= config.max_steps,
        memory_planner=SimpleNamespace(select=lambda page, shot, actions, **_: (0, actions[0])),
        capability_service=SimpleNamespace(update=lambda **_: None),
        coverage_checkpoint=SimpleNamespace(should_run=lambda _: False),
    )
    root, _ = register(images["root"], [])
    state.graph.root_page_id = root.page_id
    StateExplorer(explorer).run(root, images["root"], seed_routes=load_seed_routes(seed, PACKAGE, max_depth=3))
    assert calls == [("root", "detail"), ("detail", "next")]
    assert [(e.from_page, e.to_page, e.purpose) for e in state.graph.transitions] == [
        ("root", "detail", "gym_seed"), ("detail", "next", "exploration")]
    assert any(o.get("event") == "gym_seed" and valid_observation(o) for o in state.graph.observations)


def update_seed(seed, change):
    path = seed / "graphs/graph_000.json"
    graph = json.loads(path.read_text(encoding="utf-8"))
    change(graph)
    path.write_text(json.dumps(graph), encoding="utf-8")
    manifest_path = seed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["graphs/graph_000.json"] = sha256(path.read_bytes()).hexdigest()
    manifest["graphs"][0]["sha256"] = manifest["files"]["graphs/graph_000.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_toolbar_click_is_not_discarded_by_image_crop(seed):
    from harness.gym_seed import load_seed_routes
    update_seed(seed, lambda g: g["transitions"][0]["action_details"].update(tap_y=5))
    routes = load_seed_routes(seed, PACKAGE, max_depth=3)
    assert len(routes) == 1 and routes[0][0].y == 5


@pytest.mark.parametrize("x,y", [(-1, 40), (100, 40), (40, -1), (40, 100)])
def test_seed_still_rejects_out_of_screen_coordinates(seed, x, y):
    from harness.gym_seed import load_seed_routes
    update_seed(seed, lambda g: g["transitions"][0]["action_details"].update(tap_x=x, tap_y=y))
    assert not load_seed_routes(seed, PACKAGE, max_depth=3)


def test_loader_explains_truncated_paths_without_promoting_history(seed):
    from harness.gym_seed import load_seed_routes
    update_seed(seed, lambda g: g["transitions"][0]["action_details"].update(action_type="input"))
    routes = load_seed_routes(seed, PACKAGE, max_depth=3)
    assert routes == []
    assert routes.report["candidate_routes"] == 0
    assert routes.report["historical_pages"] == 2
    assert routes.report["rejections"]["unsupported_action"] == 1


@pytest.mark.parametrize("mode,expected", [
    ("accept", True), ("reject", False), ("unavailable", False),
    ("invalid", False), ("ambiguous", False), ("unsafe", False), ("persistent", False),
    ("invalid_bounds", False), ("missing_hint", False),
    ("stale", False), ("stale_after_judge", False), ("exact", False), ("wrong_activity", False),
    ("target_drift", True), ("target_reject", False),
])
def test_adaptive_seed_requires_verified_arrival_and_fresh_unique_navigation(seed, mode, expected):
    from dataclasses import replace
    from harness.gym_seed import bootstrap, load_seed_routes
    from harness.state_loop import Position

    if mode != "missing_hint":
        update_seed(seed, lambda g: g["transitions"][0]["action_details"].update(element_text="detail"))
    historical = capture((seed / "assets/root.png").read_bytes())
    target = capture((seed / "assets/detail.png").read_bytes())
    image = Image.open(BytesIO(base64.b64decode(historical.image_b64))).convert("RGB")
    image.putpixel((50, 50), (0, 0, 254))
    data = BytesIO()
    image.save(data, format="PNG")
    changed = capture(data.getvalue())
    target_image = Image.open(BytesIO(base64.b64decode(target.image_b64))).convert("RGB")
    target_image.putpixel((50, 50), (0, 127, 0))
    target_data = BytesIO()
    target_image.save(target_data, format="PNG")
    changed_target = capture(target_data.getvalue())
    executions, edges, logs, judgements = [], [], [], []
    live = changed
    captures = 0

    def restart():
        nonlocal live
        live = changed

    def shot():
        nonlocal captures
        captures += 1
        if (mode == "stale" and captures == 2) or (mode == "stale_after_judge" and captures == 3):
            return historical
        return live

    def execute(action, size):
        nonlocal live
        executions.append(action)
        live = changed_target if mode in {"target_drift", "target_reject"} else target
        return True

    def judge(*args, **kwargs):
        judgements.append(kwargs)
        if mode == "unavailable":
            raise RuntimeError("offline judge")
        if mode == "invalid":
            return {"status": "success", "matched": "true"}
        matched = mode != "reject" and not (mode == "target_reject" and kwargs.get("action"))
        return {"status": "success", "matched": bool(matched), "reason": "fixture verdict"}

    def observe(page, screen, path, **options):
        if options.get("finish_step"):
            path = [*path[:-1], replace(path[-1], target_guard={"page_id": page.page_id})]
        action = with_source(ExplorationAction("click", "detail", element_text="detail",
            bounds={"x": 60, "y": 60, "width": 100 if mode == "invalid_bounds" else 20, "height": 20},
            intent="mutation" if mode == "unsafe" else "navigate", replay_safe=True,
            effect_kind="persistent" if mode == "persistent" else "unknown"), screen, "fixture")
        actions = [action, replace(action, target_description="duplicate")] if mode == "ambiguous" else [action]
        return Position(page, SimpleNamespace(state_id=page.page_id),
                        SimpleNamespace(work_id=page.page_id, planned_actions=actions), screen,
                        {"layout_id": action.source_layout_id}, path)

    root, detail = SimpleNamespace(page_id="root"), SimpleNamespace(page_id="detail")
    graph = SimpleNamespace(completed_steps=0)
    state = SimpleNamespace(append_observation=logs.append,
                            record_transition=lambda **e: (edges.append(e) or {"observation_id": "o1"}))
    controller = SimpleNamespace(restart_app=restart, execute_action=execute,
        get_current_activity=lambda: "outside.app/.Main" if mode == "wrong_activity" else ACTIVITY)
    explorer = SimpleNamespace(controller=controller, analyzer=SimpleNamespace(verify_replay_state=judge),
        _capture_screen=shot, _is_target_activity=lambda a: a.startswith(PACKAGE + "/"),
        _capture_or_get_page=lambda screen, path, **_: (detail if screen in (target, changed_target) else root, False))
    runner = SimpleNamespace(explorer=explorer, state=state, graph=graph, observe=observe,
        registry=SimpleNamespace(record_result=lambda *_: None),
        config=SimpleNamespace(max_steps=4, save_screenshots=False,
                               gym_seed_verification="exact" if mode == "exact" else "adaptive"))
    runner.current = observe(root, changed, [])
    bootstrap(runner, load_seed_routes(seed, PACKAGE, max_depth=3))
    assert bool(executions) == (expected or mode == "target_reject")
    assert bool(edges and edges[0]["destination_verified"]) == expected
    if expected:
        assert executions[0].bounds["x"] == 60  # Never click the historical x=40 after drift.
        assert executions[0].grounding_method == "gym_seed_current_control"
        assert graph.seed_handoff["verified_arrivals"] == 1
    if mode in {"exact", "wrong_activity", "missing_hint"}:
        assert not judgements


def test_seed_handoff_diagnostics_are_saved_separately_from_coverage(seed, tmp_path):
    from harness.config import ExplorerConfig
    from harness.gym_seed import load_seed_routes
    from harness.state import ExplorationState
    state = ExplorationState(ExplorerConfig(app_package=PACKAGE, output_dir=str(tmp_path / "auto")))
    state.graph.seed_handoff = {**load_seed_routes(seed, PACKAGE, max_depth=3).report, "verified_arrivals": 0}
    state.save_graph(announce=False)
    saved = json.loads((tmp_path / "auto/app_graph.json").read_text(encoding="utf-8"))
    assert saved["seed_handoff"]["historical_pages"] == 2
    assert saved["seed_handoff"]["verified_arrivals"] == 0
    assert saved["pages"] == {} and saved["functional_coverage"] == {}


def test_control_hint_uses_in_app_enabled_tapped_control():
    from harness.gym_seed import _control_hint
    xml = b'''<hierarchy><node package="com.example.app" bounds="[0,0][100,100]">
      <node package="other.app" clickable="true" text="foreign" bounds="[0,0][20,20]"/>
      <node package="com.example.app" clickable="true" enabled="false" text="disabled" bounds="[0,0][20,20]"/>
      <node package="com.example.app" clickable="true" content-desc="Sort" resource-id="com.example.app:id/sort" bounds="[0,0][20,20]"/>
    </node></hierarchy>'''
    assert _control_hint(xml, 10, 10, PACKAGE) == ("Sort", "com.example.app:id/sort")
    assert _control_hint(b"invalid", 10, 10, PACKAGE) == ("", "")


def test_seed_budget_leaves_an_action_for_normal_exploration(seed):
    from harness.gym_seed import bootstrap, load_seed_routes
    runner = SimpleNamespace(explorer=SimpleNamespace(controller=SimpleNamespace()),
        graph=SimpleNamespace(completed_steps=0), current=object(), config=SimpleNamespace(max_steps=1),
        state=SimpleNamespace(append_observation=lambda _: None))
    bootstrap(runner, load_seed_routes(seed, PACKAGE, max_depth=3))
    assert runner.graph.seed_handoff["attempted_routes"] == 0
    assert runner.graph.completed_steps == 0


def test_toolbar_verification_does_not_ignore_changed_cropped_pixels(seed):
    from harness.gym_seed import _verify_screen, load_seed_routes
    update_seed(seed, lambda g: g["transitions"][0]["action_details"].update(element_text="detail"))
    step = load_seed_routes(seed, PACKAGE, max_depth=3)[0][0]
    image = Image.open(BytesIO(base64.b64decode(step.source.image_b64))).convert("RGB")
    image.putpixel((50, 3), (255, 0, 0))
    data = BytesIO()
    image.save(data, format="PNG")
    calls = []
    runner = SimpleNamespace(config=SimpleNamespace(gym_seed_verification="adaptive"),
        explorer=SimpleNamespace(controller=SimpleNamespace(get_current_activity=lambda: ACTIVITY),
            analyzer=SimpleNamespace(verify_replay_state=lambda *a, **k:
                (calls.append(k) or {"status": "success", "matched": False}))),
        state=SimpleNamespace(append_observation=lambda _: None))
    assert not _verify_screen(runner, capture(data.getvalue()), step.source, step, {}, full=True)[0]
    assert len(calls) == 1


@pytest.mark.parametrize("changed_y,tap_y,expected", [(3, 40, True), (50, 40, False), (3, 5, False)])
def test_live_click_freshness_preserves_chrome_tolerance_only_away_from_toolbar(seed, changed_y, tap_y, expected):
    from harness.gym_seed import _same_live_screen
    before = capture((seed / "assets/root.png").read_bytes())
    image = Image.open(BytesIO(base64.b64decode(before.image_b64))).convert("RGB")
    image.putpixel((50, changed_y), (255, 0, 0))
    data = BytesIO()
    image.save(data, format="PNG")
    assert _same_live_screen(before, capture(data.getvalue()), tap_y) is expected


def test_toolbar_route_dedup_keeps_distinct_full_images_and_control_identities(seed):
    from copy import deepcopy
    from harness.gym_seed import load_seed_routes
    image = Image.open(seed / "assets/root.png").convert("RGB")
    image.putpixel((40, 3), (255, 0, 0))
    alternative = seed / "assets/root_alternative.png"
    image.save(alternative)
    manifest_path = seed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["assets/root_alternative.png"] = sha256(alternative.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def change(graph):
        graph["transitions"][0]["action_details"].update(tap_y=5, element_text="Sort")
        graph["pages"]["other_root"] = {"activity": ACTIVITY, "screenshot_path": "assets/root_alternative.png"}
        edge = deepcopy(graph["transitions"][0])
        edge.update(edge_id="edge_1", from_page="other_root")
        edge["action_details"]["element_text"] = "More options"
        graph["transitions"].append(edge)
        graph["task_runs"].append({"task_id": "t2", "reset_policy": "restart",
                                   "start_page_id": "other_root", "edge_ids": ["edge_1"]})

    update_seed(seed, change)
    routes = load_seed_routes(seed, PACKAGE, max_depth=3)
    assert len(routes) == 2
    assert routes[0][0].source.digest == routes[1][0].source.digest
    assert routes[0][0].source.full_digest != routes[1][0].source.full_digest


def test_seed_verification_cache_keeps_control_identity(seed):
    from dataclasses import replace
    from harness.gym_seed import _verify_screen, load_seed_routes
    step = load_seed_routes(seed, PACKAGE, max_depth=3)[0][0]
    image = Image.open(BytesIO(base64.b64decode(step.source.image_b64))).convert("RGB")
    image.putpixel((50, 50), (255, 0, 0))
    data = BytesIO()
    image.save(data, format="PNG")
    calls = []

    def judge(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "success", "matched": len(calls) == 1}

    runner = SimpleNamespace(config=SimpleNamespace(gym_seed_verification="adaptive"),
        explorer=SimpleNamespace(controller=SimpleNamespace(get_current_activity=lambda: ACTIVITY),
                                 analyzer=SimpleNamespace(verify_replay_state=judge)),
        state=SimpleNamespace(append_observation=lambda _: None))
    cache = {}
    first = replace(step, resource_id="app:id/first")
    second = replace(step, resource_id="app:id/second")
    shot = capture(data.getvalue())
    assert _verify_screen(runner, shot, first.source, first, cache)[0]
    assert not _verify_screen(runner, shot, second.source, second, cache)[0]
    assert len(calls) == 2


def test_seed_reports_entry_that_cannot_be_registered(seed):
    from harness.gym_seed import bootstrap, load_seed_routes
    shot = capture((seed / "assets/root.png").read_bytes())
    logs = []
    runner = SimpleNamespace(config=SimpleNamespace(max_steps=4), graph=SimpleNamespace(completed_steps=0),
        current=None, state=SimpleNamespace(append_observation=logs.append),
        explorer=SimpleNamespace(controller=SimpleNamespace(restart_app=lambda: None,
            get_current_activity=lambda: ACTIVITY), _capture_screen=lambda: shot,
            _capture_or_get_page=lambda *_: (None, False)))
    bootstrap(runner, load_seed_routes(seed, PACKAGE, max_depth=3))
    assert runner.graph.seed_handoff["runtime_rejections"]["entry_not_registered"] == 1


def test_seed_counts_action_attempt_even_if_after_screenshot_fails(seed):
    from harness.gym_seed import bootstrap, load_seed_routes
    from harness.state_loop import Position
    shot = capture((seed / "assets/root.png").read_bytes())
    action = with_source(ExplorationAction("click", "detail", intent="navigate", replay_safe=True,
        bounds={"x": 30, "y": 30, "width": 20, "height": 20}), shot, "fixture")
    work = SimpleNamespace(work_id="root", planned_actions=[action])
    page = SimpleNamespace(page_id="root")
    position = Position(page, SimpleNamespace(state_id="root"), work, shot, {}, [])
    captures = 0

    def screen():
        nonlocal captures
        captures += 1
        if captures == 3:
            raise RuntimeError("after screenshot unavailable")
        return shot

    runner = SimpleNamespace(config=SimpleNamespace(max_steps=4), graph=SimpleNamespace(completed_steps=0),
        current=position, observe=lambda *_: position, state=SimpleNamespace(append_observation=lambda _: None),
        explorer=SimpleNamespace(controller=SimpleNamespace(restart_app=lambda: None,
            execute_action=lambda *_: True, get_current_activity=lambda: ACTIVITY), _capture_screen=screen,
            _capture_or_get_page=lambda *_: (page, False)))
    with pytest.raises(RuntimeError, match="after screenshot"):
        bootstrap(runner, load_seed_routes(seed, PACKAGE, max_depth=3))
    assert runner.graph.completed_steps == 1
    assert runner.graph.seed_handoff["executed_actions"] == 1


def test_cli_exposes_seed_verification_modes():
    parser = cli.build_parser()
    args = parser.parse_args(["--device", "127.0.0.1:5555", "--package", PACKAGE, "--summary-model", "", "--gym-seed-verification", "exact"])
    assert cli.config_from_args(args, parser).gym_seed_verification == "exact"


def test_seed_folds_leading_launch_wait_including_external_splash(seed):
    from harness.gym_seed import load_seed_routes

    Image.new("RGB", (100, 100), "black").save(seed / "assets/launcher.png")
    manifest_path = seed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["assets/launcher.png"] = sha256((seed / "assets/launcher.png").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def change(graph):
        graph["pages"]["launcher"] = {
            "activity": "com.android.launcher/.Launcher",
            "screenshot_path": "assets/launcher.png",
        }
        graph["transitions"].insert(0, {
            "edge_id": "edge_wait",
            "from_page": "launcher",
            "to_page": "root",
            "success": True,
            "action_details": {"action_type": "wait"},
        })
        graph["task_runs"][0]["start_page_id"] = "launcher"
        graph["task_runs"][0]["edge_ids"] = ["edge_wait", "edge_0"]

    update_seed(seed, change)
    routes = load_seed_routes(seed, PACKAGE, max_depth=1)
    assert len(routes) == 1
    assert len(routes[0]) == 1
    assert routes[0][0].edge == "edge_0"
    assert routes[0][0].source.activity == f"{PACKAGE}/{PACKAGE}.Main"
    assert routes.report["folded_wait_edges"] == 1


def test_seed_folds_post_click_settle_wait_and_preserves_continuity(seed):
    from harness.gym_seed import load_seed_routes

    for name, color in (("loading", "yellow"), ("sub", "purple")):
        path = seed / "assets" / f"{name}.png"
        Image.new("RGB", (100, 100), color).save(path)
    manifest_path = seed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("loading", "sub"):
        p = seed / "assets" / f"{name}.png"
        manifest["files"][f"assets/{name}.png"] = sha256(p.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def change(graph):
        graph["pages"]["loading"] = {"activity": ACTIVITY, "screenshot_path": "assets/loading.png"}
        graph["pages"]["sub"] = {"activity": PACKAGE + "/.Sub", "screenshot_path": "assets/sub.png"}
        graph["transitions"] = [
            {"edge_id": "edge_0", "from_page": "root", "to_page": "loading", "success": True,
             "action_details": {"action_type": "click", "tap_x": 40, "tap_y": 40}},
            {"edge_id": "edge_wait", "from_page": "loading", "to_page": "detail", "success": True,
             "action_details": {"action_type": "wait"}},
            {"edge_id": "edge_1", "from_page": "detail", "to_page": "sub", "success": True,
             "action_details": {"action_type": "click", "tap_x": 50, "tap_y": 50}},
        ]
        graph["task_runs"][0]["edge_ids"] = ["edge_0", "edge_wait", "edge_1"]

    update_seed(seed, change)
    routes = load_seed_routes(seed, PACKAGE, max_depth=2)
    assert len(routes) == 1
    assert [s.edge for s in routes[0]] == ["edge_0", "edge_1"]
    assert routes[0][0].target == routes[0][1].source
    assert routes.report["folded_wait_edges"] == 1


def test_bootstrap_prioritizes_seed_routes_matching_repair_clues(seed, tmp_path):
    from harness.gym_seed import bootstrap, load_seed_routes
    from harness.repair_clues import RepairClues

    path = seed / "assets/settings.png"
    Image.new("RGB", (100, 100), "purple").save(path)
    manifest_path = seed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["assets/settings.png"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def change(graph):
        graph["pages"]["settings"] = {"activity": PACKAGE + "/.Settings", "screenshot_path": "assets/settings.png"}
        graph["transitions"].append({
            "edge_id": "edge_1", "from_page": "root", "to_page": "settings", "success": True,
            "action_details": {"action_type": "click", "tap_x": 60, "tap_y": 60},
        })
        graph["task_runs"].append({
            "task_id": "t2", "reset_policy": "restart", "start_page_id": "root", "edge_ids": ["edge_1"],
        })

    update_seed(seed, change)
    routes = load_seed_routes(seed, PACKAGE, max_depth=3)
    assert [r[0].edge for r in routes] == ["edge_0", "edge_1"]

    clues = RepairClues({
        "format": "stg_eval_repair_clues", "schema_version": 1, "package": PACKAGE,
        "prioritized_activities": [{"activity": PACKAGE + "/.Settings", "score": 25}],
        "targets": [],
    }, tmp_path / "clues.json")
    tried_edges = []
    wrong = capture(b"\x89PNG\r\n\x1a\n" + (seed / "assets/settings.png").read_bytes()[8:])

    def log(obs):
        if obs.get("event") == "gym_seed":
            tried_edges.append(obs["edge"])

    runner = SimpleNamespace(
        config=SimpleNamespace(max_steps=4, save_screenshots=False, gym_seed_verification="exact"),
        graph=SimpleNamespace(completed_steps=0),
        current=None,
        state=SimpleNamespace(append_observation=log),
        explorer=SimpleNamespace(
            repair_clues=clues,
            controller=SimpleNamespace(restart_app=lambda: None, get_current_activity=lambda: ACTIVITY),
            _capture_screen=lambda: wrong,
            _capture_or_get_page=lambda *_: (None, False),
        ),
    )
    bootstrap(runner, routes)
    assert tried_edges == ["edge_1", "edge_0"]
    assert runner.graph.seed_handoff["prioritized_by_clues"] is True
