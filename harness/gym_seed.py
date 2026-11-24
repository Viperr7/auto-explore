"""Bridge Gym Explore traces into the live state queue, never importing old work items."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import base64
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from PIL import Image

from .actions import ActionCodec
from .grounding import matches_source, with_source
from .identity import behavioral_changes
from .verify import ExplorationVerifier


MAX_SEED_ACTIONS = 20
TOP, BOTTOM = 0.07, 0.05


@dataclass(frozen=True)
class Screen:
    activity: str
    size: tuple[int, int]
    digest: str
    image_b64: str = field(default="", compare=False, repr=False)
    full_digest: str = field(default="", compare=False)


@dataclass(frozen=True)
class SeedStep:
    source: Screen
    target: Screen
    x: int
    y: int
    trace: str
    edge: str
    labels: tuple[str, ...] = ()
    resource_id: str = ""


class SeedRoutes(list):
    """List-compatible routes with historical/candidate counts, never coverage."""

    def __init__(self, report):
        super().__init__()
        self.report = report


def _activity(value):
    package, _, component = str(value).partition("/")
    return package + "/" + (package + component if component.startswith(".") else component) if component else ""


def _screen(activity, data):
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
        pixels = image.convert("RGB").crop((0, int(height * TOP), width, int(height * (1 - BOTTOM))))
        return Screen(_activity(activity), image.size, sha256(pixels.tobytes()).hexdigest(),
                      base64.b64encode(data).decode("ascii"), sha256(image.convert("RGB").tobytes()).hexdigest())


def load_seed_routes(path, package, *, max_depth):
    root = Path(path).resolve()
    if not (root / "manifest.json").is_file():
        root = root / "auto_explore_seed"
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read Gym seed at {root}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != "gym_explore_seed" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Gym seed format/version")
    if manifest.get("app_package") != package:
        raise ValueError("Gym seed package differs from the target App")
    if not isinstance(manifest.get("files"), dict) or not isinstance(manifest.get("graphs"), list):
        raise ValueError("Gym seed manifest lacks files/graphs")

    def checked(name, folder):
        if not isinstance(name, str):
            raise ValueError("Gym seed file path must be a string")
        target = (root / name).resolve()
        if Path(name).is_absolute() or not target.is_relative_to(root / folder):
            raise ValueError(f"Gym seed path outside {folder}: {name}")
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise ValueError(f"Missing Gym seed file: {name}") from exc
        if sha256(data).hexdigest() != manifest["files"].get(name):
            raise ValueError(f"Gym seed checksum mismatch: {name}")
        return data

    @lru_cache(maxsize=512)
    def image(activity, name):
        return _screen(activity, checked(name, "assets"))

    rejected = Counter()
    report = {"source": str(root), "historical_graphs": len(manifest["graphs"]), "historical_pages": 0,
              "historical_edges": 0, "historical_tasks": 0, "candidate_routes": 0,
              "candidate_steps": 0, "rejections": {}, "history_is_current_coverage": False}
    routes, seen = SeedRoutes(report), set()
    for item in manifest["graphs"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid Gym seed graph entry")
        name = item.get("path")
        data = checked(name, "graphs")
        if sha256(data).hexdigest() != item.get("sha256"):
            raise ValueError(f"Gym seed graph checksum mismatch: {name}")
        graph = json.loads(data)
        if not isinstance(graph, dict) or graph.get("app_package") != package:
            raise ValueError("Gym seed graph package differs from the target App")
        pages = graph.get("pages")
        if not isinstance(pages, dict):
            raise ValueError("Gym seed graph lacks pages")
        report["historical_pages"] += len(pages)
        report["historical_edges"] += len(graph.get("transitions", []))
        report["historical_tasks"] += len(graph.get("task_runs", []))
        edges = {e["edge_id"]: e for e in graph.get("transitions", [])
                 if isinstance(e, dict) and isinstance(e.get("edge_id"), str)}
        for task in graph.get("task_runs", []):
            if not isinstance(task, dict) or task.get("reset_policy") != "restart":
                rejected["task_not_restart"] += 1
                continue
            source_id = task.get("start_page_id")
            steps = []
            for edge_id in task.get("edge_ids", [])[:max_depth]:
                edge = edges.get(edge_id, {})
                target_id = edge.get("to_page")
                action = edge.get("action_details") or {}
                x, y = action.get("tap_x"), action.get("tap_y")
                reason = ("failed_or_disconnected_edge" if edge.get("success") is not True
                          or edge.get("from_page") != source_id or not target_id else
                          "self_loop" if target_id == source_id else
                          "unsupported_action" if action.get("action_type") != "click" else
                          "invalid_coordinates" if type(x) is not int or type(y) is not int else "")
                if reason:
                    rejected[reason] += 1
                    break  # Only an uninterrupted, successful navigation prefix can be seeded.
                source, target = pages.get(source_id, {}), pages.get(target_id, {})
                activities = [page.get("activity") or "" for page in (source, target)]
                shots = [edge.get(key) or page.get("screenshot_path") for key, page in
                         (("before_screenshot", source), ("after_screenshot", target))]
                if not all(a.startswith(package + "/") for a in activities) or not all(shots):
                    rejected["missing_or_external_evidence"] += 1
                    break
                before, after = (image(a, s) for a, s in zip(activities, shots))
                width, height = before.size
                # Cropping noisy system chrome must not discard actual App toolbars.
                # Live planning still authorizes the target; crop-area taps get a full-image check.
                if not (0 <= x < width and 0 <= y < height):
                    rejected["invalid_coordinates"] += 1
                    break
                if steps and steps[-1].target != before:
                    rejected["discontinuous_screenshots"] += 1
                    break
                labels = [action.get("element_text") or ""]
                resource_id = (action.get("locator") or {}).get("resource_id", "")
                hierarchy = edge.get("before_hierarchy_path") or source.get("hierarchy_path")
                if hierarchy:
                    label, rid = _control_hint(checked(hierarchy, "assets"), x, y, package)
                    labels.append(label)
                    resource_id = rid or resource_id
                steps.append(SeedStep(before, after, x, y, f"{name}:{task.get('task_id', '')}", edge_id,
                                      tuple(dict.fromkeys(v for v in labels if isinstance(v, str) and v.strip())),
                                      resource_id))
                source_id = target_id
            key = tuple((s.source, s.target, s.x, s.y, s.labels, s.resource_id,
                         (s.source.full_digest, s.target.full_digest)
                         if not int(s.source.size[1] * TOP) <= s.y < int(s.source.size[1] * (1 - BOTTOM)) else ())
                        for s in steps)
            if steps and key not in seen:
                seen.add(key)
                routes.append(tuple(steps))
    report.update(candidate_routes=len(routes), candidate_steps=sum(map(len, routes)), rejections=dict(rejected))
    return routes


def _control_hint(data, x, y, package):
    """Recover a semantic hint at a recorded tap; XML is evidence, never executable."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return "", ""
    hits = []
    for node in root.iter("node"):
        bounds = [int(v) for v in re.findall(r"-?\d+", node.get("bounds", ""))]
        if (len(bounds) != 4 or node.get("package", package) != package or node.get("enabled") == "false"
                or not (bounds[0] <= x < bounds[2] and bounds[1] <= y < bounds[3])):
            continue
        label = node.get("text") or node.get("content-desc") or ""
        rid = node.get("resource-id") or ""
        if label or rid:
            hits.append((node.get("clickable") != "true", (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]), label, rid))
    return min(hits)[2:] if hits else ("", "")


def _matches(activity, capture, expected):
    return (_activity(activity) == expected.activity and tuple(capture.screen_size) == expected.size
            and _screen(activity, base64.b64decode(capture.image_b64)) == expected)


def _same_live_screen(before, current, tap_y):
    """Preserve chrome tolerance only when the click is in identical content pixels."""
    if tuple(before.screen_size) != tuple(current.screen_size):
        return False
    if before == current:
        return True
    first = _screen("", base64.b64decode(before.image_b64))
    last = _screen("", base64.b64decode(current.image_b64))
    height = before.screen_size[1]
    return (first.full_digest == last.full_digest or
            (int(height * TOP) <= tap_y < int(height * (1 - BOTTOM)) and first == last))


def _navigation(position, step, *, relocate=False):
    if position.work is None or position.guard.get("conflicts"):
        return None
    hits = []
    labels = {" ".join(v.casefold().split()) for v in step.labels}
    rid = step.resource_id.rsplit("/", 1)[-1]
    for action in position.work.planned_actions:
        bounds = action.bounds
        if (not bounds or action.action_type != "click" or not matches_source(action, position.capture)
                or not ExplorationVerifier.has_executable_action(action, position.capture.screen_size)):
            continue
        identity = {" ".join(v.casefold().split()) for v in (action.element_text, action.target_description) if v}
        semantic_hit = bool(labels & identity or (rid and action.control_key.rsplit("/", 1)[-1] == rid))
        if (semantic_hit if relocate else
                bounds["x"] <= step.x < bounds["x"] + bounds["width"]
                and bounds["y"] <= step.y < bounds["y"] + bounds["height"]):
            hits.append(action)
    if len(hits) == 1 and hits[0].intent == "navigate" and hits[0].replay_safe and hits[0].effect_kind != "persistent":
        return hits[0]
    return None


def _verify_screen(runner, capture, expected, step, cache, *, target=False, full=False):
    """Exact evidence is the fast path; drift requires the existing bounded replay judge."""
    explorer = runner.explorer
    activity = explorer.controller.get_current_activity()
    if _activity(activity) != expected.activity or tuple(capture.screen_size) != expected.size:
        return False, "activity_or_size_mismatch"
    actual = _screen(activity, base64.b64decode(capture.image_b64))
    if actual == expected and (not full or actual.full_digest == expected.full_digest):
        return True, "pixels"
    if not target and not step.labels and not step.resource_id:
        return False, "missing_control_identity"
    judge = getattr(getattr(explorer, "analyzer", None), "verify_replay_state", None)
    if (getattr(runner.config, "gym_seed_verification", "adaptive") != "adaptive"
            or not callable(judge) or not expected.image_b64):
        return False, "pixel_mismatch"
    key = (actual.full_digest, expected.full_digest, expected.activity, target, step.labels, step.resource_id)
    if key not in cache:
        operation = {"action_type": "click", "target_description": " / ".join(step.labels),
                     "element_text": next(iter(step.labels), ""), "resource_id": step.resource_id}
        try:
            verdict = judge(capture.image_b64, capture.screen_size,
                expected={"activity": expected.activity, "current_activity": activity,
                          "state": {}, "context": {}, "values": {}, "conflicts": []},
                page_description="Historical Research navigation observation; verify workflow state, not template only",
                reference_b64=expected.image_b64, reference_size=expected.size, reference_is_template=False,
                action=operation if target else None, next_action=None if target else operation)
            matched = (isinstance(verdict, dict) and verdict.get("status") == "success"
                       and type(verdict.get("matched")) is bool and verdict["matched"])
            reason = str(verdict.get("reason", ""))[:500] if isinstance(verdict, dict) else "invalid_response"
        except Exception as exc:
            matched, reason = False, f"{type(exc).__name__}: {exc}"
        cache[key] = matched, "vlm" if matched else "vlm_rejected_or_unavailable"
        runner.state.append_observation({"kind": "state_restore", "event": "gym_seed_verification",
            "trace": step.trace, "edge": step.edge, "phase": "target" if target else "source",
            "arrived": matched, "verification_method": "vlm", "reason": reason,
            "current_fingerprint": capture.screenshot_hash})
    return cache[key]


def bootstrap(runner, routes):
    """Re-observe navigation prefixes; only the ordinary live registry creates queue entries."""
    explorer = runner.explorer
    controller = explorer.controller
    report = {**getattr(routes, "report", {}), "attempted_routes": 0,
              "executed_actions": 0, "verified_arrivals": 0, "runtime_rejections": {},
              "verification_mode": getattr(runner.config, "gym_seed_verification", "adaptive")}
    runner.graph.seed_handoff = report
    cache = {}
    rejected = Counter()
    budget = MAX_SEED_ACTIONS
    if runner.config.max_steps:
        budget = min(budget, max(0, runner.config.max_steps - runner.graph.completed_steps - 1))

    def log(step, *, position=None, **details):
        if details.get("arrived") is False:
            rejected[details.get("reason", "unknown")] += 1
            report["runtime_rejections"] = dict(rejected)
        runner.state.append_observation({"kind": "state_restore", "event": "gym_seed",
            "trace": step.trace, "edge": step.edge,
            "page_id": position.page.page_id if position else "",
            "work_id": position.work.work_id if position and position.work else "", **details})

    # Bound restart attempts too, including prefixes rejected before their first tap.
    for route in routes[:MAX_SEED_ACTIONS]:
        if budget <= 0:
            break
        report["attempted_routes"] += 1
        runner.current = None
        controller.restart_app()
        capture = explorer._capture_screen()
        first = route[0]
        toolbar = not int(first.source.size[1] * TOP) <= first.y < int(first.source.size[1] * (1 - BOTTOM))
        matched, method = _verify_screen(runner, capture, first.source, first, cache, full=toolbar)
        if not matched:
            log(first, arrived=False, reason="entry_mismatch", verification_method=method)
            continue
        page, _ = explorer._capture_or_get_page(capture, [])
        if page is None:
            log(first, arrived=False, reason="entry_not_registered", verification_method=method)
            continue
        runner.current = runner.observe(page, capture, [])
        for step in route[:budget]:
            before = runner.current
            if before is None:
                break
            fresh = explorer._capture_screen()
            # Never rebind a plan to an image that changed while it was analyzed.
            if not _same_live_screen(before.capture, fresh, step.y):
                log(step, arrived=False, position=before, reason="stale_current_analysis")
                runner.current = None
                break
            toolbar = not int(step.source.size[1] * TOP) <= step.y < int(step.source.size[1] * (1 - BOTTOM))
            source_matches, method = _verify_screen(runner, fresh, step.source, step, cache, full=toolbar)
            planned = _navigation(before, step, relocate=method == "vlm") if source_matches else None
            if planned is None or not source_matches:
                log(step, arrived=False, position=before,
                    reason="source_mismatch" if not source_matches else "unsafe_or_unresolved_navigation",
                    verification_method=method)
                # Analysis may outlive a live screen. Never continue with a stale position.
                if not source_matches or fresh != before.capture:
                    runner.current = None
                break
            if method == "vlm":
                tap_y = ActionCodec.tap_point_for_bounds(planned.bounds)[1]
                checked = explorer._capture_screen()
                if (not _same_live_screen(before.capture, fresh, tap_y)
                        or not _same_live_screen(fresh, checked, tap_y)
                        or _activity(controller.get_current_activity()) != step.source.activity):
                    log(step, arrived=False, position=before, reason="stale_after_verification")
                    runner.current = None
                    break
                fresh = checked
            action = (with_source(planned, fresh, "gym_seed_current_control") if method == "vlm" else
                      with_source(replace(planned, bounds={"x": step.x, "y": step.y, "width": 1, "height": 1}),
                                  fresh, "gym_seed_verified_pixels"))
            seq = replace(ActionCodec.to_sequence_step(action, fresh.screen_size), source_guard=before.guard)
            path = before.path + [seq]
            budget -= 1
            try:
                success = bool(controller.execute_action(action, fresh.screen_size))
            finally:
                runner.graph.completed_steps += 1
                report["executed_actions"] += 1
            after_capture = explorer._capture_screen()
            activity = controller.get_current_activity()
            outside = not explorer._is_target_activity(activity)
            matched = False
            target_method = "not_executed_or_external"
            if success and not outside:
                matched, target_method = _verify_screen(runner, after_capture, step.target, step, cache,
                                                       target=True, full=toolbar)
            after = None
            if matched:
                page, created = explorer._capture_or_get_page(after_capture, path, source_action=action)
                if page is not None:
                    after = runner.observe(page, after_capture, path, finish_step=True)
                    if created:
                        page.action_sequence = list(after.path)
            before_image = after_image = ""
            if runner.config.save_screenshots:
                before_image, after_image = explorer._save_transition_screens(fresh, after_capture)
            observation = runner.state.record_transition(
                from_page=before.page.page_id, to_page=after.page.page_id if after else "",
                action=action, success=success, destination_verified=after is not None,
                state_changed=bool(after and behavioral_changes(before.guard, after.guard)),
                before_screenshot=before_image, after_screenshot=after_image,
                before_fingerprint=fresh.screenshot_hash, after_fingerprint=after_capture.screenshot_hash,
                from_state_id=before.visual.state_id, to_state_id=after.visual.state_id if after else "",
                from_work_id=before.work.work_id,
                to_work_id=after.work.work_id if after and after.work else "",
                recovery="gym_seed", purpose="gym_seed", arrival_path=before.path,
                current_layout_id=action.source_layout_id,
                outside_app=outside, external_activity=activity if outside else "",
                visual_changed=after_capture.screenshot_hash != fresh.screenshot_hash,
                error_stage="" if after else "identification" if success else "execution",
                error="" if after else "seed_target_unverified",
            )
            log(step, position=after, arrived=after is not None,
                reason="target_matched" if after else "seed_target_unverified",
                verification_method=target_method, action_observation_id=observation["observation_id"])
            report["verified_arrivals"] += int(after is not None)
            if after and after.work:
                runner.registry.record_result(before.work, planned, action, after.work,
                                              before.guard, after.guard, observation, "gym_seed")
            runner.current = after
    if runner.current is None:
        controller.restart_app()
        capture = explorer._capture_screen()
        page, _ = explorer._capture_or_get_page(capture, [])
        if page is not None:
            runner.current = runner.observe(page, capture, [])
