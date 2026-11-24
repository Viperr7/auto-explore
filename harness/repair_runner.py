"""Bounded repair runner. Only live, positively verified action evidence is exported."""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from .repair import (FORMAT, MAX_ACTIONS, MAX_JOBS, MAX_SECONDS, PACKAGE, Budget, Frame,
                     action_point, activity_name, bind, guard, pinned, safe_selector, same_state,
                     sha, version_tuple, write_new)
from .repair_plan import PRIORITY, check_review_frame, frame_of, recorded_prefix


def load_plan(path: Path, package: str, environment: str, *, plan_bytes: bytes | None = None) -> dict:
    plan = json.loads(path.read_bytes() if plan_bytes is None else plan_bytes)
    if (plan.get("format") != FORMAT or plan.get("schema_version") != 1 or not PACKAGE.fullmatch(package)
            or plan.get("package") != package or not environment or plan.get("environment") != environment):
        raise ValueError("repair_plan_identity_or_environment_mismatch")
    bounds = {"jobs": MAX_JOBS, "actions": MAX_ACTIONS, "seconds": MAX_SECONDS}
    limits = plan["limits"]
    if any(type(limits.get(k)) is not int or not 1 <= limits[k] <= n for k, n in bounds.items()):
        raise ValueError("invalid_repair_limits")
    bindings = plan["bindings"]
    if not {"baseline", "report", "gaps", "version"} <= bindings.keys() or not any(k.startswith("source:") for k in bindings):
        raise ValueError("missing_baseline_bindings")
    for pin in bindings.values():
        pinned(pin)
    if list(version_tuple(pinned(bindings["version"]))) != plan["version"]:
        raise ValueError("repair_plan_version_mismatch")
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or len(jobs) > limits["jobs"] or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate_jobs_or_limit")
    if not jobs:
        return plan
    review = json.loads(pinned(bindings["review"]))
    if (review.get("package") != package or review.get("version") != plan["version"]
            or review.get("environment") != environment or review.get("format") != "stg_repair_review"
            or review.get("schema_version") != 1 or not review.get("trace_id") or not review.get("reviewer")
            or review["trace"] != bindings["verify_trace"] or review["version_evidence"] != bindings["verify_version"]
            or list(version_tuple(pinned(bindings["verify_version"]))) != plan["version"]
            or any(review[f"{k}_sha256"] != bindings[k]["sha256"] for k in ("baseline", "report", "gaps"))
            or review.get("source_sha256") != {k: b["sha256"] for k, b in bindings.items() if k.startswith("source:")}):
        raise ValueError("invalid_verification_binding")
    cases = {c["candidate_id"]: c for c in review["cases"]}
    raw = json.loads(pinned(bindings["verify_trace"]))
    index = {e["edge_id"]: e for e in raw["transitions"]}
    raw_path = Path(bindings["verify_trace"]["path"]).resolve()
    if (raw.get("app_package") != package or len(index) != len(raw["transitions"]) or len(cases) != len(review["cases"])
            or bindings["verify_trace"]["sha256"] in {v["sha256"] for k, v in bindings.items()
                                                      if k.startswith("source:") or k == "baseline"}):
        raise ValueError("invalid_independent_trace")
    for job in jobs:
        case = cases.get(job["candidate_id"]) or {}
        steps = [*job["route"], job["target"]]
        approved = [*case.get("route", []), case.get("target", {})]
        if (job.get("risk") != "read_only" or job.get("kind") not in PRIORITY or case.get("verdict") != "verified_missing"
                or not case.get("reason") or not job.get("baseline_origins")
                or len(steps) > 11 or len(steps) != len(approved)
                or not recorded_prefix(raw, steps) or any(a["to"] != b["from"] for a, b in zip(steps, steps[1:]))
                or (job.get("retry_no_feedback") and (job["kind"] != "no_reaction" or case.get("retry_no_feedback") is not True))):
            raise ValueError("unapproved_repair_job")
        for step, approval in zip(steps, approved):
            original = index[step["edge_id"]]
            if ((step["from"], step["to"]) != (str(original["from_page"]), str(original["to_page"]))
                    or original.get("success") is not True or (original.get("action_details") or {}).get("action_type") != "click"
                    or any(step[key] != approval.get(key) for key in ("edge_id", "intent", "selector"))
                    or step["intent"] not in {"read_only_navigation", "read_only_expansion"}
                    or not safe_selector(step["selector"], package)):
                raise ValueError("unsafe_or_tampered_repair_step")
            for side, raw_id in (("before", step["from"]), ("after", step["to"])):
                page, pin = raw["pages"][raw_id], step[side]
                if (pin["activity"] != page["activity"] or any(
                        Path(pin[key]["path"]).resolve() != (raw_path.parent / page[field]).resolve()
                        for key, field in (("png", "screenshot_path"), ("xml", "hierarchy_path")))):
                    raise ValueError("repair_step_references_wrong_frame")
                check_review_frame(pin, review, raw_path)
            if action_point(frame_of(step["before"]), step["selector"], package) is None or not guard(frame_of(step["after"]), package):
                raise ValueError("unproven_repair_step")
        target = job["target"]
        if (not same_state(frame_of(job["baseline_evidence"]), frame_of(target["before"]), package)
                or same_state(frame_of(target["before"]), frame_of(target["after"]), package)):
            raise ValueError("repair_source_mismatch")
    return plan


class AdbRepairDevice:
    """Narrow controller: no model, shell directives, arbitrary input, install or account operations."""
    def __init__(self, adb, serial, package, budget, *, delay=1.0):
        if not serial or not re.fullmatch(r"[\w.:%-]+", serial, re.ASCII):
            raise ValueError("explicit_valid_adb_serial_required")
        self.adb, self.serial, self.package, self.budget = adb, serial, package, budget
        self.delay = max(0.0, min(delay, 10.0))

    def adb_call(self, *args):
        return subprocess.run([self.adb, "-s", self.serial, *args], check=True, capture_output=True,
                              timeout=min(25.0, self.budget.remaining())).stdout

    def active(self):
        text = self.adb_call("shell", "dumpsys", "activity", "activities").decode("utf-8", "replace")
        rows = [line for line in text.splitlines() if "topResumedActivity" in line or "mResumedActivity" in line]
        names = {m for line in rows for m in re.findall(r"([\w.]+/[.\w$]+)", line)}
        canon = {activity_name(name, self.package) or name for name in names}
        return next(iter(canon)) if len(canon) == 1 else ""

    def preflight(self):
        if self.adb_call("get-state").strip() != b"device":
            raise ValueError("adb_device_unavailable")
        if not activity_name(self.active(), self.package):
            raise ValueError("target_app_must_be_foreground")
        return version_tuple(self.adb_call("shell", "dumpsys", "package", self.package))

    def capture(self):
        activity = self.active()
        png = self.adb_call("exec-out", "screencap", "-p")
        xml = b""
        try:
            data = self.adb_call("exec-out", "uiautomator", "dump", "/dev/tty")
            start, end = data.find(b"<hierarchy"), data.rfind(b"</hierarchy>")
            if 0 <= start < end:
                xml = data[start:end+len(b"</hierarchy>")]
        except subprocess.CalledProcessError:
            pass  # Screenshot-only evidence is preserved but never authorizes a tap.
        after_png = self.adb_call("exec-out", "screencap", "-p")
        if self.active() != activity:
            return Frame("", after_png)  # The capture straddled a foreground change.
        frame = Frame(activity, png, xml)
        after = Frame(activity, after_png)
        # A tree obtained during a visible transition must not authorize a tap.
        if xml and (frame.size() != after.size() or frame.pixels() != after.pixels()):
            return after
        return frame

    def tap(self, x, y):
        if not activity_name(self.active(), self.package):
            return False
        self.adb_call("shell", "input", "tap", str(x), str(y))
        self.settle()
        return True

    def prepare_restart(self):
        if not activity_name(self.active(), self.package):
            raise ValueError("restart_outside_target_app")
        data = self.adb_call("shell", "cmd", "package", "resolve-activity", "--brief", "-a",
                             "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER",
                             self.package).decode("utf-8", "replace")
        choices = {activity_name(line, self.package) for line in data.splitlines()} - {""}
        if len(choices) != 1:
            raise ValueError("no_unique_launcher")
        return next(iter(choices))

    def stop(self):
        self.adb_call("shell", "am", "force-stop", self.package)

    def launch(self, component):
        self.adb_call("shell", "am", "start", "-n", component)
        self.settle()

    def settle(self):
        time.sleep(min(self.delay, self.budget.remaining()))


class Evidence:
    def __init__(self, output, package):
        self.output = output
        self.package = package
        (output / "assets").mkdir()
        self.graph = {"app_package": package, "pages": {}, "transitions": [], "task_runs": []}
        self.events = []

    def observe(self, frame):
        oid = f"obs_{len(self.graph['pages']):04}"
        png = f"assets/{oid}.png"
        (self.output / png).write_bytes(frame.png)
        tree = ""
        if frame.xml:
            tree = f"assets/{oid}.xml"
            (self.output / tree).write_bytes(frame.xml)
        self.graph["pages"][oid] = {"activity": frame.activity, "screenshot_path": png,
            "hierarchy_path": tree, "evidence_kind": "visual_and_tree" if guard(frame, self.package) else "visual_only"}
        return oid

    def record(self, job, phase, a, b, point, status, *, visual_changed=False):
        event = {"job_id": job, "phase": phase, "from": a, "to": b, "point": point, "status": status}
        if status == "verified":
            # Recovery is evidence for the route, not a newly sampled business transition.
            if phase == "target":
                x, y = point
                p, q = (self.graph["pages"][key] for key in (a, b))
                eid = f"repair_{len(self.graph['transitions']):04}"
                self.graph["transitions"].append({"edge_id": eid, "from_page": a, "to_page": b,
                    "success": True, "visual_changed": visual_changed, "state_changed": True,
                    "action_details": {"action_type": "click", "tap_x": x, "tap_y": y},
                    "before_screenshot": p["screenshot_path"], "after_screenshot": q["screenshot_path"],
                    "before_hierarchy_path": p["hierarchy_path"], "after_hierarchy_path": q["hierarchy_path"],
                    "repair_job": job})
                event["edge_id"] = eid
        self.events.append(event)
        return event


def run_repairs(plan_path, output, package, environment, device_factory, *, clock=time.monotonic):
    plan_bytes = plan_path.read_bytes()
    plan = load_plan(plan_path, package, environment, plan_bytes=plan_bytes)
    output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence(output, package)
    limits = plan["limits"]
    budget = Budget(limits["actions"], limits["seconds"], clock=clock)
    jobs = plan["jobs"]
    results = []
    started = clock()
    current = None
    device = None
    stop_reason = "no_executable_jobs" if not jobs else "completed"

    def observe():
        budget.remaining()
        return device.capture()

    def endpoint(frame):
        return evidence.observe(frame)

    def perform(job, step, frame, phase):
        fresh = observe()
        if (not same_state(frame, fresh, package) or fresh.pixels() != frame.pixels()
                or not same_state(fresh, frame_of(step["before"]), package)):
            return fresh, evidence.record(job, phase, endpoint(frame), endpoint(fresh), None, "stale_or_wrong_source")
        point = action_point(fresh, step["selector"], package)
        fid = endpoint(fresh)
        if point is None:
            return fresh, evidence.record(job, phase, fid, None, None, "unsafe_or_ambiguous_control")
        try:
            executed = budget.invoke(device.tap, *point)
            after = observe()
        except (TimeoutError, OSError, ValueError, subprocess.SubprocessError) as exc:
            evidence.record(job, phase, fid, None, point, "attempt_failed:" + type(exc).__name__)
            raise
        aid = endpoint(after)
        if not executed:
            status = "action_failed"
        elif not activity_name(after.activity, package):
            status = "out_of_app"
        elif guard(after, package) is None:
            status = "visual_only"
        elif same_state(after, frame_of(step["after"]), package):
            status = "verified"
        elif same_state(after, fresh, package) and after.pixels() == fresh.pixels():
            status = "no_feedback"
        else:
            status = "different_branch"
        return after, evidence.record(job, phase, fid, aid, point, status,
                                      visual_changed=after.pixels() != fresh.pixels())

    try:
        if jobs:
            device = device_factory(budget)
            if list(device.preflight()) != plan["version"]:
                raise ValueError("installed_app_version_mismatch")
        for job in jobs:
            jid, target = job["job_id"], job["target"]
            if current is None:
                current = observe()
            if not same_state(current, frame_of(target["before"]), package):
                for _ in range(2):
                    prefix = job["route"]
                    entry = frame_of((prefix[0] if prefix else target)["before"])
                    if not same_state(current, entry, package):
                        if not activity_name(current.activity, package):
                            break
                        launcher = device.prepare_restart()
                        # Reserve room for both lifecycle commands before force-stopping.
                        if budget.limit - budget.used < 2:
                            raise TimeoutError("restart_would_exhaust_budget")
                        budget.invoke(device.stop)
                        budget.invoke(device.launch, launcher)
                        current = observe()
                        evidence.record(jid, "restart", None, endpoint(current), None, "lifecycle")
                    if not same_state(current, entry, package):
                        continue
                    route_ok = True
                    for step in prefix:
                        current, event = perform(jid, step, current, "recovery")
                        if event["status"] != "verified":
                            route_ok = False
                            break
                    if route_ok and same_state(current, frame_of(target["before"]), package):
                        break
                else:
                    route_ok = False
                if not activity_name(current.activity, package) or not route_ok:
                    results.append({"job_id": jid, "status": "recovery_failed", "observation": endpoint(current)})
                    if not activity_name(current.activity, package):
                        stop_reason = "out_of_app"
                        break
                    continue
            if not same_state(current, frame_of(target["before"]), package):
                results.append({"job_id": jid, "status": "recovery_failed", "observation": endpoint(current)})
                if not activity_name(current.activity, package):
                    stop_reason = "out_of_app"
                    break
                continue
            current, event = perform(jid, target, current, "target")
            if event["status"] == "no_feedback" and job.get("retry_no_feedback"):
                current, event = perform(jid, target, current, "target")
            results.append({**event, "status": {"verified": "sampled", "unsafe_or_ambiguous_control": "safety_skip",
                                                "stale_or_wrong_source": "safety_skip"}.get(event["status"], event["status"])})
            if event["status"] == "out_of_app":
                stop_reason = "out_of_app"
                break
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        stop_reason = "budget_exhausted" if isinstance(exc, TimeoutError) else "device_timeout"
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        stop_reason = "blocked:" + str(exc)[:160]
    finally:
        finished = {r["job_id"] for r in results}
        results.extend({"job_id": j["job_id"], "status": "not_run", "reason": stop_reason}
                       for j in jobs if j["job_id"] not in finished)
        write_new(output / "repair_observations.json", evidence.graph)
        keep = {e[k] for e in evidence.graph["transitions"] for k in ("from_page", "to_page")}
        merge_pages = {k: v for k, v in evidence.graph["pages"].items() if k in keep}
        write_new(output / "app_graph.json", {**evidence.graph, "pages": merge_pages})
        payload = {"schema_version": 1, "package": package, "plan_sha256": sha(plan_bytes),
                   "termination": stop_reason, "device_actions": budget.used, "wall_seconds": round(clock()-started, 3),
                   "jobs": results, "events": evidence.events, "graph_sha256": bind(output / "app_graph.json")["sha256"],
                   "asset_sha256": {p[key]: bind(output / p[key])["sha256"] for p in merge_pages.values()
                                    for key in ("screenshot_path", "hierarchy_path")}}
        write_new(output / "repair_results.json", payload)
    return payload
