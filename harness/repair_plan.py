"""Compile STG-eval diagnostics and independent, reviewed trace A into a repair plan."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

from .repair import (FORMAT, MAX_ACTIONS, MAX_JOBS, MAX_SECONDS, Frame, action_point,
                     bind, guard, pinned, sha, version_tuple, write_new)

CONTROL_TYPES = {"unexercised", "unresolved_target", "dangling", "no_reaction", "inconsistent_control"}
PRIORITY = {"missing_screen": 0, "dangling": 0, "unresolved_target": 1, "unexercised": 2,
            "no_reaction": 3, "inconsistent_control": 3}


def read(path: Path):
    return json.loads(path.read_bytes())


def baseline_artifact(path: Path) -> Path:
    if path.is_file():
        return path
    merge = path / "merge_report.json"
    return merge if merge.is_file() else path / "topology/data.json"


def asset(graph_path, name):
    if not name:
        raise ValueError("reference_assets_missing")
    return (graph_path.parent / name).resolve()


def frame_pin(page: dict, graph_path: Path) -> dict:
    return {"activity": page["activity"],
            "png": bind(asset(graph_path, page.get("screenshot_path"))),
            "xml": bind(asset(graph_path, page.get("hierarchy_path")))}


def frame_of(pin: dict) -> Frame:
    return Frame(pin["activity"], pinned(pin["png"]), pinned(pin["xml"]))


def recorded_prefix(graph: dict, steps: list) -> bool:
    ids = [s["edge_id"] for s in steps]
    return any(t.get("reset_policy") == "restart" and str(t.get("start_page_id")) == steps[0]["from"]
               and t.get("edge_ids", [])[:len(ids)] == ids for t in graph.get("task_runs", []))


def candidate_id(row: dict) -> str:
    key = [row.get(k, "") for k in ("type", "node", "screen", "observation", "control")]
    return sha(json.dumps(key, ensure_ascii=False, sort_keys=True).encode())[:24]


def candidates(report: dict, gaps: list) -> dict:
    rows = [r for r in report["findings"] if r["type"] not in {k+"_control" for k in CONTROL_TYPES - {"inconsistent_control"}}]
    rows += [{**gap, "type": gap["status"]} for gap in gaps]
    return {candidate_id(r): r for r in rows}


def baseline_guard_ids(stg: dict, package: str) -> dict:
    result = {}
    for oid, obs in stg["observations"].items():
        if not stg["in_app"].get(oid) or not obs.get("screenshot") or not obs.get("hierarchy"):
            continue
        try:
            pin = {"activity": obs["activity"], "png": bind(Path(obs["screenshot"])), "xml": bind(Path(obs["hierarchy"]))}
            state = guard(frame_of(pin), package)
        except (OSError, ValueError):
            continue
        if state:
            result[oid] = state, pin
    return result


def check_review_frame(frame: dict, review: dict, graph_path: Path) -> None:
    assets = review.get("asset_sha256") or {}
    for key in ("png", "xml"):
        pin = frame[key]
        path = Path(pin["path"])
        if not path.is_relative_to(graph_path.parent):
            raise ValueError("review_asset_outside_trace_directory")
        if assets.get(path.relative_to(graph_path.parent).as_posix()) != pin["sha256"]:
            raise ValueError("review_asset_hash_mismatch")


def trace_step(trace, graph_path, row, package, review_step):
    src, dst = str(row["from_page"]), str(row["to_page"])
    action = row.get("action_details") or {}
    if (row.get("success") is not True or action.get("action_type") != "click"
            or review_step.get("intent") not in {"read_only_navigation", "read_only_expansion"}):
        raise ValueError("unsafe_or_unverified_action")
    before, after = (frame_pin(trace["pages"][key], graph_path) for key in (src, dst))
    a, b = frame_of(before), frame_of(after)
    selector = review_step["selector"]
    point = action_point(a, selector, package)
    if point is None or guard(b, package) is None:
        raise ValueError("unsafe_ambiguous_or_external_control")
    if type(action.get("tap_x")) is not int or type(action.get("tap_y")) is not int:
        raise ValueError("trace_tap_missing")
    # The reviewed selector must be the control actually hit in the independent trace.
    from gym_explore.evaluation.stg import hit_test
    target = hit_test(before["xml"]["path"], action["tap_x"], action["tap_y"])
    if (not target or not target.get("clickable") or target.get("resource_id") != selector["resource_id"].split("/")[-1]
            or target.get("label") != selector["label"]):
        raise ValueError("trace_selector_mismatch")
    for field, frame, key in (("before_screenshot", before, "png"), ("after_screenshot", after, "png"),
                               ("before_hierarchy_path", before, "xml"), ("after_hierarchy_path", after, "xml")):
        if not row.get(field) or bind(asset(graph_path, row[field]))["sha256"] != frame[key]["sha256"]:
            raise ValueError("missing_or_mismatched_step_evidence")
    return {"edge_id": row["edge_id"], "from": src, "to": dst, "before": before, "after": after,
            "intent": review_step["intent"], "selector": selector}


def compile_plan(stg_path, report_path, gaps_path, version_path, environment, *, review_path=None):
    from gym_explore.evaluation.stg import load_stg
    report = read(report_path)
    stg = load_stg(stg_path)
    package = stg["package"]
    if (report["stg"]["package"] != package or Path(report["stg"]["path"]).resolve() != stg_path.resolve()
            or {k: Path(v).resolve() for k, v in report["stg"]["sources"].items()}
            != {k: Path(v).resolve() for k, v in stg["sources"].items()}):
        raise ValueError("report_baseline_mismatch")
    if not environment.strip():
        raise ValueError("repair_environment_required")
    version = version_tuple(version_path.read_bytes())
    bindings = {"baseline": bind(baseline_artifact(stg_path)), "report": bind(report_path),
                "gaps": bind(gaps_path), "version": bind(version_path),
                **{f"source:{k}": bind(Path(v)) for k, v in stg["sources"].items()}}
    rows = candidates(report, read(gaps_path))
    mergeable = (stg_path.is_dir() and not (stg_path / "merge_report.json").exists()
                 and (stg_path / "topology/data.json").is_file())
    review = read(review_path) if review_path else None
    assignments, seen = {}, set()
    if review is not None:
        if (review.get("format") != "stg_repair_review" or review.get("schema_version") != 1
                or review.get("package") != package or review.get("version") != list(version)
                or review.get("environment") != environment or not review.get("trace_id") or not review.get("reviewer")
                or review.get("report_sha256") != bindings["report"]["sha256"]
                or review.get("gaps_sha256") != bindings["gaps"]["sha256"]
                or review.get("baseline_sha256") != bindings["baseline"]["sha256"]
                or review.get("source_sha256") != {k: b["sha256"] for k, b in bindings.items() if k.startswith("source:")}):
            raise ValueError("independent_review_does_not_pin_baseline_or_environment")
        trace_pin, vpin = review["trace"], review["version_evidence"]
        graph_path = Path(trace_pin["path"]).resolve()
        if version_tuple(pinned(vpin)) != version:
            raise ValueError("review_trace_version_mismatch")
        if trace_pin["sha256"] in {b["sha256"] for b in bindings.values()}:
            raise ValueError("verify_trace_used_in_baseline")
        trace = json.loads(pinned(trace_pin))
        if trace.get("app_package") != package:
            raise ValueError("verify_trace_package_mismatch")
        bindings.update(review=bind(review_path), verify_trace=trace_pin, verify_version=vpin)
        edge_index = {e["edge_id"]: e for e in trace["transitions"]}
        if len(edge_index) != len(trace["transitions"]):
            raise ValueError("verification_trace_duplicate_edges")
        states = baseline_guard_ids(stg, package)
        known_images = {b["png"]["sha256"] for _, b in states.values()}
        from gym_explore.graph_import import load_graph
        for value in stg["sources"].values():
            path = Path(value)
            graph = load_graph(path)
            for item, keys in [(p, ("screenshot_path",)) for p in graph["pages"].values()] + [
                    (e, ("before_screenshot", "after_screenshot")) for e in graph.get("transitions", [])]:
                for name in keys:
                    shot = item.get(name)
                    if shot and (p := asset(path, shot)).is_file():
                        known_images.add(bind(p)["sha256"])
        for item in review["cases"]:
            cid = item["candidate_id"]
            if cid not in rows or cid in assignments:
                raise ValueError("unknown_or_duplicate_review_candidate")
            assignments[cid] = (None, "review_did_not_confirm_gap")
            if rows[cid]["type"] not in PRIORITY or item.get("verdict") != "verified_missing":
                continue
            try:
                if not mergeable:
                    raise ValueError("baseline_not_supported_for_repair_merge")
                if not str(item.get("reason", "")).strip():
                    raise ValueError("missing_review_reason")
                steps = [trace_step(trace, graph_path, edge_index[s["edge_id"]], package, s)
                         for s in [*item.get("route", []), item["target"]]]
                for step in steps:
                    for side in ("before", "after"):
                        check_review_frame(step[side], review, graph_path)
                if len(steps) > 11 or any(a["to"] != b["from"] for a, b in zip(steps, steps[1:])):
                    raise ValueError("discontinuous_or_long_route")
                if not recorded_prefix(trace, steps):
                    raise ValueError("route_not_bound_to_cold_start")
                if any(s[side]["png"]["sha256"] in known_images for s in steps for side in ("before", "after")):
                    raise ValueError("verify_frame_reused_from_baseline")
                source, target = (guard(frame_of(steps[-1][k]), package) for k in ("before", "after"))
                if source == target:
                    raise ValueError("expected_change_not_observed")
                oid = item["baseline_observation"]
                if (oid not in states or states[oid][0] != source
                        or (rows[cid].get("observation") and rows[cid]["observation"] != oid)):
                    raise ValueError("baseline_source_state_not_verified")
                control = rows[cid].get("control") or {}
                if control and (not control.get("resource_id") or control["resource_id"] not in
                                {steps[-1]["selector"]["resource_id"], steps[-1]["selector"]["resource_id"].split("/")[-1]}):
                    raise ValueError("gap_control_does_not_match_trace")
                matching = {o for o, pair in states.items() if pair[0] == target}
                if rows[cid]["type"] == "missing_screen" and matching:
                    raise ValueError("already_covered_check_mapping")
                source_ids = {o for o, pair in states.items() if pair[0] == source}
                if any(e["from_obs"] in source_ids and e["to_obs"] in matching and e.get("success") is True
                       and e.get("action_type") == "click" and (e.get("target") or {}).get("resource_id") == steps[-1]["selector"]["resource_id"].split("/")[-1]
                       for e in stg["edges"]):
                    raise ValueError("already_covered_edge")
                key = sha(json.dumps([source, steps[-1]["selector"], target], sort_keys=True).encode())
                assignments[cid] = ({"job_id": key[:24], "candidate_id": cid, "kind": rows[cid]["type"],
                    "reason": item["reason"], "risk": "read_only", "route": steps[:-1], "target": steps[-1],
                    "baseline_evidence": states[oid][1], "baseline_origins": stg["observations"][oid]["origins"],
                    "retry_no_feedback": rows[cid]["type"] == "no_reaction" and item.get("retry_no_feedback") is True}, "")
            except (KeyError, TypeError, ValueError, OSError) as exc:
                assignments[cid] = (None, str(exc))
    jobs = [job for job, _ in assignments.values() if job is not None]
    jobs.sort(key=lambda j: (PRIORITY[j["kind"]], len(j["route"]), j["job_id"], j["candidate_id"]))
    accepted = []
    for job in jobs:
        cid, jid = job["candidate_id"], job["job_id"]
        if jid in seen or len(accepted) >= MAX_JOBS:
            assignments[cid] = (None, "duplicate_goal" if jid in seen else "round_job_limit")
            continue
        accepted.append(job)
        seen.add(jid)
    ledger = []
    for cid, row in rows.items():
        job, reason = assignments.get(cid, (None, "independent_review_required"))
        decision = "executable" if job else "review"
        if not job and (row["type"] not in PRIORITY or reason.startswith(("unsafe", "already_covered"))
                        or reason in {"duplicate_goal", "round_job_limit"}):
            decision = "skip"
            if row["type"] not in PRIORITY:
                reason = "mapping_or_correctness_diagnostic"
        ledger.append({"candidate_id": cid, "decision": decision, "reason": reason, "finding": row})
    return {"format": FORMAT, "schema_version": 1, "package": package, "version": list(version),
            "environment": environment, "bindings": bindings,
            "limits": {"jobs": MAX_JOBS, "actions": MAX_ACTIONS, "seconds": MAX_SECONDS},
            "jobs": accepted, "counts": dict(Counter(row["decision"] for row in ledger)), "candidates": ledger}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("stg", "report", "gaps", "version-evidence", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[2] / "research-explore")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.research_root.resolve()))
    plan = compile_plan(args.stg, args.report, args.gaps, args.version_evidence, args.environment,
                        review_path=args.review)
    write_new(args.output, plan)
    print(json.dumps(plan["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
