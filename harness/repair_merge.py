"""Publish baseline topology plus verified repair evidence using the existing STG merger."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path

from .repair import Frame, action_point, bind, guard, same_state, sha, write_new
from .repair_plan import baseline_guard_ids, frame_of, read
from .repair_runner import load_plan


def merge_repair(plan_path, run_dir, baseline_path, output):
    from gym_explore.evaluation.stg import load_stg
    from gym_explore.graph_import import load_graph
    from gym_explore.topology import write_topology

    header = read(plan_path)
    package = header["package"]
    plan = load_plan(plan_path, package, header["environment"])
    baseline_path = baseline_path.resolve()
    baseline_file = baseline_path / "topology/data.json"
    if (baseline_path / "merge_report.json").exists() or not baseline_file.is_file():
        raise ValueError("repair_merge_requires_framework_topology_without_semantic_merge_report")
    if bind(baseline_file) != plan["bindings"]["baseline"]:
        raise ValueError("baseline_topology_changed")
    result = read(run_dir / "repair_results.json")
    graph_path = (run_dir / "app_graph.json").resolve()
    if (result.get("schema_version") != 1 or result["package"] != package
            or result["plan_sha256"] != sha(plan_path.read_bytes()) or result["graph_sha256"] != bind(graph_path)["sha256"]):
        raise ValueError("repair_graph_not_bound_to_results")
    graph = read(graph_path)
    if graph.get("app_package") != package:
        raise ValueError("repair_graph_package_mismatch")
    results = {r["edge_id"]: r for r in result["jobs"] if r["status"] == "sampled"}
    jobs = {j["job_id"]: j for j in plan["jobs"]}
    if not results:
        raise ValueError("no_verified_repair_edges")
    if len(results) != len(graph["transitions"]):
        raise ValueError("nonverified_or_duplicate_repair_edge")
    for e in graph["transitions"]:
        row = results.get(e["edge_id"])
        if (not row or e.get("success") is not True or e["repair_job"] not in jobs
                or row["job_id"] != e["repair_job"] or (e["from_page"], e["to_page"]) != (row["from"], row["to"])):
            raise ValueError("repair_transition_not_verified")

    before = load_stg(baseline_path)
    if before["package"] != package or {
            f"source:{k}": bind(Path(p)) for k, p in before["sources"].items()} != {
                k: v for k, v in plan["bindings"].items() if k.startswith("source:")}:
        raise ValueError("baseline_sources_changed")
    by_origin = {}
    by_obs = {}
    for nid, node in before["nodes"].items():
        for oid in node["members"]:
            if by_obs.setdefault(oid, nid) != nid:
                raise ValueError("baseline_mapping_ambiguous")
            for o in before["observations"][oid]["origins"]:
                by_origin[(o["source"], str(o["page_id"]))] = nid
    paths, mapping, states = [], {}, defaultdict(set)
    for i, (sid, path) in enumerate(before["sources"].items()):
        path = Path(path)
        paths.append(path)
        for raw_id in load_graph(path)["pages"]:
            nid = by_origin.get((sid, str(raw_id)))
            if not nid:
                raise ValueError("baseline_raw_page_has_no_mapping")
            mapping[(f"graph_{i:03}", raw_id)] = {"id": nid, "state": {}, "page_name": before["nodes"][nid]["name"]}
    for oid, (state, _) in baseline_guard_ids(before, package).items():
        if oid in by_obs:
            states[json.dumps(state, sort_keys=True)].add(by_obs[oid])
    source = f"graph_{len(paths):03}"
    resolved, frames = {}, {}
    for pid, page in graph["pages"].items():
        def verified(field):
            value = page[field]
            p = (graph_path.parent / value).resolve()
            if not p.is_relative_to(graph_path.parent) or bind(p)["sha256"] != result["asset_sha256"].get(value):
                raise ValueError("repair_asset_missing_or_modified")
            return p.read_bytes()

        frames[pid] = Frame(page["activity"], verified("screenshot_path"), verified("hierarchy_path"))
        state = guard(frames[pid], package)
        if not state:
            raise ValueError("repair_endpoint_without_real_hierarchy")
        key = json.dumps(state, sort_keys=True)
        matches = states[key]
        if len(matches) > 1:
            raise ValueError("repair_endpoint_baseline_identity_ambiguous")
        nid = next(iter(matches), "repair_state_" + sha(key.encode())[:24])
        if not matches and nid in before["nodes"]:
            raise ValueError("repair_node_id_collision")
        resolved[pid] = nid
        mapping[(source, pid)] = {"id": nid, "state": {}, "page_name":
            before["nodes"].get(nid, {}).get("name") or page["activity"].split("/")[-1]}
    for e in graph["transitions"]:
        job = jobs[e["repair_job"]]
        a, b = e["from_page"], e["to_page"]
        target = job["target"]
        action = e["action_details"]
        origin_nodes = {by_origin.get((o["source"], str(o["page_id"]))) for o in job["baseline_origins"]}
        if origin_nodes != {resolved[a]}:
            raise ValueError("repair_source_not_bound_to_baseline")
        if (not same_state(frames[a], frame_of(target["before"]), package)
                or not same_state(frames[b], frame_of(target["after"]), package)
                or action.get("action_type") != "click"
                or action_point(frames[a], target["selector"], package) != (action.get("tap_x"), action.get("tap_y"))
                or any(e[key] != graph["pages"][pid][field] for pid, key, field in
                       ((a, "before_screenshot", "screenshot_path"), (b, "after_screenshot", "screenshot_path"),
                        (a, "before_hierarchy_path", "hierarchy_path"), (b, "after_hierarchy_path", "hierarchy_path")))):
            raise ValueError("repair_edge_evidence_does_not_confirm_goal")
    paths.append(graph_path)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "run_manifest.json", {"initial_graphs": [str(p.resolve()) for p in paths]})
    write_topology(paths, output, app_package=package, state_mapping=mapping)
    after = load_stg(output)

    def edge_keys(stg):
        return Counter((e["id"], e["from"], e["to"], e["action_type"], e.get("success")) for e in stg["edges"])

    before_remap = {sid: f"graph_{i:03}" for i, sid in enumerate(before["sources"])}
    required = Counter((before_remap[e["source"]] + ":" + e["id"].split(":", 1)[1], e["from"], e["to"],
                        e["action_type"], e.get("success")) for e in before["edges"])
    actual = edge_keys(after)
    if required - actual:
        raise ValueError("baseline_edges_lost_during_merge")
    if any((f"{source}:{e['edge_id']}", resolved[e["from_page"]], resolved[e["to_page"]], "click", True)
           not in actual for e in graph["transitions"]):
        raise ValueError("repair_endpoint_or_edge_missing_after_merge")
    receipt = {"package": package, "preserved_baseline_edges": sum(required.values()),
               "verified_added_edges": len(graph["transitions"]), "before_nodes": len(before["nodes"]),
               "after_nodes": len(after["nodes"]), "repair_graph_sha256": result["graph_sha256"],
               "topology": bind(output / "topology/data.json")}
    write_new(output / "repair_merge.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "repair-run", "baseline", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[2] / "research-explore")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.research_root.resolve()))
    print(json.dumps(merge_repair(args.plan, args.repair_run, args.baseline, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
