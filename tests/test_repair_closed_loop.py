"""Independent Verify-A / synthetic device / original merger+eval / frozen Verify-B."""
import io
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research-explore"))
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "auto-explore"))

from harness.repair import Frame, bind, write_new
from harness.repair_merge import merge_repair
from harness.repair_plan import compile_plan
from harness.repair_runner import AdbRepairDevice, run_repairs
from gym_explore.evaluation.evaluate import evaluate
from gym_explore.evaluation.srm import add_screen, empty_srm
from gym_explore.evaluation.stg import load_stg
from gym_explore.evaluation.visible_coverage import evaluate_visible
from gym_explore.topology import write_topology

PACKAGE = "test.fixture"
ENV = "synthetic-test-only"
IDS = {"home": ("open_details", "Open details"), "detail": ("show_filters", "Show filters"),
       "overlay": ("filter_title", "Filter options")}


def frame(state, color, *, foreign=False):
    rid, label = IDS[state]
    png = io.BytesIO()
    Image.new("RGB", (100, 180), color).save(png, format="PNG")
    activity = ".HomeActivity" if state == "home" else ".DetailsActivity"
    xml = (f'<hierarchy><node package="{PACKAGE}" resource-id="{PACKAGE}:id/{rid}" text="{label}"'
           f' class="android.widget.Button" clickable="{str(state != "overlay").lower()}" enabled="true"'
           f' bounds="[4,12][30,36]"/>'
           + ('<node package="other.app" text="External"/>' if foreign else '') + '</hierarchy>').encode()
    return Frame(f"{PACKAGE}/{activity}", png.getvalue(), xml)


def make_trace(path, color, *, complete=True):
    path.mkdir()
    states = list(IDS) if complete else ["home", "detail"]
    graph = {"app_package": PACKAGE, "root_page_id": "home", "pages": {}, "transitions": [],
             "task_runs": [{"task_id": "fixture", "start_page_id": "home", "reset_policy": "restart",
                            "page_visit_sequence": states}]}
    for i, state in enumerate(states):
        f = frame(state, (color, i*45, 90))
        shot, xml = state+".png", state+".xml"
        (path/shot).write_bytes(f.png)
        (path/xml).write_bytes(f.xml)
        graph["pages"][state] = {"activity": f.activity, "screenshot_path": shot, "hierarchy_path": xml}
    for a, b in zip(states, states[1:]):
        graph["transitions"].append({"edge_id": a+"-"+b, "from_page": a, "to_page": b, "success": True,
            "action_details": {"action_type": "click", "tap_x": 17, "tap_y": 24},
            "before_screenshot": a+".png", "after_screenshot": b+".png",
            "before_hierarchy_path": a+".xml", "after_hierarchy_path": b+".xml"})
    graph["task_runs"][0]["edge_ids"] = [edge["edge_id"] for edge in graph["transitions"]]
    write_new(path/"app_graph.json", graph)
    (path/"package.txt").write_text("versionName=1.0\nversionCode=1\n", encoding="utf-8")
    return path/"app_graph.json"


def selector(state):
    rid, label = IDS[state]
    return {"resource_id": f"{PACKAGE}:id/{rid}", "label": label}


class Device:
    def __init__(self, *, outcome="overlay", version=("1.0", 1)):
        self.state = "home"
        self.actions = []
        self.outcome, self.version = outcome, version

    def preflight(self):
        return self.version

    def capture(self):
        return frame(self.state, (190, list(IDS).index(self.state)*45, 90))

    def tap(self, x, y):
        assert (x, y) == (17, 24)
        self.actions.append(("tap", self.state))
        self.state = "detail" if self.state == "home" else self.outcome
        return True

    def prepare_restart(self):
        return f"{PACKAGE}/.HomeActivity"

    def stop(self):
        self.actions.append(("stop", self.state))

    def launch(self, _):
        self.actions.append(("launch", self.state))
        self.state = "home"


def setup(output):
    output.mkdir()
    raw = make_trace(output/"old_source", 30, complete=False)
    independent_a = make_trace(output/"verify_a", 70)
    independent_b = make_trace(output/"verify_b", 130)
    baseline = output/"baseline"
    baseline.mkdir()
    write_new(baseline/"run_manifest.json", {"initial_graphs": [str(raw)]})
    write_topology([raw], baseline)
    srm = empty_srm(PACKAGE, "fixture", {})
    for state, (rid, label) in IDS.items():
        kind = "dialog" if state == "overlay" else "activity"
        activity = (PACKAGE + (".HomeActivity" if state == "home" else ".DetailsActivity")) if kind == "activity" else ""
        add_screen(srm, state, kind=kind, activity=activity, labels=[label], resource_ids=[rid],
                   evidence=["layout:fixture_"+rid], source="fixture")
        if state != "overlay":
            srm["controls"].append({"screen_id": state, "control_id": rid, "label": label,
                                    "resource_id": rid, "source": "layout"})
    srm_path = output/"srm.json"
    write_new(srm_path, srm)
    report_path, gaps = (output/"before"/name for name in ("stg_quality_report.json", "closure_gaps.json"))
    before = evaluate(srm_path, baseline, report_path.parent)
    ver = raw.parent/"package.txt"
    args = baseline, report_path, gaps, ver, ENV
    dry = compile_plan(*args)
    assert not dry["jobs"]
    missing = next(c for c in dry["candidates"] if c["finding"].get("screen") == "overlay")
    detail = next(k for k, o in load_stg(baseline)["observations"].items() if "Details" in o["activity"])
    def step(edge, state, intent):
        return {"edge_id": edge, "intent": intent, "selector": selector(state)}
    case = {"candidate_id": missing["candidate_id"], "verdict": "verified_missing",
            "reason": "Known overlay and transition intentionally left out of baseline fixture.",
            "baseline_observation": detail,
            "route": [step("home-detail", "home", "read_only_navigation")],
            "target": step("detail-overlay", "detail", "read_only_expansion")}
    refs = dry["bindings"]
    review = {"format": "stg_repair_review", "schema_version": 1, "package": PACKAGE, "environment": ENV,
              "version": ["1.0", 1], "trace_id": "verify-a", "reviewer": "fixture-oracle",
              "trace": bind(independent_a), "version_evidence": bind(independent_a.parent/"package.txt"),
              "asset_sha256": {p.name: bind(p)["sha256"] for p in independent_a.parent.iterdir()
                               if p.suffix in {".png", ".xml"}},
              "source_sha256": {k: v["sha256"] for k, v in refs.items() if k.startswith("source:")},
              **{f"{k}_sha256": refs[k]["sha256"] for k in ("report", "gaps", "baseline")},
              "cases": [case]}
    review_path = output/"verify_A_review.json"
    write_new(review_path, review)
    plan_path = output/"repair_plan.json"
    plan = compile_plan(*args, review_path=review_path)
    assert len(plan["jobs"]) == 1, [c for c in plan["candidates"] if c["candidate_id"] == case["candidate_id"]]
    write_new(plan_path, plan)
    # Freeze B before execution. Only verifier consumes it; its assets aren't in the plan.
    catalog = {"schema_version": 1, "package": PACKAGE, "version": "1.0", "version_code": 1, "trace_id": "verify-b",
        "scope": "fixture detail and one filter overlay", "trace": bind(independent_b), "reference_version": bind(independent_b.parent/"package.txt"),
        "capture_support": [], "surfaces": [
            {"id": s, "status": "verified", "entry": "fixture UI", "required": [IDS[s][1]],
             "screenshot": bind(independent_b.parent/(s+".png")), "hierarchy": bind(independent_b.parent/(s+".xml"))}
            for s in ("detail", "overlay")],
        "paths": [{"id": "filters", "status": "verified", "steps": [
            {"from": "detail", "to": "overlay", "action": "click",
             "target": {"resource_id": "show_filters", "label": "Show filters"}}]}]}
    frozen = output/"verify_B_frozen.json"
    write_new(frozen, catalog)
    assert str(independent_b) not in plan_path.read_text(encoding="utf8")
    return baseline, before, srm_path, plan_path, frozen, ver


def validate_b(output, phase, topology, frozen, ver, *, repair=False):
    stg = load_stg(topology)
    raw = {k: Path(v) for k, v in stg["sources"].items()}
    def witness(sid, pid):
        base = raw[sid].parent
        graph = json.loads(raw[sid].read_bytes())
        page = graph["pages"][pid]
        return {"source": sid, "page_id": pid, "mode": "visual_and_tree",
                "screenshot_sha256": bind(base/page["screenshot_path"])["sha256"],
                "hierarchy_sha256": bind(base/page["hierarchy_path"])["sha256"]}
    detail = [witness("graph_000", "detail")]
    overlay = []
    edges = []
    if repair:
        e = next(e for e in stg["edges"] if e["id"].endswith(":repair_0000"))
        graph = json.loads(raw[e["source"]].read_bytes())
        t, = graph["transitions"]
        detail.append(witness(e["source"], t["from_page"]))
        overlay.append(witness(e["source"], t["to_page"]))
        edges.append(e["id"])
    review = {"schema_version": 1, "reference_sha256": bind(frozen)["sha256"],
              "stg_artifact": bind(topology/"topology/data.json"),
              "independence_note": "Separate fixture captures with independent image bytes; not passed to planner or merger.",
              "source_bindings": {k: {"graph": bind(v), "version": bind(ver)} for k, v in raw.items()},
              "surfaces": [{"id": sid, "result": "covered" if hits else "missing", "reason": "fixture oracle", "witnesses": hits}
                           for sid, hits in (("detail", detail), ("overlay", overlay))],
              "paths": [{"id": "filters", "result": "covered" if edges else "missing", "reason": "fixture oracle", "edge_ids": edges}]}
    review_path = output/f"verify_B_{phase}_review.json"
    write_new(review_path, review)
    r = evaluate_visible(frozen, review_path, topology)
    write_new(output/f"verify_B_{phase}_result.json", r)
    return r


def demo(output):
    baseline, before, srm, plan, frozen, ver = setup(output)
    old_b = validate_b(output, "before", baseline, frozen, ver)
    run_dir = output/"auto_run"
    device = Device()
    result = run_repairs(plan, run_dir, PACKAGE, ENV, lambda _: device)
    assert [r["status"] for r in result["jobs"]] == ["sampled"]
    assert device.actions == [("tap", "home"), ("tap", "detail")] and result["device_actions"] == 2
    after_stg = output/"merged"
    merge = merge_repair(plan, run_dir, baseline, after_stg)
    assert (merge["preserved_baseline_edges"], merge["verified_added_edges"], merge["before_nodes"], merge["after_nodes"]) == (1, 1, 2, 3)
    after = evaluate(srm, after_stg, output/"after")
    new_b = validate_b(output, "after", after_stg, frozen, ver, repair=True)
    assert [x["primary"]["numerator"] for x in (old_b, new_b)] == [1, 2]
    assert [x["secondary"]["numerator"] for x in (old_b, new_b)] == [0, 1]
    assert old_b["primary"]["denominator"] == new_b["primary"]["denominator"] == 2
    assert old_b["secondary"]["denominator"] == new_b["secondary"]["denominator"] == 1
    write_new(output/"comparison.json", {"mode": "synthetic fixture, not a real-device result", "merge": merge,
        "device_actions": result["device_actions"], "before": before["headline"], "after": after["headline"],
        "independent_before": {k: old_b[k] for k in ("primary", "secondary")},
        "independent_after": {k: new_b[k] for k in ("primary", "secondary")}})


def test_closed_loop(tmp_path):
    demo(tmp_path/"demo")


def test_wrong_version_does_not_tap_and_wrong_branch_is_not_exported(tmp_path):
    _, _, _, plan, _, _ = setup(tmp_path/"inputs")
    for name, device, expected in (("version", Device(version=("2.0", 2)), "not_run"),
                                    ("branch", Device(outcome="home"), "different_branch")):
        out = tmp_path/name
        result = run_repairs(plan, out, PACKAGE, ENV, lambda _: device)
        assert result["jobs"][0]["status"] == expected
        assert not json.loads((out/"app_graph.json").read_bytes())["transitions"]
        if name == "version":
            assert device.actions == []
    with pytest.raises(FileExistsError):
        run_repairs(plan, tmp_path/"version", PACKAGE, ENV, lambda _: pytest.fail("existing output must not touch device"))


def test_budget_includes_restart_and_never_exports_recovery_as_target(tmp_path):
    _, _, _, plan, _, _ = setup(tmp_path/"inputs")
    obj = json.loads(plan.read_bytes())
    obj["limits"]["actions"] = 3
    limited = tmp_path/"limited.json"
    write_new(limited, obj)
    device = Device()
    device.state = "overlay"
    out = tmp_path/"limited"
    result = run_repairs(limited, out, PACKAGE, ENV, lambda _: device)
    assert device.actions == [("stop", "overlay"), ("launch", "overlay"), ("tap", "home")]
    assert result["device_actions"] == 3 and result["termination"] == "budget_exhausted"
    assert not json.loads((out/"app_graph.json").read_bytes())["transitions"]


def test_stale_and_cross_package_fail_closed(tmp_path):
    _, _, _, plan, _, _ = setup(tmp_path/"inputs")
    for how in ("stale", "outside"):
        class Changed(Device):
            def capture(self):
                f = super().capture()
                if how == "stale" and self.state == "detail" and len(self.actions) == 1:
                    self.calls = getattr(self, "calls", 0) + 1
                    if self.calls > 1:
                        return frame("detail", (240, 0, 0))
                if how == "outside" and self.state == "overlay":
                    return Frame("other.app/.Overlay", f.png, f.xml)
                return f
        d = Changed()
        out = tmp_path/how
        result = run_repairs(plan, out, PACKAGE, ENV, lambda _: d)
        assert result["jobs"][0]["status"] == ("safety_skip" if how == "stale" else "out_of_app")
        assert not json.loads((out/"app_graph.json").read_bytes())["transitions"]
        assert result["device_actions"] == (1 if how == "stale" else 2)


def test_modified_review_evidence_cannot_trigger_actions(tmp_path):
    _, _, _, plan, _, _ = setup(tmp_path/"inputs")
    p = plan.parent/"verify_a"/"overlay.png"
    p.write_bytes(b"not same reference")
    with pytest.raises(ValueError, match="hash_mismatch"):
        run_repairs(plan, tmp_path/"run", PACKAGE, ENV, lambda _: pytest.fail("tampered evidence must not touch device"))
    obj = json.loads(plan.read_bytes())
    obj["jobs"][0]["target"]["after"]["png"] = bind(p)
    rebound = tmp_path/"rebound.json"
    write_new(rebound, obj)
    with pytest.raises(ValueError, match="review_asset_hash_mismatch"):
        run_repairs(rebound, tmp_path/"run", PACKAGE, ENV, lambda _: pytest.fail("a plan cannot re-pin reviewed evidence"))


def test_adb_tree_captured_during_pixel_change_is_only_visual(monkeypatch):
    device = AdbRepairDevice("adb", "fixture-device", PACKAGE, None)
    first, last = (frame("detail", (n, 55, 90)) for n in (30, 100))
    monkeypatch.setattr(device, "active", lambda: first.activity)
    calls = iter([first.png, b"notice" + first.xml, last.png])
    monkeypatch.setattr(device, "adb_call", lambda *_: next(calls))
    assert device.capture() == Frame(first.activity, last.png)


def test_merge_rejects_modified_live_evidence(tmp_path):
    baseline, _, _, plan, _, _ = setup(tmp_path/"inputs")
    run = tmp_path/"run"
    run_repairs(plan, run, PACKAGE, ENV, lambda _: Device())
    graph = json.loads((run/"app_graph.json").read_bytes())
    row = next(iter(graph["pages"].values()))
    (run/row["hierarchy_path"]).write_bytes(b"<hierarchy />")
    with pytest.raises(ValueError, match="repair_asset_missing_or_modified"):
        merge_repair(plan, run, baseline, tmp_path/"merged")
    assert not (tmp_path/"merged").exists()


if __name__ == "__main__":
    demo(Path(sys.argv[1]).resolve())
