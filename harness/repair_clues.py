"""Load STG-eval repair clues to prioritize gaps during Auto Explore runs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

from .contracts import ExplorationAction

FORMAT = "stg_eval_repair_clues"
SCHEMA_VERSION = 1


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _activity_keys(activity: str) -> tuple[str, ...]:
    raw = str(activity or "").strip()
    if not raw:
        return ()
    pkg, sep, cls = raw.partition("/")
    full = (pkg + cls if cls.startswith(".") else cls) if sep else pkg
    short = full.rsplit(".", 1)[-1]
    return tuple(dict.fromkeys(k for k in (raw.casefold(), full.casefold(), short.casefold()) if k))


class RepairClues:
    """Bounded, read-only exploration priority hints derived from STG evaluation."""

    def __init__(self, payload: dict, source: Path):
        self.source = str(source)
        self.package = payload["package"]
        self.summary = dict(payload.get("summary") or {})
        self.targets = list(payload.get("targets") or [])
        self.activity_scores: dict[str, int] = {}
        self.activity_targets: dict[str, list[dict]] = defaultdict(list)
        self.global_missing: list[dict] = []

        for row in payload.get("prioritized_activities") or []:
            score = int(row.get("score") or 0)
            for key in _activity_keys(row.get("activity", "")):
                self.activity_scores[key] = max(self.activity_scores.get(key, 0), score)

        for target in self.targets:
            act = target.get("activity", "")
            for key in _activity_keys(act):
                self.activity_targets[key].append(target)
            if target.get("kind") in {"missing_srm_screen_1hop", "missing_srm_screen_remote"}:
                self.global_missing.append(target)

    def _targets_for(self, activity: str) -> list[dict]:
        seen = set()
        found = []
        for key in _activity_keys(activity):
            for t in self.activity_targets.get(key, ()):
                tid = t.get("target_id")
                if tid not in seen:
                    seen.add(tid)
                    found.append(t)
        found.sort(key=lambda t: (int(t.get("priority", 9)), str(t.get("target_id", ""))))
        return found

    def activity_priority(self, activity: str) -> int:
        return max((self.activity_scores.get(k, 0) for k in _activity_keys(activity)), default=0)

    def match_action(self, activity: str, action: ExplorationAction) -> tuple[int, list[str]]:
        """Score how well a live candidate action matches an STG evaluation repair clue."""
        local = self._targets_for(activity)
        if not local:
            return 0, []
        act_rid = str(action.control_key or "").rsplit("/", 1)[-1].casefold()
        act_text = _norm(action.element_text)
        act_desc = _norm(action.target_description)
        best_score = 0
        matches: list[str] = []

        for t in local:
            priority = int(t.get("priority", 3))
            base = max(1, 5 - priority)
            kind = t.get("kind", "")
            if kind == "closure_control_gap":
                ctl = t.get("control") or {}
                ctl_rid = str(ctl.get("resource_id") or "").rsplit("/", 1)[-1].casefold()
                ctl_label = _norm(ctl.get("label") or ctl.get("sample") or "")
                rid_hit = bool(ctl_rid and act_rid and ctl_rid == act_rid)
                label_hit = bool(
                    ctl_label and len(ctl_label) >= 2 and ctl_label != "列表项"
                    and (ctl_label == act_text or ctl_label in act_text or ctl_label in act_desc)
                )
                if rid_hit or label_hit:
                    score = base * (3 if rid_hit else 2)
                    best_score = max(best_score, score)
                    matches.append(f"[P{priority}:{t.get('gap_type')}] {t.get('guidance', ctl_label or ctl_rid)}")
            elif kind == "missing_srm_screen_1hop":
                title = _norm(t.get("target_title") or "")
                labels = [_norm(v) for v in (t.get("expected_labels") or []) if len(_norm(v)) >= 2]
                terms = [x for x in [title, *labels[:5]] if len(x) >= 2]
                if any(term in act_text or term in act_desc for term in terms):
                    score = base * 2
                    best_score = max(best_score, score)
                    matches.append(f"[P{priority}:1hop->{t.get('target_title')}] {t.get('guidance', '')}")
        return best_score, matches[:3]

    def prompt_context(self, activity: str, page_description: str = "",
                       actions: list[ExplorationAction] | None = None) -> dict:
        """Compact bounded clue context injected into VLM action selection."""
        local = self._targets_for(activity)[:8]
        local_brief = []
        for t in local:
            item = {
                "priority": f"P{t.get('priority', 3)}",
                "kind": t.get("kind"),
                "guidance": str(t.get("guidance", ""))[:160],
            }
            if t.get("control"):
                ctl = t["control"]
                item["control"] = {k: ctl.get(k) for k in ("label", "resource_id", "class") if ctl.get(k)}
            if t.get("target_title"):
                item["target_title"] = t["target_title"]
                item["expected_labels"] = (t.get("expected_labels") or [])[:5]
            local_brief.append(item)
        missing_brief = [
            {
                "priority": f"P{t.get('priority', 3)}",
                "title": t.get("target_title", ""),
                "from": t.get("from_title", ""),
                "labels": (t.get("expected_labels") or [])[:4],
            }
            for t in self.global_missing[:6]
        ]
        return {
            "source": self.source,
            "activity_priority_score": self.activity_priority(activity),
            "activity_clues": local_brief,
            "top_missing_screens": missing_brief,
            "instruction": (
                "Prioritize untried controls and entry points matching activity_clues "
                "or leading toward top_missing_screens, while staying safe and grounded in the live screen."
            ),
        }


def load_repair_clues(path: str | Path, package: str) -> RepairClues:
    source = Path(path).resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read repair clues at {source}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported repair clues format/version")
    if payload.get("package") != package:
        raise ValueError(f"Repair clues package {payload.get('package')!r} differs from target App {package!r}")
    if not isinstance(payload.get("targets"), list):
        raise ValueError("Repair clues missing targets list")
    return RepairClues(payload, source)
