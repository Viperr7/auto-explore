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
MISSING_TEMPLATE_KINDS = frozenset({
    "missing_core_template",
    "missing_srm_screen_1hop",
    "missing_srm_screen_remote",
})
GENERIC_CLUE_TERMS = frozenset({
    "返回", "取消", "确定", "确认", "关闭", "完成", "保存", "提交", "删除", "编辑", "更多",
    "全部", "查看", "打开", "点击", "进入", "下一步", "上一步", "跳过", "重试", "刷新",
    "首页", "我的", "设置", "详情", "列表", "搜索", "分享", "back", "cancel", "ok", "more",
})
GENERIC_HOST_SHORT = frozenset({
    "mainactivity", "tabmainactivity", "hostactivity", "containeractivity",
    "maincomposeactivity", "composeactivity", "browseractivity", "androidwebactivity",
    "webactivity", "webviewactivity", "commonwebactivity", "hybridactivity", "h5activity",
    "flutteractivity", "reactactivity",
})


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


def _short_activity(activity: str) -> str:
    keys = _activity_keys(activity)
    return keys[-1] if keys else ""


class RepairClues:
    """Bounded, priority-guided exploration hints derived from STG evaluation, with live hit decay."""

    def __init__(self, payload: dict, source: Path):
        self.source = str(source)
        self.package = payload["package"]
        self.source_mode = str(payload.get("source_mode") or "legacy")
        self.summary = dict(payload.get("summary") or {})
        self.targets = list(payload.get("targets") or [])
        self.activity_scores: dict[str, int] = {}
        self.activity_targets: dict[str, list[dict]] = defaultdict(list)
        self.global_missing: list[dict] = []
        self.target_activity_index: dict[str, list[str]] = defaultdict(list)
        self.target_by_id: dict[str, dict] = {}
        self.hit_targets: set[str] = set()
        self.hit_titles: list[str] = []
        self.covered_core_names: list[str] = []
        self.covered_core_terms: dict[str, str] = {}
        self.covered_core_activities: set[str] = set()

        for row in payload.get("prioritized_activities") or []:
            score = int(row.get("score") or 0)
            for key in _activity_keys(row.get("activity", "")):
                self.activity_scores[key] = max(self.activity_scores.get(key, 0), score)

        missing_terms_reserved: set[str] = set()
        missing_acts_reserved: set[str] = set()
        for target in self.targets:
            tid = str(target.get("target_id") or "")
            if tid:
                self.target_by_id[tid] = target
            act = target.get("activity", "")
            for key in _activity_keys(act):
                self.activity_targets[key].append(target)
                missing_acts_reserved.add(key)
            if target.get("kind") in MISSING_TEMPLATE_KINDS:
                self.global_missing.append(target)
                for raw_term in [target.get("target_title"), target.get("canonical_name"), *(target.get("expected_labels") or [])]:
                    nt = _norm(raw_term)
                    if nt:
                        missing_terms_reserved.add(nt)
                t_acts = list(target.get("target_activities") or [])
                if target.get("target_activity"):
                    t_acts.append(target["target_activity"])
                for ta in t_acts:
                    short = _short_activity(ta)
                    if short and short not in GENERIC_HOST_SHORT:
                        for key in _activity_keys(ta):
                            missing_acts_reserved.add(key)
                            if tid and tid not in self.target_activity_index[key]:
                                self.target_activity_index[key].append(tid)

        for entry in payload.get("covered_core_templates") or []:
            if not isinstance(entry, dict):
                continue
            cname = str(entry.get("name") or "").strip()
            if cname and cname not in self.covered_core_names:
                self.covered_core_names.append(cname)
            for raw_term in [cname, *(entry.get("labels") or [])]:
                nt = _norm(raw_term)
                if (
                    len(nt) >= 2
                    and nt not in GENERIC_CLUE_TERMS
                    and nt not in missing_terms_reserved
                    and not any(nt in mt or mt in nt for mt in missing_terms_reserved if len(mt) >= 2)
                ):
                    self.covered_core_terms.setdefault(nt, cname or str(raw_term))
            for act in entry.get("activities") or []:
                short = _short_activity(act)
                if short and short not in GENERIC_HOST_SHORT:
                    keys = _activity_keys(act)
                    if not any(k in missing_acts_reserved for k in keys):
                        self.covered_core_activities.update(keys)

        kind_rank = {"missing_core_template": 0, "missing_srm_screen_1hop": 1, "missing_srm_screen_remote": 2}
        self.global_missing.sort(
            key=lambda t: (
                int(t.get("priority", 9)),
                kind_rank.get(str(t.get("kind", "")), 9),
                -int(t.get("member_count", 0)),
                str(t.get("target_id", "")),
            )
        )

    def _targets_for(self, activity: str, *, include_hit: bool = False) -> list[dict]:
        seen = set()
        found = []
        for key in _activity_keys(activity):
            for t in self.activity_targets.get(key, ()):
                tid = t.get("target_id")
                if tid in seen or (not include_hit and tid in self.hit_targets):
                    continue
                seen.add(tid)
                found.append(t)
        found.sort(key=lambda t: (int(t.get("priority", 9)), str(t.get("target_id", ""))))
        return found

    def mark_observed(self, activity: str, page_description: str = "",
                      content_hints: list[str] | tuple[str, ...] = ()) -> list[dict]:
        """Record a live screen observation and retire matched missing-template clues for this run."""
        newly_hit: list[dict] = []
        matched_ids: list[str] = []

        # 1. Exact non-generic Android Activity match against target_activities
        short = _short_activity(activity)
        if short and short not in GENERIC_HOST_SHORT:
            for key in _activity_keys(activity):
                for tid in self.target_activity_index.get(key, ()):
                    if tid not in self.hit_targets and tid not in matched_ids:
                        matched_ids.append(tid)

        # 2. Conservative semantic title match on observed page_description / content_hints (len >= 3)
        obs_blob = _norm(page_description)
        hint_norms = {_norm(h) for h in (content_hints or ()) if len(_norm(h)) >= 2}
        if obs_blob or hint_norms:
            for t in self.global_missing:
                tid = str(t.get("target_id") or "")
                if not tid or tid in self.hit_targets or tid in matched_ids:
                    continue
                title = _norm(t.get("target_title") or t.get("canonical_name") or "")
                if not title or title in GENERIC_CLUE_TERMS:
                    continue
                if (len(title) >= 3 and title in obs_blob) or title in hint_norms:
                    matched_ids.append(tid)

        for tid in matched_ids:
            self.hit_targets.add(tid)
            t = self.target_by_id.get(tid) or {}
            title = str(t.get("target_title") or t.get("canonical_name") or tid)
            if title not in self.hit_titles:
                self.hit_titles.append(title)
            newly_hit.append({"target_id": tid, "title": title, "kind": t.get("kind", "")})
        return newly_hit

    def progress(self) -> dict:
        total = len(self.global_missing)
        hit = sum(1 for t in self.global_missing if t.get("target_id") in self.hit_targets)
        return {
            "total_missing_targets": total,
            "hit_in_run": hit,
            "remaining_missing_targets": max(0, total - hit),
            "hit_titles": self.hit_titles[:20],
        }

    def activity_priority(self, activity: str) -> int:
        base = max((self.activity_scores.get(k, 0) for k in _activity_keys(activity)), default=0)
        if base <= 0:
            return 0
        all_local = self._targets_for(activity, include_hit=True)
        if all_local and all(t.get("target_id") in self.hit_targets for t in all_local):
            return 0
        return base

    def match_action(self, activity: str, action: ExplorationAction) -> tuple[int, list[str]]:
        """Score how well a live candidate action matches local OR global missing-core clues."""
        act_rid = str(action.control_key or "").rsplit("/", 1)[-1].casefold()
        act_text = _norm(action.element_text)
        act_desc = _norm(action.target_description)
        best_score = 0
        matches: list[str] = []
        seen_tids: set[str] = set()

        # 1. Local Activity-anchored clues (P0 closure defects + P1 1-hop targets)
        for t in self._targets_for(activity):
            tid = str(t.get("target_id") or "")
            seen_tids.add(tid)
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

        # 2. Global active missing core templates (covers zero-inbound-edge & multi-tab Fragment apps)
        if act_text or act_desc:
            for t in self.global_missing:
                tid = str(t.get("target_id") or "")
                if tid in self.hit_targets or tid in seen_tids:
                    continue
                kind = str(t.get("kind") or "")
                if kind not in {"missing_core_template", "missing_srm_screen_remote"}:
                    continue
                priority = int(t.get("priority", 2))
                base = max(1, 5 - priority)
                title = _norm(t.get("target_title") or t.get("canonical_name") or "")
                short_btn_in_title = bool(
                    act_text and 2 <= len(act_text) <= 4
                    and act_text not in GENERIC_CLUE_TERMS
                    and title and act_text in title
                )
                exact_title_hit = bool(
                    title and len(title) >= 2
                    and (
                        act_text == title
                        or (len(title) >= 3 and (title in act_text or title in act_desc))
                        or short_btn_in_title
                    )
                )
                label_terms = [
                    _norm(v) for v in (t.get("expected_labels") or [])[:6]
                    if len(_norm(v)) >= 2 and _norm(v) not in GENERIC_CLUE_TERMS
                ]
                label_hit = bool(
                    not exact_title_hit
                    and any(
                        (
                            term == act_text
                            or (len(term) >= 3 and term in act_text)
                            or (len(term) >= 3 and term in act_desc)
                            or (act_text and 2 <= len(act_text) <= 4 and act_text not in GENERIC_CLUE_TERMS and act_text in term)
                        )
                        for term in label_terms
                    )
                )
                if exact_title_hit or label_hit:
                    score = base * 2 if exact_title_hit else base
                    best_score = max(best_score, score)
                    matches.append(f"[P{priority}:core->{t.get('target_title')}] {t.get('guidance', '')}")
                    if len(matches) >= 3:
                        break
        # 3. Demote actions that only target already-covered baseline core templates (avoid list)
        if best_score == 0 and self.covered_core_terms and (act_text or act_desc):
            for term, cname in self.covered_core_terms.items():
                if (
                    act_text == term
                    or (len(term) >= 3 and (term in act_text or term in act_desc))
                ):
                    return -2, [f"[AVOID:already_covered->{cname}] 已在基线覆盖，低优先级"]
        return best_score, matches[:3]

    def is_covered_only_activity(self, activity: str) -> bool:
        keys = _activity_keys(activity)
        return bool(keys and any(k in self.covered_core_activities for k in keys))

    def page_priority(self, activity: str, pending_actions: list[ExplorationAction], depth: int) -> tuple[int, int, int, int, int]:
        """Higher tuple = higher BFS expansion priority for a pending page."""
        best_action_score = 0
        pos_hits = 0
        for act in pending_actions:
            sc, _ = self.match_action(activity, act)
            if sc > best_action_score:
                best_action_score = sc
            if sc > 0:
                pos_hits += 1
        act_prio = self.activity_priority(activity)
        covered_penalty = 1 if self.is_covered_only_activity(activity) and pos_hits == 0 else 0
        return (
            1 if best_action_score > 0 else 0,
            best_action_score,
            act_prio,
            -covered_penalty,
            -depth,
        )

    def screen_analysis_guidance(self, activity: str) -> str:
        """Compact Phase-1 guidance injected into analyze_screen so VLM extracts missing-core entries first."""
        active_missing = [t for t in self.global_missing if t.get("target_id") not in self.hit_targets]
        if not active_missing and not self.covered_core_names:
            return ""
        lines = ["\nREPAIR COVERAGE GUIDANCE (prioritize missing core features during element selection):"]
        if active_missing:
            missing_parts = []
            for t in active_missing[:14]:
                title = str(t.get("target_title") or t.get("canonical_name") or "").strip()
                kws = [str(k).strip() for k in (t.get("expected_labels") or [])[:3] if str(k).strip() and str(k).strip() != title]
                missing_parts.append(f"{title}({'/'.join(kws)})" if kws else title)
            lines.append("- PRIORITY MISSING CORE TEMPLATES TO HUNT: " + ", ".join(missing_parts))
        if self.covered_core_names:
            lines.append(
                "- ALREADY COVERED IN BASELINE (DEPRIORITIZE/SKIP unless it is a bottom nav tab or leads to a missing target): "
                + ", ".join(self.covered_core_names[:18])
            )
        lines.append(
            "- HUB-FIRST & DEDUP RULE: (1) ALWAYS include ALL visible bottom navigation tabs and top-level "
            "search/profile/settings/compose/message entries first; (2) NEVER pick more than ONE representative "
            "from homogeneous repeating groups (e.g. at most 1 meal slot from 早餐/午餐/晚餐/加餐, at most 1 feed card, "
            "at most 1 sibling category chip); (3) spend remaining slots ONLY on controls leading toward PRIORITY MISSING CORE TEMPLATES."
        )
        return "\n".join(lines)

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
        active_missing = [t for t in self.global_missing if t.get("target_id") not in self.hit_targets]
        missing_brief = []
        for t in active_missing[:15]:
            row = {
                "priority": f"P{t.get('priority', 1)}",
                "title": t.get("target_title", ""),
                "from": t.get("from_title", ""),
                "labels": (t.get("expected_labels") or [])[:5],
            }
            if t.get("target_purpose"):
                row["purpose"] = str(t["target_purpose"])[:60]
            missing_brief.append(row)
        return {
            "source": self.source,
            "source_mode": self.source_mode,
            "activity_priority_score": self.activity_priority(activity),
            "activity_clues": local_brief,
            "top_missing_screens": missing_brief,
            **({"covered_core_templates_avoid": self.covered_core_names[:20]} if self.covered_core_names else {}),
            "progress": self.progress(),
            "instruction": (
                "Prioritize untried controls and entry points carrying repair_clue_score > 0, matching activity_clues, "
                "or leading toward top_missing_screens (missing core product templates). Deprioritize actions marked "
                "repair_clue_avoid or leading only to covered_core_templates_avoid, while staying safe and grounded in the live screen."
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
