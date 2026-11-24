"""Fail-closed, offline-first execution of independently reviewed Android repairs."""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

FORMAT = "stg_repair_plan"
MAX_JOBS, MAX_ACTIONS, MAX_SECONDS = 5, 40, 1800
PACKAGE = re.compile(r"[A-Za-z][\w]*(?:\.[A-Za-z][\w]*)+", re.ASCII)
DANGER = re.compile(
    r"付款|支付|下单|购买|结算|拨打|电话|评论|发布|提交|授权|同意|保存|删除|退出登录|发送|转账|分享|收藏|点赞|关注|退款|开通|确认|安装|"
    r"pay|buy|purchase|checkout|order|dial|phone|comment|publish|submit|allow|agree|save|delete|logout|send|transfer|share|favorite|follow|refund|confirm|install",
    re.I)
STATE_ATTRS = ("package", "resource-id", "class", "text", "content-desc", "bounds",
               "clickable", "checkable", "checked", "selected", "enabled", "focused", "scrollable", "visible-to-user")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bind(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha(path.read_bytes())}


def pinned(binding: dict) -> bytes:
    data = Path(binding["path"]).read_bytes()
    if sha(data) != binding["sha256"]:
        raise ValueError("repair_input_hash_mismatch")
    return data


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def activity_name(activity: str, package: str) -> str:
    prefix, sep, component = activity.strip().partition("/")
    if not sep or prefix != package or not re.fullmatch(r"[.A-Za-z0-9_$]+", component, re.ASCII):
        return ""
    if component.startswith("."):
        component = package + component
    return prefix + "/" + component


def version_tuple(data: bytes) -> tuple[str, int]:
    text = data.decode("utf-8", errors="replace")
    names = set(re.findall(r"(?m)^\s*versionName=(\S+)", text))
    codes = set(re.findall(r"(?m)^\s*versionCode=(\d+)\b", text))
    if len(names) != 1 or len(codes) != 1:
        raise ValueError("unverified_app_version")
    return next(iter(names)), int(next(iter(codes)))


@dataclass(frozen=True)
class Frame:
    activity: str
    png: bytes
    xml: bytes = b""

    def size(self) -> tuple[int, int]:
        with Image.open(io.BytesIO(self.png)) as image:
            if image.format != "PNG":
                raise ValueError("not_a_png")
            image.load()
            return image.size

    def pixels(self) -> bytes:
        with Image.open(io.BytesIO(self.png)) as image:
            return image.convert("RGB").tobytes()


def ui_tree(frame: Frame, package: str):
    if not activity_name(frame.activity, package) or not frame.xml:
        return None
    try:
        root = ET.fromstring(frame.xml)
    except ET.ParseError:
        return None
    if root.tag != "hierarchy" or not any(n.get("package") == package for n in root.iter("node")):
        return None
    # UIAutomator may report system chrome, but never act through a foreign overlay.
    if any(n.get("package") not in (None, "", package) for n in root.iter("node")):
        return None
    return root


def guard(frame: Frame, package: str) -> dict | None:
    root = ui_tree(frame, package)
    if root is None:
        return None

    def structure(node):
        return ([node.get(k, "") for k in STATE_ATTRS], [structure(child) for child in node if child.tag == "node"])

    data = json.dumps(structure(root), ensure_ascii=False, separators=(",", ":")).encode()
    return {"activity": activity_name(frame.activity, package), "size": list(frame.size()), "tree_state": sha(data)}


def same_state(first: Frame, second: Frame, package: str) -> bool:
    state = guard(first, package)
    return state is not None and state == guard(second, package)


def safe_selector(selector: dict, package: str) -> bool:
    rid, label = selector.get("resource_id"), selector.get("label")
    return (isinstance(rid, str) and bool(re.fullmatch(re.escape(package) + r":id/[A-Za-z][\w]*", rid, re.ASCII))
            and isinstance(label, str) and 0 < len(label.strip()) <= 100 and not DANGER.search(rid + " " + label))


def action_point(frame: Frame, selector: dict, package: str):
    root = ui_tree(frame, package)
    if root is None or not safe_selector(selector, package):
        return None
    candidates = [n for n in root.iter("node") if n.get("resource-id") == selector["resource_id"]
                  and selector["label"] in (n.get("text"), n.get("content-desc"))]
    if len(candidates) != 1:
        return None
    n = candidates[0]
    if (n.get("package") != package or n.get("clickable") != "true" or n.get("enabled") != "true"
            or n.get("checkable") == "true" or n.get("visible-to-user") == "false"
            or DANGER.search((n.get("text") or "") + " " + (n.get("content-desc") or ""))):
        return None

    def bounds(node):
        m = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.get("bounds", ""))
        return tuple(map(int, m.groups())) if m else None

    if not (box := bounds(n)):
        return None
    x1, y1, x2, y2 = box
    w, h = frame.size()
    if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        # A distinct touch target over the grounded point can intercept the tap.
        for other in root.iter("node"):
            if (other is n or other.get("clickable") != "true" or other.get("enabled") == "false"
                    or other.get("visible-to-user") == "false" or not (b := bounds(other))):
                continue
            if b[0] <= x < b[2] and b[1] <= y < b[3]:
                if n not in other.iter("node") or DANGER.search(" ".join(other.get(k, "") for k in
                                              ("resource-id", "text", "content-desc"))):
                    return None
        return x, y
    return None


class Budget:
    def __init__(self, actions=MAX_ACTIONS, seconds=MAX_SECONDS, *, clock=time.monotonic):
        self.clock = clock
        self.deadline = clock() + min(seconds, MAX_SECONDS)
        self.limit = min(actions, MAX_ACTIONS)
        self.used = 0

    def remaining(self) -> float:
        seconds = self.deadline - self.clock()
        if seconds <= 0:
            raise TimeoutError("wall_clock_budget")
        return seconds

    def invoke(self, fn, *args):
        self.remaining()
        if self.used >= self.limit:
            raise TimeoutError("device_action_budget")
        self.used += 1
        return fn(*args)
