#!/usr/bin/env python3
"""
Adapters: textual observations and action parsing for LLM policies.

We format observations with a numbered action menu. The environment uses
the same menu to interpret chosen indices. This keeps the policy glue simple.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def format_observation(
    state: Dict[str, Any],
    goal_text: str,
    actions: List[Tuple[str, str, int]],  # (etype, target_url, count)
    include_history: List[str] | None = None,
    add_read_and_stop: bool = True,
    *,
    extras: List[str] | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
) -> str:
    """Create a compact text observation with numbered actions.

    actions are pre-capped by the env (top_k).
    """
    title = state.get("title", "") or "(untitled)"
    page_type = state.get("page_type", "") or "general"
    url = state.get("url", "")
    lines = []
    if prefix:
        lines.append(prefix)
    lines.append(f"GOAL: {goal_text}")
    lines.append(f"PAGE: {title} [{page_type}] — {url}")
    if include_history:
        lines.append("HISTORY:")
        for h in include_history[-2:]:  # last 2
            lines.append(f"- {h}")
    lines.append("ACTIONS:")
    idx = 1
    for etype, tgt, cnt in actions:
        slug = (tgt or "").split("//")[-1][:80]
        lines.append(f"[{idx}] {etype} -> {slug} (obs={cnt})")
        idx += 1
    if add_read_and_stop:
        lines.append(f"[{idx}] READ_PAGE (stay)")
        idx += 1
        lines.append(f"[{idx}] STOP")
    lines.append("Respond with the number of the next action.")

    # Optional extras (e.g., reflection prompts, alternatives, metadata)
    if extras:
        for block in extras:
            if block:
                lines.append("")
                lines.append(block)

    if suffix:
        lines.append("")
        lines.append(suffix)
    return "\n".join(lines)


_DIGIT_RE = re.compile(r"(\d+)")


def parse_action_number(text: str, max_index: int) -> int:
    """Extract the first positive integer from the model output.

    Returns 0-based index; clamps to [0, max_index-1]. Defaults to STOP if not found.
    """
    m = _DIGIT_RE.search(text or "")
    if not m:
        return max_index - 1  # STOP
    try:
        n = int(m.group(1))
    except Exception:
        return max_index - 1
    n = max(1, min(n, max_index))
    return n - 1
