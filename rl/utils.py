#!/usr/bin/env python3
"""Shared RL utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse


ACTION_PATTERN = re.compile(r"^\[(\d+)\]\s+(.*)$")


@dataclass
class ActionInfo:
    type: str
    target: str
    host: str


def _canonical_host(target: str) -> str:
    if not target:
        return ""
    target = target.strip()
    if not target:
        return ""
    if target.startswith("http://") or target.startswith("https://"):
        parsed = urlparse(target)
        return parsed.netloc.lower()
    if target.startswith("www."):
        return target.split("/")[0].lower()
    if target.startswith("/"):
        return ""
    return target.split("/")[0].lower()


def parse_actions(observation: str) -> List[ActionInfo]:
    actions: List[ActionInfo] = []
    if not observation:
        return actions
    for line in observation.splitlines():
        match = ACTION_PATTERN.match(line.strip())
        if not match:
            continue
        payload = match.group(2).strip()
        if "(obs=" in payload:
            payload = payload.split("(obs=", 1)[0].strip()
        action_type = payload
        target = ""
        if "->" in payload:
            head, tail = payload.split("->", 1)
            action_type = head.strip().split()[0]
            target = tail.strip()
        else:
            parts = payload.split()
            if parts:
                action_type = parts[0]
        host = _canonical_host(target)
        actions.append(ActionInfo(type=action_type.strip(), target=target, host=host))
    return actions

