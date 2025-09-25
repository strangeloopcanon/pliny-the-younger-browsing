#!/usr/bin/env python3
"""
Task generation from trajectories/graph.

Task format:
{
  "task_id": str,
  "start_url": str,
  "goal_url": str,
  "reference_path": [url, ...],  # optional; used for shaping
  "domain": str,                  # derived from page_type or URL
  "difficulty": str              # easy|medium|hard based on path length/branching
}
"""
from __future__ import annotations

import json
import logging
import os
import random
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from .utils import domain_label
from urllib.parse import urlparse


@dataclass
class TaskGenerationConfig:
    include_domains: Optional[Set[str]] = None
    exclude_domains: Optional[Set[str]] = None
    include_hosts: Optional[Set[str]] = None
    exclude_hosts: Optional[Set[str]] = None
    min_hops: int = 1
    max_hops: Optional[int] = None
    sample_rate: float = 1.0

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "TaskGenerationConfig":
        if not data:
            return TaskGenerationConfig()
        return TaskGenerationConfig(
            include_domains=set(map(str.lower, data.get("include_domains", []))) or None,
            exclude_domains=set(map(str.lower, data.get("exclude_domains", []))) or None,
            include_hosts=set(map(str.lower, data.get("include_hosts", []))) or None,
            exclude_hosts=set(map(str.lower, data.get("exclude_hosts", []))) or None,
            min_hops=int(data.get("min_hops", 1)),
            max_hops=int(data["max_hops"]) if data.get("max_hops") is not None else None,
            sample_rate=float(data.get("sample_rate", 1.0)),
        )

    def allow(self, start_url: str, goal_url: str, hops: int, domain: str) -> bool:
        if hops < self.min_hops:
            return False
        if self.max_hops is not None and hops > self.max_hops:
            return False
        domain_lower = (domain or "").lower()
        if self.include_domains and domain_lower not in self.include_domains:
            return False
        if self.exclude_domains and domain_lower in self.exclude_domains:
            return False
        host = urlparse(goal_url).netloc.lower()
        if self.include_hosts and host not in self.include_hosts:
            return False
        if self.exclude_hosts and host in self.exclude_hosts:
            return False
        return True


def _maybe_sample(rng: random.Random, cfg: TaskGenerationConfig) -> bool:
    if cfg.sample_rate >= 1.0:
        return True
    return rng.random() < cfg.sample_rate


def tasks_from_trajectories(
    trajs: List[List[Dict[str, Any]]],
    max_tasks: int | None = None,
    *,
    config: Optional[TaskGenerationConfig] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    cfg = config or TaskGenerationConfig()
    rng = random.Random(seed)
    tasks: List[Dict[str, Any]] = []
    for i, traj in enumerate(trajs):
        if len(traj) < 2:
            continue
        start = traj[0].get("state", {}).get("url", "")
        goal = traj[-1].get("state", {}).get("url", "")
        if not start or not goal:
            continue
        ref = []
        for step in traj:
            a = step.get("action", {})
            dst = a.get("target_url")
            if dst:
                ref.append(dst)
        page_type = traj[-1].get("state", {}).get("page_type", "")
        domain = domain_label(goal, page_type)
        hops = len(ref)
        if not cfg.allow(start, goal, hops, domain):
            continue
        if not _maybe_sample(rng, cfg):
            continue
        difficulty = "easy" if hops <= 3 else "medium" if hops <= 5 else "hard"
        tasks.append(
            {
                "task_id": f"traj_{i:04d}",
                "start_url": start,
                "goal_url": goal,
                "reference_path": ref,
                "domain": domain,
                "difficulty": difficulty,
            }
        )
        if max_tasks and len(tasks) >= max_tasks:
            break
    return tasks


def tasks_from_csv_sequences(
    sequences: List[List[Dict[str, Any]]],
    *,
    max_tasks: int | None = None,
    max_hops: int = 5,
    seed: int = 13,
    config: Optional[TaskGenerationConfig] = None,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    seen = set()
    counter = 0
    rng = random.Random(seed)
    cfg = config or TaskGenerationConfig()
    if cfg.max_hops is not None:
        max_hops = min(max_hops, cfg.max_hops)

    for seq_idx, sequence in enumerate(sequences):
        if len(sequence) < 2:
            continue
        urls = [item.get("url") for item in sequence if item.get("url")]
        if len(urls) < 2:
            continue
        for start_idx in range(len(urls) - 1):
            # Optionally subsample longer sequences to avoid explosion
            if len(urls) > max_hops * 2 and rng.random() > 0.6:
                continue
            for end_idx in range(start_idx + 1, min(len(urls), start_idx + max_hops + 1)):
                start_url = urls[start_idx]
                goal_url = urls[end_idx]
                ref = urls[start_idx + 1 : end_idx + 1]
                if not start_url or not goal_url or not ref:
                    continue
                key = (start_url, goal_url, tuple(ref))
                if key in seen:
                    continue
                seen.add(key)
                meta = sequence[end_idx]
                domain = domain_label(goal_url, meta.get("page_type"))
                steps = len(ref)
                if not cfg.allow(start_url, goal_url, steps, domain):
                    continue
                if not _maybe_sample(rng, cfg):
                    continue
                difficulty = "easy" if steps <= 3 else "medium" if steps <= 6 else "hard"
                tasks.append(
                    {
                        "task_id": f"csv_{counter:06d}",
                        "start_url": start_url,
                        "goal_url": goal_url,
                        "reference_path": ref,
                        "domain": domain,
                        "difficulty": difficulty,
                        "source": "csv",
                        "source_file": meta.get("source_file"),
                    }
                )
                counter += 1
                if max_tasks and len(tasks) >= max_tasks:
                    return tasks
    return tasks


def tasks_from_csv_sequences_file(
    path: str,
    *,
    max_tasks: Optional[int] = None,
    max_hops: int = 5,
    seed: int = 13,
    config: Optional[TaskGenerationConfig] = None,
) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        logger.warning("CSV sequences file not found: %s", path)
        return []
    cfg = config or TaskGenerationConfig()
    rng = random.Random(seed)
    tasks: List[Dict[str, Any]] = []
    seen = set()
    max_tasks = max_tasks or 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sequence = json.loads(line)
            seq_tasks = tasks_from_csv_sequences(
                [sequence],
                max_tasks=max_tasks - len(tasks) if max_tasks else None,
                max_hops=max_hops,
                seed=rng.randint(0, 1_000_000),
                config=cfg,
            )
            for task in seq_tasks:
                key = (task["start_url"], task["goal_url"], tuple(task["reference_path"]))
                if key in seen:
                    continue
                seen.add(key)
                tasks.append(task)
                if max_tasks and len(tasks) >= max_tasks:
                    return tasks
    return tasks


def split_tasks(tasks: List[Dict[str, Any]], test_ratio: float = 0.2, seed: int = 7) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = tasks[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test


def save_tasks(train: List[Dict[str, Any]], test: List[Dict[str, Any]], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "tasks_train.json")
    test_path = os.path.join(out_dir, "tasks_test.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f)
    logger.info("Wrote tasks: %s (%d) | %s (%d)", train_path, len(train), test_path, len(test))
    return train_path, test_path
