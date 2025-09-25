#!/usr/bin/env python3
"""
Build the browsing graph and tasks from available data.

Usage:
  python scripts/build_env_graph.py \
    --traj corrected_trajectories.json \
    --csv data/log.csv --csv data/other.csv \
    --csv-glob "data/logs/*.csv" \
    --out env_artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from glob import glob
from typing import Any, Dict, List, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pliny_env.graph import GraphBuildConfig, build_graph
from pliny_env.tasks import (
    TaskGenerationConfig,
    tasks_from_csv_sequences,
    tasks_from_csv_sequences_file,
    tasks_from_trajectories,
    split_tasks,
    save_tasks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_env_graph")


def _load_task_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load task config %s: %s", path, exc)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="corrected_trajectories.json")
    ap.add_argument(
        "--csv",
        action="append",
        default=[],
        help="CSV browsing log file (repeatable)",
    )
    ap.add_argument(
        "--csv-glob",
        action="append",
        default=[],
        help="Glob pattern for CSV browsing logs",
    )
    ap.add_argument(
        "--csv-max",
        type=int,
        default=None,
        help="Optional cap on number of CSV files processed after globbing",
    )
    ap.add_argument("--out", default="env_artifacts")
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--max_csv_tasks", type=int, default=None)
    ap.add_argument("--csv_max_hops", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="Incrementally update existing artifacts")
    ap.add_argument("--sequence-out", default=None, help="Optional override for CSV sequences JSONL output")
    ap.add_argument("--task-config", default=None, help="JSON file specifying task filtering preferences")
    args = ap.parse_args()

    csv_paths: List[str] = []
    csv_paths.extend(args.csv)
    for pattern in args.csv_glob:
        csv_paths.extend(glob(pattern))
    # Deduplicate while preserving order
    seen_paths = set()
    deduped: List[str] = []
    for path in csv_paths:
        if path not in seen_paths:
            deduped.append(path)
            seen_paths.add(path)
    if args.csv_max is not None:
        deduped = deduped[: args.csv_max]

    sequence_out_path = args.sequence_out or os.path.join(args.out, "csv_sequences.jsonl")

    skip_paths = set()
    base_graph_path = None
    append_sequences = False

    if args.resume:
        base_graph_path = os.path.join(args.out, "graph.json")
        if os.path.exists(base_graph_path):
            try:
                with open(base_graph_path, "r", encoding="utf-8") as f:
                    base_graph = json.load(f)
                csv_meta = base_graph.get("meta", {}).get("csv_meta", {})
                skip_paths = {entry.get("path") for entry in csv_meta.get("files", []) if entry.get("path")}
            except Exception as exc:
                logger.warning("Failed to load existing graph for resume: %s", exc)
        if os.path.exists(sequence_out_path):
            append_sequences = True

    cfg = GraphBuildConfig(
        trajectories_path=args.traj,
        raw_csv_paths=deduped or None,
        output_dir=args.out,
        base_graph_path=base_graph_path,
        sequence_out_path=sequence_out_path if deduped else None,
        append_sequences=append_sequences,
        skip_csv_paths={p for p in skip_paths if p},
    )
    graph, csv_result = build_graph(cfg)

    # Derive tasks directly from the trajectories JSON
    with open(args.traj, "r", encoding="utf-8") as f:
        trajs = json.load(f).get("trajectories", [])

    task_cfg = TaskGenerationConfig.from_dict(_load_task_config(args.task_config))

    tasks = tasks_from_trajectories(
        trajs,
        max_tasks=args.max_tasks,
        config=task_cfg,
        seed=42,
    )

    if csv_result and csv_result.sequences:
        csv_tasks = tasks_from_csv_sequences(
            csv_result.sequences,
            max_tasks=args.max_csv_tasks,
            max_hops=args.csv_max_hops,
            config=task_cfg,
        )
        logger.info("Generated %d tasks from in-memory CSV sequences", len(csv_tasks))
        tasks.extend(csv_tasks)
    elif os.path.exists(sequence_out_path):
        csv_tasks = tasks_from_csv_sequences_file(
            sequence_out_path,
            max_tasks=args.max_csv_tasks,
            max_hops=args.csv_max_hops,
            config=task_cfg,
        )
        logger.info("Generated %d tasks from CSV sequences file", len(csv_tasks))
        tasks.extend(csv_tasks)

    # Deduplicate tasks by path signature
    unique_tasks = []
    seen_keys = set()
    for task in tasks:
        key = (task.get("start_url"), task.get("goal_url"), tuple(task.get("reference_path", [])))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_tasks.append(task)

    tasks = unique_tasks
    logger.info("Total tasks before split: %d", len(tasks))
    train, test = split_tasks(tasks, test_ratio=0.2)
    save_tasks(train, test, args.out)

    # Write a small env config to point to artifacts
    env_cfg_path = os.path.join(args.out, "env_config.json")
    env_cfg = {
        "graph_path": os.path.join(args.out, "graph.json"),
        "tasks_path": os.path.join(args.out, "tasks_train.json"),
        "max_steps": 10,
        "top_k_actions": 10,
    }
    with open(env_cfg_path, "w", encoding="utf-8") as f:
        json.dump(env_cfg, f, indent=2)
    logger.info("Wrote env config: %s", env_cfg_path)

    if deduped:
        new_files = 0
        if csv_result:
            new_files = csv_result.meta.get("file_count", 0)
        logger.info("CSV files requested: %d | newly processed: %d", len(deduped), new_files)


if __name__ == "__main__":
    main()
