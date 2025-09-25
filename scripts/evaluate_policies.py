#!/usr/bin/env python3
"""Evaluate GSPO policies without updating weights."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import torch

from train_gspo import build_policy, load_env
from rl.rollout import RolloutCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_policies")


def summarise(episodes, reference_lookup=None) -> Dict[str, float]:
    rewards = [ep.total_reward for ep in episodes]
    lengths = [ep.length for ep in episodes]
    successes = [1 if getattr(ep, "success", False) else 0 for ep in episodes]
    path_ratios = []
    if reference_lookup:
        for ep in episodes:
            ref = reference_lookup.get(ep.task_id)
            if ref:
                path_ratios.append(ep.length / max(ref, 1))
    return {
        "episodes": len(episodes),
        "avg_reward": sum(rewards) / max(len(rewards), 1),
        "avg_length": sum(lengths) / max(len(lengths), 1),
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "success_rate": sum(successes) / max(len(successes), 1),
        "avg_path_ratio": sum(path_ratios) / len(path_ratios) if path_ratios else None,
    }


def build_reference_lookup(tasks: List[dict]) -> Dict[str, int]:
    lookup = {}
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            continue
        ref = task.get("reference_path") or []
        lookup[task_id] = max(len(ref), 1)
    return lookup


def evaluate_policy(
    args: argparse.Namespace,
    env_config: str,
    train_tasks: List[dict],
    *,
    store_features: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], List, List]:
    env, _, test_tasks = load_env(env_config)
    device = torch.device("cpu")
    policy = build_policy(args, device)

    collector = RolloutCollector(
        env,
        train_tasks,
        max_episode_steps=env.cfg.max_steps,
        seed=args.seed,
    )
    reference_lookup = build_reference_lookup(train_tasks + (test_tasks or []))
    train_eps = collector.collect_group(
        policy,
        args.episodes,
        sample_actions=False,
        store_features=store_features,
    )

    test_collector = RolloutCollector(
        env,
        test_tasks or train_tasks,
        max_episode_steps=env.cfg.max_steps,
        seed=args.seed + 1,
    )
    test_eps = test_collector.collect_group(
        policy,
        args.episodes,
        sample_actions=False,
        store_features=store_features,
    )

    return summarise(train_eps, reference_lookup), summarise(test_eps, reference_lookup), train_eps, test_eps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="env_artifacts/env_config.json")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--seeds", help="Comma-separated list of seeds (overrides --seed)")
    parser.add_argument("--config", help="Optional JSON file describing multiple policies")
    parser.add_argument("--output", help="Path to write JSON summary")
    parser.add_argument("--csv", help="Optional CSV file for aggregated metrics")
    parser.add_argument("--save-trajectories", help="Directory to dump episode JSONL files")

    subparsers = parser.add_subparsers(dest="policy")

    hash_parser = subparsers.add_parser("hash")

    hf_parser = subparsers.add_parser("hf")
    hf_parser.add_argument("--hf-model", default="sshleifer/tiny-gpt2")
    hf_parser.add_argument("--hf-temperature", type=float, default=0.7)
    hf_parser.add_argument("--hf-top-p", type=float, default=0.9)
    hf_parser.add_argument("--hf-top-k", type=int, default=20)
    hf_parser.add_argument("--hf-max-new", type=int, default=4)

    mlx_parser = subparsers.add_parser("mlx")
    mlx_parser.add_argument("--mlx-model", default="Qwen/Qwen2-0.5B-Instruct")
    mlx_parser.add_argument("--mlx-cache", default="mlx_cache")
    mlx_parser.add_argument("--mlx-quantize", action="store_true")
    mlx_parser.add_argument("--mlx-temperature", type=float, default=0.7)
    mlx_parser.add_argument("--mlx-top-p", type=float, default=0.95)
    mlx_parser.add_argument("--mlx-max-tokens", type=int, default=8)

    http_parser = subparsers.add_parser("http")
    http_parser.add_argument("--http-endpoint", required=True)
    http_parser.add_argument("--http-headers")
    http_parser.add_argument("--http-timeout", type=float, default=30.0)

    args = parser.parse_args()
    if not args.config and not args.policy:
        parser.error("Specify a policy subcommand or provide --config")
    return args


def main() -> None:
    args = parse_args()
    env_cfg = args.env_config

    if not Path(env_cfg).exists():
        raise FileNotFoundError(f"Env config not found: {env_cfg}")

    with open(args.env_config, "r", encoding="utf-8") as f:
        env_dict = json.load(f)
    tasks_train_path = env_dict.get("tasks_path") or "env_artifacts/tasks_train.json"
    with open(tasks_train_path, "r", encoding="utf-8") as f:
        train_tasks = json.load(f)
    policies: List[tuple[str, argparse.Namespace]] = []

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_specs = json.load(f)
        policies = []
        for spec in config_specs:
            namespace = argparse.Namespace(**{**vars(args), **spec.get("args", {})})
            namespace.policy = spec["policy"]
            policies.append((spec.get("name", spec["policy"]), namespace))
    else:
        policies.append((args.policy, args))

    seeds = [args.seed]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    save_dir = Path(args.save_trajectories) if args.save_trajectories else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    csv_rows = []
    for name, ns in policies:
        per_seed = []
        for seed in seeds:
            ns.seed = seed
            train_stats, test_stats, train_eps, test_eps = evaluate_policy(
                ns, env_cfg, train_tasks, store_features=save_dir is not None
            )
            per_seed.append({"seed": seed, "train": train_stats, "test": test_stats})

            if save_dir:
                for split, eps in [("train", train_eps), ("test", test_eps)]:
                    out_path = save_dir / f"{name}_{split}_seed{seed}.jsonl"
                    with out_path.open("w", encoding="utf-8") as f:
                        for ep in eps:
                            record = {
                                "task_id": ep.task_id,
                                "goal_url": ep.goal_url,
                                "total_reward": ep.total_reward,
                                "length": ep.length,
                            }
                            f.write(json.dumps(record) + "\n")

            logger.info("[%s seed=%s] train=%s test=%s", name, seed, train_stats, test_stats)

        agg = {
            "per_seed": per_seed,
            "avg_reward_train": sum(item["train"]["avg_reward"] for item in per_seed) / len(per_seed),
            "avg_reward_test": sum(item["test"]["avg_reward"] for item in per_seed) / len(per_seed),
            "avg_length_train": sum(item["train"]["avg_length"] for item in per_seed) / len(per_seed),
            "avg_length_test": sum(item["test"]["avg_length"] for item in per_seed) / len(per_seed),
            "std_reward_train": stdev(item["train"]["avg_reward"] for item in per_seed) if len(per_seed) > 1 else 0.0,
            "std_reward_test": stdev(item["test"]["avg_reward"] for item in per_seed) if len(per_seed) > 1 else 0.0,
        }
        results[name] = agg

        csv_rows.append(
            {
                "policy": name,
                "avg_reward_train": agg["avg_reward_train"],
                "avg_reward_test": agg["avg_reward_test"],
                "std_reward_train": agg["std_reward_train"],
                "std_reward_test": agg["std_reward_test"],
                "avg_length_train": agg["avg_length_train"],
                "avg_length_test": agg["avg_length_test"],
                "success_rate_train": sum(item["train"]["success_rate"] for item in per_seed) / len(per_seed),
                "success_rate_test": sum(item["test"]["success_rate"] for item in per_seed) / len(per_seed),
                "avg_path_ratio_train": sum(
                    item["train"].get("avg_path_ratio", 0.0) or 0.0 for item in per_seed
                ) / len(per_seed),
                "avg_path_ratio_test": sum(
                    item["test"].get("avg_path_ratio", 0.0) or 0.0 for item in per_seed
                ) / len(per_seed),
                "episodes": len(per_seed),
            }
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.config or args.output:
        print(json.dumps(results, indent=2))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "policy",
                    "avg_reward_train",
                    "std_reward_train",
                    "avg_reward_test",
                    "std_reward_test",
                    "avg_length_train",
                    "avg_length_test",
                    "success_rate_train",
                    "success_rate_test",
                    "avg_path_ratio_train",
                    "avg_path_ratio_test",
                    "episodes",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

if __name__ == "__main__":
    main()
