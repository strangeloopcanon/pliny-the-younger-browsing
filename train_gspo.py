#!/usr/bin/env python3
"""GSPO training entrypoint supporting hash/HF/MLX policies."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import torch

from pliny_env.env import EnvConfig, PlinyBrowseEnv

from rl.gspo_trainer import GSPOTrainer
from rl.policy_base import SequencePolicy
from rl.policy_hash import HashSoftmaxPolicy
from rl.policy_hf import HFPolicy
from rl.policy_mlx import MLXPolicy
from rl.rollout import RolloutCollector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_gspo")


def load_env(env_config_path: str) -> tuple[PlinyBrowseEnv, List[dict], List[dict]]:
    with open(env_config_path, "r", encoding="utf-8") as f:
        env_cfg_dict = json.load(f)

    env_cfg = EnvConfig(
        graph_path=env_cfg_dict.get("graph_path", "env_artifacts/graph.json"),
        tasks_path=env_cfg_dict.get("tasks_path"),
        max_steps=int(env_cfg_dict.get("max_steps", 10)),
        top_k_actions=int(env_cfg_dict.get("top_k_actions", 10)),
    )
    env = PlinyBrowseEnv(env_cfg)

    tasks_train_path = env_cfg_dict.get("tasks_path") or "env_artifacts/tasks_train.json"
    tasks_test_path = tasks_train_path.replace("train", "test")

    with open(tasks_train_path, "r", encoding="utf-8") as f:
        train_tasks = json.load(f)
    test_tasks: List[dict] = []
    if os.path.exists(tasks_test_path):
        with open(tasks_test_path, "r", encoding="utf-8") as f:
            test_tasks = json.load(f)

    return env, train_tasks, test_tasks


def build_policy(args: argparse.Namespace, device: torch.device) -> SequencePolicy:
    if args.policy == "hash":
        return HashSoftmaxPolicy(device=device)
    if args.policy == "hf":
        return HFPolicy(
            model_name=args.hf_model,
            temperature=args.hf_temperature,
            top_p=args.hf_top_p,
            top_k=args.hf_top_k,
            max_new_tokens=args.hf_max_new,
            device=device,
        )
    if args.policy == "mlx":
        return MLXPolicy(
            model_id=args.mlx_model,
            cache_dir=args.mlx_cache,
            quantize=args.mlx_quantize,
            temperature=args.mlx_temperature,
            top_p=args.mlx_top_p,
            max_tokens=args.mlx_max_tokens,
            seed=args.seed,
        )
    if args.policy == "http":
        headers = json.loads(args.http_headers) if args.http_headers else None
        return HTTPPolicy(
            endpoint=args.http_endpoint,
            headers=headers,
            timeout=args.http_timeout,
        )
    raise ValueError(f"Unsupported policy: {args.policy}")


def evaluate(policy: SequencePolicy, env: PlinyBrowseEnv, tasks: List[dict], episodes: int = 10) -> float:
    if not tasks:
        return 0.0
    collector = RolloutCollector(env, tasks, max_episode_steps=env.cfg.max_steps)
    episodes_data = collector.collect_group(policy, episodes, sample_actions=False, store_features=False)
    rewards = [ep.total_reward for ep in episodes_data]
    return sum(rewards) / max(len(rewards), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default="env_artifacts/env_config.json")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--clip-epsilon", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--refresh-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=17)

    sub = ap.add_subparsers(dest="policy", required=True)

    sub.add_parser("hash")

    hf = sub.add_parser("hf")
    hf.add_argument("--hf-model", default="sshleifer/tiny-gpt2")
    hf.add_argument("--hf-temperature", type=float, default=0.7)
    hf.add_argument("--hf-top-p", type=float, default=0.9)
    hf.add_argument("--hf-top-k", type=int, default=20)
    hf.add_argument("--hf-max-new", type=int, default=4)

    mlx = sub.add_parser("mlx")
    mlx.add_argument("--mlx-model", default="Qwen/Qwen2-0.5B-Instruct")
    mlx.add_argument("--mlx-cache", default="mlx_cache")
    mlx.add_argument("--mlx-quantize", action="store_true")
    mlx.add_argument("--mlx-temperature", type=float, default=0.7)
    mlx.add_argument("--mlx-top-p", type=float, default=0.95)
    mlx.add_argument("--mlx-max-tokens", type=int, default=8)

    http = sub.add_parser("http")
    http.add_argument("--http-endpoint", required=True)
    http.add_argument(
        "--http-headers",
        help="JSON string of headers, e.g. '{\"Authorization\":\"Bearer ...\"}'",
    )
    http.add_argument("--http-timeout", type=float, default=30.0)

    args = ap.parse_args()

    env, train_tasks, test_tasks = load_env(args.env_config)
    device = torch.device("cpu")

    policy = build_policy(args, device)
    params = list(policy.parameters()) if hasattr(policy, "parameters") else []
    optimizer = torch.optim.Adam(params, lr=args.lr) if params else None
    trainer = GSPOTrainer(policy, optimizer, clip_epsilon=args.clip_epsilon, device=device)
    collector = RolloutCollector(env, train_tasks, max_episode_steps=env.cfg.max_steps, seed=args.seed)

    logger.info("Starting GSPO training: iterations=%d group_size=%d policy=%s", args.iterations, args.group_size, args.policy)

    for iteration in range(1, args.iterations + 1):
        episodes = collector.collect_group(trainer.policy_old, args.group_size, sample_actions=True, store_features=True)
        metrics = trainer.update(episodes)
        if iteration % args.refresh_every == 0:
            trainer.sync_old_policy()

        logger.info(
            "Iter %d | loss=%.4f | reward=%.3f±%.3f | ratio=%.3f | len=%.2f",
            iteration,
            metrics.loss,
            metrics.mean_reward,
            metrics.reward_std,
            metrics.mean_ratio,
            metrics.mean_length,
        )

    eval_reward = evaluate(policy, env, test_tasks, episodes=5)
    logger.info("Evaluation average reward: %.3f", eval_reward)


if __name__ == "__main__":
    main()
