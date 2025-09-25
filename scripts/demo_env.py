#!/usr/bin/env python3
"""
Run a quick demo episode with the PlinyBrowseEnv using the built graph/tasks.

Usage:
  python scripts/demo_env.py --cfg env_artifacts/env_config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pliny_env.env import EnvConfig, PlinyBrowseEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo_env")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="env_artifacts/env_config.json")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        c = json.load(f)
    cfg = EnvConfig(
        graph_path=c.get("graph_path", "env_artifacts/graph.json"),
        tasks_path=c.get("tasks_path"),
        max_steps=int(c.get("max_steps", 10)),
        top_k_actions=int(c.get("top_k_actions", 10)),
    )
    env = PlinyBrowseEnv(cfg)
    ep = env.reset()
    print("=== RESET ===")
    print(ep["observation"]) 
    total = 0.0
    for t in range(cfg.max_steps):
        # naive random chooser among menu entries -> pick 1 (first edge) or STOP
        # We parse max index from info or recompute
        # Here we just choose the first real edge if present; else STOP
        actions_count = len(env._actions_for(env.cur_url))
        if actions_count > 0:
            a = 0
        else:
            a = actions_count + 1  # STOP
        step = env.step(a)
        total += step["reward"]
        print("\n=== STEP", t + 1, "===")
        print(step["observation"]) 
        print("reward=", step["reward"], "done=", step["done"], "info=", {k: step["info"][k] for k in ("reached_goal", "current_url", "goal_url")})
        if step["done"]:
            break
    print("\nTotal reward:", round(total, 4))


if __name__ == "__main__":
    main()
