import json

import torch

from pliny_env.env import EnvConfig, PlinyBrowseEnv
from rl.policy_hash import HashSoftmaxPolicy
from rl.rollout import RolloutCollector


def make_env(tmp_path):
    graph = {
        "nodes": {
            "https://start": {"title": "Start", "page_type": "general"},
            "https://goal": {"title": "Goal", "page_type": "general"},
        },
        "edges": {
            "https://start": [
                {"type": "internal_navigate", "target": "https://goal", "count": 1},
                {"type": "STOP", "target": "https://start", "count": 1},
            ]
        },
        "meta": {},
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    tasks = [
        {
            "task_id": "test",
            "start_url": "https://start",
            "goal_url": "https://goal",
            "reference_path": ["https://goal"],
            "domain": "general",
            "difficulty": "easy",
        }
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    cfg = EnvConfig(graph_path=str(graph_path), tasks_path=str(tasks_path), max_steps=2)
    return PlinyBrowseEnv(cfg)


def test_rollout_collect(tmp_path):
    env = make_env(tmp_path)
    policy = HashSoftmaxPolicy(device=torch.device("cpu"))
    collector = RolloutCollector(env, env.tasks, max_episode_steps=env.cfg.max_steps, seed=5)
    episodes = collector.collect_group(policy, group_size=1, sample_actions=True, store_features=True)
    assert episodes
    ep = episodes[0]
    assert ep.steps  # features stored
    assert isinstance(ep.total_reward, float)
