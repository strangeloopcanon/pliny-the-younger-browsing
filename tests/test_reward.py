import json

from pliny_env.env import EnvConfig, PlinyBrowseEnv
from pliny_env.reward import RewardConfig
from web_browsing_reward import WebBrowsingReward


def test_reward_structure():
    task = {
        "expected_actions": [
            {"type": "internal_navigate", "target_url": "https://docs", "reasoning": "Find docs"},
            {"type": "internal_navigate", "target_url": "https://result", "reasoning": "Open result"},
        ],
        "goal": "Find documentation",
    }
    response = (
        "CONTEXT: sample\nTHINKING PROCESS: step by step\n"
        "Step 1: internal_navigate to https://docs\nStep 2: internal_navigate to https://result"
    )
    reward_fn = WebBrowsingReward()
    score = reward_fn.calculate_reward(task, response)
    assert 0 <= score <= 1
    assert score > 0


def test_custom_reward(tmp_path):
    graph = {
        "nodes": {
            "a": {"title": "A", "page_type": "general"},
            "b": {"title": "B", "page_type": "general"},
        },
        "edges": {"a": [{"type": "internal_navigate", "target": "b", "count": 1}]},
        "meta": {},
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    tasks = [
        {
            "task_id": "t1",
            "start_url": "a",
            "goal_url": "b",
            "reference_path": ["b"],
        }
    ]
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")

    cfg = EnvConfig(
        graph_path=str(graph_path),
        tasks_path=str(tasks_path),
        reward=RewardConfig(
            reward_module="tests.dummy_reward_plugin",
            reward_class="ConstantReward",
            reward_config={"value": 0.25},
        ),
        max_steps=2,
    )
    env = PlinyBrowseEnv(cfg)
    env.reset(tasks[0])
    step = env.step(0)
    assert step["reward"] > 0
