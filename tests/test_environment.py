import json

from pliny_env.env import EnvConfig, PlinyBrowseEnv


def test_env_step_cycle(tmp_path):
    graph = {
        "nodes": {
            "https://start": {"title": "Start", "page_type": "general"},
            "https://goal": {"title": "Goal", "page_type": "general"},
        },
        "edges": {
            "https://start": [
                {"type": "internal_navigate", "target": "https://goal", "count": 1}
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

    config = EnvConfig(
        graph_path=str(graph_path),
        tasks_path=str(tasks_path),
        max_steps=2,
        reflection_prompts=True,
        show_alternatives=True,
    )
    env = PlinyBrowseEnv(config)
    obs = env.reset()
    assert "ACTIONS" in obs["observation"]
    assert "REFLECTION" in obs["observation"]
    assert "ALTERNATIVES" in obs["observation"]
    step = env.step(0)
    assert isinstance(step["reward"], float)
    assert isinstance(step["done"], bool)
