import json

from pliny_env.tasks import TaskGenerationConfig, tasks_from_trajectories


SAMPLE_TRAJS = [
    [
        {
            "state": {"url": "https://example.com/start", "page_type": "general"},
            "action": {"type": "internal_navigate", "target_url": "https://example.com/mid"},
        },
        {
            "state": {"url": "https://example.com/mid", "page_type": "docs"},
            "action": {"type": "internal_navigate", "target_url": "https://github.com/end"},
        },
        {
            "state": {"url": "https://github.com/end", "page_type": "github_repository"},
            "action": {"type": "internal_navigate", "target_url": "https://github.com/done"},
        },
    ]
]


def test_include_domain_filter():
    cfg = TaskGenerationConfig.from_dict({"include_domains": ["github"]})
    tasks = tasks_from_trajectories(SAMPLE_TRAJS, config=cfg, max_tasks=10)
    assert tasks, "expected filtered tasks"
    assert all(task["domain"] == "github" for task in tasks)


def test_sample_rate_filter():
    cfg = TaskGenerationConfig.from_dict({"sample_rate": 0.0})
    tasks = tasks_from_trajectories(SAMPLE_TRAJS, config=cfg)
    assert tasks == []
