"""Pliny RL environment package.

Modules:
- graph: build/load a directed browsing graph from JSON and optional CSV.
- tasks: generate start→goal tasks from trajectories/graph.
- env: lightweight text-based RL environment over the browsing graph.
- adapters: observation formatting and action parsing for LLM policies.
- reward: step and terminal reward shaping helpers.
"""

__all__ = [
    "graph",
    "tasks",
    "env",
    "adapters",
    "reward",
    "csv_ingest",
    "utils",
]
