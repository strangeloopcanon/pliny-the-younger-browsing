#!/usr/bin/env python3
"""
PlinyBrowseEnv: a simple graph-based RL environment built from browsing data.

- State: current URL node (with metadata) + goal URL + small history
- Actions: numbered menu of outgoing edges (etype, target), plus READ and STOP
- Transition: chosen edge -> move to target; READ keeps current node
- Reward: provided by reward.compute_step_reward
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment as JinjaEnvironment, FileSystemLoader as JinjaFileSystemLoader

from .adapters import format_observation
from .reward import RewardConfig, compute_step_reward, load_custom_reward


@dataclass
class EnvConfig:
    graph_path: str = "env_artifacts/graph.json"
    tasks_path: Optional[str] = None  # optional, else dynamic sampling from graph
    max_steps: int = 10
    top_k_actions: int = 10
    include_read_and_stop: bool = True
    seed: int = 7
    reward: RewardConfig = field(default_factory=RewardConfig)
    # Observation customization
    history_window: int = 2
    reflection_prompts: bool = False
    reflection_every_n: Optional[int] = None  # e.g., 1 for every step, 0/None disables
    show_alternatives: bool = False
    alternatives_k: int = 3
    observation_prefix: Optional[str] = None
    observation_suffix: Optional[str] = None
    observation_template: Optional[str] = None  # path to Jinja template


class PlinyBrowseEnv:
    def __init__(self, cfg: EnvConfig) -> None:
        with open(cfg.graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)
        self.cfg = cfg
        self.rng_state = cfg.seed
        self.nodes: Dict[str, Dict[str, Any]] = self.graph.get("nodes", {})
        self.edges: Dict[str, List[Dict[str, Any]]] = self.graph.get("edges", {})
        # Tasks are optional; when present we sample from them
        self.tasks: List[Dict[str, Any]] = []
        if cfg.tasks_path and cfg.tasks_path.endswith(".json"):
            try:
                with open(cfg.tasks_path, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                if isinstance(arr, list):
                    self.tasks = arr
            except Exception:
                self.tasks = []
        # Episode state
        self.step_num: int = 0
        self.cur_url: str = ""
        self.goal_url: str = ""
        self.history: List[str] = []
        self.reference_path: List[str] = []
        self.custom_reward = load_custom_reward(self.cfg.reward)
        self.current_task: Dict[str, Any] = {}
        self._last_observation: str = ""

        self.jinja_env: Optional[JinjaEnvironment] = None
        self.jinja_template: Optional[str] = None
        self._load_template()

    # Simple linear congruential generator for deterministic sampling without numpy
    def _rand(self) -> float:
        self.rng_state = (1103515245 * self.rng_state + 12345) % (1 << 31)
        return self.rng_state / float(1 << 31)

    def _choice(self, seq: List[Any]) -> Any:
        if not seq:
            return None
        idx = int(self._rand() * len(seq))
        return seq[min(idx, len(seq) - 1)]

    def _goal_text(self) -> str:
        goal_meta = self.nodes.get(self.goal_url, {})
        title = goal_meta.get("title") or self.goal_url
        return f"Reach: {title}"

    def _actions_for(self, url: str) -> List[Tuple[str, str, int]]:
        arr = self.edges.get(url, [])
        out = []
        for e in arr[: self.cfg.top_k_actions]:
            out.append((e.get("type", "internal_navigate"), e.get("target", ""), int(e.get("count", 1))))
        return out

    def _state_meta(self, url: str) -> Dict[str, Any]:
        m = dict(self.nodes.get(url, {}))
        m["url"] = url
        return m

    def reset(self, task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.step_num = 0
        self.history = []
        self.reference_path = []
        self.current_task = {}
        self.custom_reward = load_custom_reward(self.cfg.reward)
        if task is None:
            # Sample a random task from tasks if present; else any node pair connected by edges
            if self.tasks:
                task = self._choice(self.tasks)
            else:
                # fallback: pick a random src and one of its outgoing targets as goal
                src = self._choice(list(self.edges.keys()))
                tgt = ""
                if src:
                    outs = self.edges.get(src, [])
                    if outs:
                        tgt = self._choice(outs).get("target", "")
                task = {
                    "task_id": "ad_hoc",
                    "start_url": src or "",
                    "goal_url": tgt or src or "",
                    "reference_path": [],
                }
        self.cur_url = task.get("start_url", "")
        self.goal_url = task.get("goal_url", "")
        self.current_task = task
        self.reference_path = list(task.get("reference_path", []))
        obs = self._build_observation()
        self._last_observation = obs
        return {
            "observation": obs,
            "task": task,
        }

    def step(self, action_index: int) -> Dict[str, Any]:
        self.step_num += 1
        actions = self._actions_for(self.cur_url)
        max_index = len(actions) + (2 if self.cfg.include_read_and_stop else 0)
        # map index to chosen edge or READ/STOP
        chosen_edge: Optional[Dict[str, str]] = None
        chose_read = False
        chose_stop = False
        if action_index < len(actions):
            et, tgt, _ = actions[action_index]
            chosen_edge = {"type": et, "target": tgt}
        else:
            if self.cfg.include_read_and_stop:
                if action_index == len(actions):
                    chose_read = True
                else:
                    chose_stop = True

        is_terminal = False
        reached_goal = False
        ref_next_url = self.reference_path[0] if self.reference_path else None
        ref_next_type = None  # unknown in generic case

        # Transition
        if chose_read:
            # stay in place
            self.history.append("READ_PAGE")
        elif chosen_edge is not None:
            prev = self.cur_url
            self.cur_url = chosen_edge.get("target", self.cur_url)
            self.history.append(f"{chosen_edge.get('type')}:{self.cur_url}")
            # consume reference path if matched
            if ref_next_url and self.cur_url == ref_next_url:
                self.reference_path = self.reference_path[1:]
        else:
            # STOP or invalid
            is_terminal = True
            chose_stop = True

        # Check terminal conditions
        if self.cur_url == self.goal_url:
            is_terminal = True
            reached_goal = True
        if self.step_num >= self.cfg.max_steps:
            is_terminal = True

        reward = compute_step_reward(
            self.cfg.reward,
            is_terminal=is_terminal,
            reached_goal=reached_goal,
            chose_stop=chose_stop,
            chose_read=chose_read,
            chosen_edge=chosen_edge,
            ref_next_url=ref_next_url,
            ref_next_type=ref_next_type,
        )
        if self.custom_reward:
            try:
                bonus = self.custom_reward.calculate_step_reward(
                    task=self.current_task,
                    observation=getattr(self, "_last_observation", ""),
                    history=list(self.history),
                    chosen_edge=chosen_edge,
                    reward=reward,
                    info={
                        "is_terminal": is_terminal,
                        "reached_goal": reached_goal,
                        "chose_stop": chose_stop,
                        "chose_read": chose_read,
                        "step": self.step_num,
                    },
                )
                reward += float(bonus or 0.0)
            except Exception:
                pass

        obs = self._build_observation()
        self._last_observation = obs

        return {
            "observation": obs,
            "reward": reward,
            "done": is_terminal,
            "info": {
                "reached_goal": reached_goal,
                "step": self.step_num,
                "current_url": self.cur_url,
                "goal_url": self.goal_url,
                "history": list(self.history),
                "max_index": max_index,
            },
        }

    # -----------------------------
    # Observation builder with extras
    # -----------------------------
    def _build_observation(self) -> str:
        state_meta = self._state_meta(self.cur_url)
        actions = self._actions_for(self.cur_url)

        reflection = self.cfg.reflection_prompts and self._should_reflect()
        alternatives = []
        if self.cfg.show_alternatives and actions:
            k = max(1, int(self.cfg.alternatives_k))
            for action in actions[:k]:
                tgt = action[1] or ""
                alternatives.append(
                    {
                        "type": action[0],
                        "target": tgt,
                        "host": tgt.split("//")[-1].split("/")[0],
                        "count": action[2],
                    }
                )

        context = {
            "state": state_meta,
            "goal_text": self._goal_text(),
            "actions": [
                {
                    "index": idx + 1,
                    "type": action[0],
                    "target": action[1],
                    "count": action[2],
                }
                for idx, action in enumerate(actions)
            ],
            "history": self.history[-self.cfg.history_window :] if self.history else [],
            "include_read_stop": self.cfg.include_read_and_stop,
            "reflection": reflection,
            "alternatives": alternatives,
        }

        if self.jinja_env and self.jinja_template:
            template = self.jinja_env.get_template(self.jinja_template)
            return template.render(
                **context,
                prefix=self.cfg.observation_prefix,
                suffix=self.cfg.observation_suffix,
            )

        extras: List[str] = []
        if reflection:
            extras.append(
                "REFLECTION (optional):\n"
                "- What should you think before acting?\n"
                "- List 1–2 plausible alternatives and why they might help."
            )
        if alternatives:
            extras.append(
                "ALTERNATIVES (candidates):\n"
                + "\n".join(
                    "  {index}. {type} -> {target} (host={host}, obs={count})".format(index=i + 1, **alt)
                    for i, alt in enumerate(alternatives)
                )
            )

        return format_observation(
            state=state_meta,
            goal_text=self._goal_text(),
            actions=actions,
            include_history=context["history"],
            add_read_and_stop=self.cfg.include_read_and_stop,
            extras=extras,
            prefix=self.cfg.observation_prefix,
            suffix=self.cfg.observation_suffix,
        )

    def _should_reflect(self) -> bool:
        n = self.cfg.reflection_every_n
        if not n or n <= 0:
            return True
        # Reflect every n steps (1-indexed)
        return (self.step_num + 1) % n == 1

    # -----------------------------
    # Template loading helpers
    # -----------------------------
    def _load_template(self) -> None:
        template_path = self.cfg.observation_template
        if not template_path:
            self.jinja_env = None
            self.jinja_template = None
            return

        path = Path(template_path)
        if not path.exists():
            raise FileNotFoundError(f"Observation template not found: {template_path}")

        search_path = str(path.parent)
        env = JinjaEnvironment(loader=JinjaFileSystemLoader(search_path), autoescape=False)
        env.globals.update(len=len)
        self.jinja_env = env
        self.jinja_template = path.name
