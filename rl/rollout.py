#!/usr/bin/env python3
"""Rollout machinery for GSPO training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Sequence

from pliny_env.env import PlinyBrowseEnv

from .policy_base import PolicyStep, SequencePolicy
from .utils import parse_actions


@dataclass
class StepRecord:
    data: Any
    action_index: int
    log_prob: float
    reward: float


@dataclass
class EpisodeRecord:
    task_id: str
    goal_url: str
    steps: List[StepRecord]
    total_reward: float
    old_log_prob: float
    success: bool
    path_length: int

    @property
    def length(self) -> int:
        return len(self.steps)


class RolloutCollector:
    """Collect episodes from the environment using a given policy."""

    def __init__(
        self,
        env: PlinyBrowseEnv,
        tasks: Sequence[dict],
        max_episode_steps: int = 10,
        seed: int = 7,
    ) -> None:
        self.env = env
        self.tasks = list(tasks)
        self.max_episode_steps = max_episode_steps
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def sample_task(self) -> dict:
        return self.rng.choice(self.tasks)

    def collect_group(
        self,
        policy: SequencePolicy,
        group_size: int,
        *,
        sample_actions: bool = True,
        store_features: bool = True,
    ) -> List[EpisodeRecord]:
        episodes: List[EpisodeRecord] = []
        while len(episodes) < group_size:
            task = self.sample_task()
            episode = self.run_episode(policy, task, sample_actions=sample_actions, store_features=store_features)
            if episode and (episode.length > 0 or not store_features):
                episodes.append(episode)
        return episodes

    # ------------------------------------------------------------------
    def run_episode(
        self,
        policy: SequencePolicy,
        task: dict,
        *,
        sample_actions: bool,
        store_features: bool,
    ) -> EpisodeRecord | None:
        obs = self.env.reset(task)
        observation = obs["observation"]
        steps: List[StepRecord] = []
        total_reward = 0.0
        old_log_prob = 0.0
        success = False
        path_length = 0

        goal_url = task.get("goal_url", "")

        for _ in range(self.max_episode_steps):
            action_infos = parse_actions(observation)
            if not action_infos:
                break
            step = policy.act(observation, goal_url, action_infos, sample=sample_actions)

            next_step = self.env.step(step.action_index)
            reward = float(next_step["reward"])
            total_reward += reward
            if store_features:
                steps.append(
                    StepRecord(
                        data=step.data,
                        action_index=step.action_index,
                        log_prob=step.log_prob,
                        reward=reward,
                    )
                )
                old_log_prob += step.log_prob

            observation = next_step["observation"]
            info = next_step.get("info", {})
            success = success or bool(info.get("reached_goal"))
            path_length = info.get("step", path_length)
            if next_step["done"]:
                break

        if not steps and store_features:
            return None

        return EpisodeRecord(
            task_id=task.get("task_id", "unknown"),
            goal_url=goal_url,
            steps=steps,
            total_reward=total_reward,
            old_log_prob=old_log_prob,
            success=success,
            path_length=path_length,
        )
