#!/usr/bin/env python3
"""GSPO trainer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

from .policy_base import SequencePolicy
from .rollout import EpisodeRecord


@dataclass
class GSPOMetrics:
    loss: float
    mean_reward: float
    reward_std: float
    mean_ratio: float
    mean_length: float


class GSPOTrainer:
    def __init__(
        self,
        policy: SequencePolicy,
        optimizer: Optional[torch.optim.Optimizer],
        clip_epsilon: float = 0.1,
        eps: float = 1e-6,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.policy_old = policy.clone_policy()
        self.policy_old.eval()
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.eps = eps
        self.device = device or torch.device("cpu")

    # ------------------------------------------------------------------
    def sync_old_policy(self) -> None:
        if hasattr(self.policy, "state_dict") and hasattr(self.policy_old, "load_state_dict"):
            try:
                state = self.policy.state_dict()
                self.policy_old.load_state_dict(state)
            except Exception:
                pass
        self.policy_old.eval()

    # ------------------------------------------------------------------
    def _sequence_log_prob(self, episode: EpisodeRecord) -> torch.Tensor:
        steps = [(step.data, step.action_index) for step in episode.steps]
        return self.policy.seq_log_prob(steps)

    # ------------------------------------------------------------------
    def update(self, episodes: Iterable[EpisodeRecord]) -> GSPOMetrics:
        episodes = list(episodes)
        if not episodes:
            raise ValueError("No episodes provided for GSPO update")

        rewards = torch.tensor([ep.total_reward for ep in episodes], device=self.device)
        lengths = torch.tensor([max(ep.length, 1) for ep in episodes], device=self.device, dtype=torch.float32)
        logpi_old = torch.tensor([ep.old_log_prob for ep in episodes], device=self.device)

        logpi_new = torch.stack([self._sequence_log_prob(ep) for ep in episodes])

        ratios = torch.exp((logpi_new - logpi_old) / lengths)

        reward_mean = rewards.mean()
        reward_std = rewards.std(unbiased=False)
        advantages = (rewards - reward_mean) / (reward_std + self.eps)

        # Sequence-level clipping
        clipped = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        adjusted = torch.min(ratios * advantages, clipped * advantages)

        loss = -adjusted.mean()

        can_train = (
            self.optimizer is not None
            and loss.requires_grad
            and any(p.requires_grad for p in self.policy.parameters())
        )
        if can_train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        return GSPOMetrics(
            loss=float(loss.item()),
            mean_reward=float(reward_mean.item()),
            reward_std=float(reward_std.item()),
            mean_ratio=float(ratios.mean().item()),
            mean_length=float(lengths.float().mean().item()),
        )
