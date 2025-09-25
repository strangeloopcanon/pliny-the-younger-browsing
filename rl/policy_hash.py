#!/usr/bin/env python3
"""Hashed softmax policy (baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy_base import PolicyStep, SequencePolicy
from .utils import ActionInfo


@dataclass
class ActionBatch:
    type_ids: List[int]
    host_ids: List[int]
    context_id: int


class HashSoftmaxPolicy(SequencePolicy):
    def __init__(
        self,
        num_buckets: int = 4096,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.temperature = temperature
        self.device = device or torch.device("cpu")

        self.type_embedding = nn.Embedding(num_buckets, hidden_dim)
        self.host_embedding = nn.Embedding(num_buckets, hidden_dim)
        self.context_embedding = nn.Embedding(num_buckets, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    # ------------------------------------------------------------------
    def _hash(self, prefix: str, value: str) -> int:
        return abs(hash(f"{prefix}:{value}")) % self.num_buckets

    def encode_actions(self, goal_url: str, actions: List[ActionInfo]) -> ActionBatch:
        context_id = self._hash("context", goal_url or "")
        type_ids = [self._hash("type", action.type.lower()) for action in actions]
        host_ids = [self._hash("host", action.host or "") for action in actions]
        return ActionBatch(type_ids=type_ids, host_ids=host_ids, context_id=context_id)

    def _prepare(self, batch: ActionBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type_tensor = torch.tensor(batch.type_ids, dtype=torch.long, device=self.device)
        host_tensor = torch.tensor(batch.host_ids, dtype=torch.long, device=self.device)
        context_tensor = torch.full((len(batch.type_ids),), batch.context_id, dtype=torch.long, device=self.device)
        return type_tensor, host_tensor, context_tensor

    def compute_logits(self, batch: ActionBatch) -> torch.Tensor:
        type_tensor, host_tensor, context_tensor = self._prepare(batch)
        embed = (
            self.type_embedding(type_tensor)
            + self.host_embedding(host_tensor)
            + self.context_embedding(context_tensor)
        )
        hidden = torch.tanh(self.hidden(embed))
        logits = self.output(hidden).squeeze(-1)
        return logits

    def log_probs_from_batch(self, batch: ActionBatch) -> torch.Tensor:
        logits = self.compute_logits(batch)
        return F.log_softmax(logits / max(self.temperature, 1e-6), dim=-1)

    # ------------------------------------------------------------------
    def act(
        self,
        observation: str,
        goal_url: str,
        actions: List[ActionInfo],
        *,
        sample: bool = True,
    ) -> PolicyStep:
        if not actions:
            raise ValueError("No actions to choose from")
        batch = self.encode_actions(goal_url, actions)
        log_probs = self.log_probs_from_batch(batch)
        if sample:
            dist = torch.distributions.Categorical(logits=log_probs)
            action_index = dist.sample()
            log_prob = dist.log_prob(action_index)
        else:
            action_index = torch.argmax(log_probs)
            log_prob = log_probs[action_index]
        return PolicyStep(
            action_index=int(action_index.item()),
            log_prob=float(log_prob.item()),
            data=batch,
        )

    def seq_log_prob(self, steps: Sequence[tuple[Any, int]]) -> torch.Tensor:
        total = torch.zeros(1, device=self.device)
        for batch, action_index in steps:
            log_probs = self.log_probs_from_batch(batch)
            total = total + log_probs[action_index]
        return total.squeeze(0)

    def clone_policy(self) -> "HashSoftmaxPolicy":
        clone = HashSoftmaxPolicy(
            num_buckets=self.num_buckets,
            hidden_dim=self.type_embedding.embedding_dim,
            temperature=self.temperature,
            device=self.device,
        )
        clone.load_state_dict(self.state_dict())
        return clone
