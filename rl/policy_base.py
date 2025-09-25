#!/usr/bin/env python3
"""Base policy abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import torch
import torch.nn as nn

from .utils import ActionInfo


@dataclass
class PolicyStep:
    action_index: int
    log_prob: float
    data: Any


class SequencePolicy(nn.Module):
    """Common interface for policies used in GSPO training."""

    def __init__(self) -> None:
        super().__init__()

    def act(
        self,
        observation: str,
        goal_url: str,
        actions: List[ActionInfo],
        *,
        sample: bool = True,
    ) -> PolicyStep:
        raise NotImplementedError

    def seq_log_prob(self, steps: Sequence[tuple[Any, int]]) -> torch.Tensor:
        raise NotImplementedError

    def clone_policy(self) -> "SequencePolicy":
        raise NotImplementedError

