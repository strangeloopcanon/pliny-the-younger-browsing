#!/usr/bin/env python3
"""HTTP policy wrapper for external engines (evaluation only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import requests
import torch

from .policy_base import PolicyStep, SequencePolicy
from .utils import ActionInfo


logger = logging.getLogger(__name__)


@dataclass
class HTTPPolicyConfig:
    endpoint: str
    headers: Optional[Mapping[str, str]] = None
    timeout: float = 30.0


class HTTPPolicy(SequencePolicy):
    """Calls a remote endpoint to pick actions.

    Expected JSON response format:
    {
      "action_index": int,               # 0-based index
      "log_prob": float (optional)       # sequence log-prob, defaults to 0.
    }
    """

    def __init__(self, endpoint: str, headers: Optional[Mapping[str, str]] = None, timeout: float = 30.0) -> None:
        super().__init__()
        self.cfg = HTTPPolicyConfig(endpoint=endpoint, headers=headers, timeout=timeout)

    def act(
        self,
        observation: str,
        goal_url: str,
        actions: List[ActionInfo],
        *,
        sample: bool = True,
    ) -> PolicyStep:
        payload: Dict[str, Any] = {
            "observation": observation,
            "goal_url": goal_url,
            "actions": [action.__dict__ for action in actions],
            "sample": sample,
        }
        response = requests.post(
            self.cfg.endpoint,
            json=payload,
            headers=self.cfg.headers,
            timeout=self.cfg.timeout,
        )
        response.raise_for_status()
        data = response.json()
        action_index = int(data.get("action_index", len(actions) - 1))
        log_prob = float(data.get("log_prob", 0.0))
        return PolicyStep(action_index=action_index, log_prob=log_prob, data=None)

    def seq_log_prob(self, steps: Sequence[tuple[Any, int]]) -> torch.Tensor:
        # Remote engines typically do not return token-level probabilities.
        return torch.tensor(0.0, dtype=torch.float32)

    def clone_policy(self) -> "HTTPPolicy":
        return HTTPPolicy(
            endpoint=self.cfg.endpoint,
            headers=self.cfg.headers,
            timeout=self.cfg.timeout,
        )

