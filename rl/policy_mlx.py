#!/usr/bin/env python3
"""MLX policy wrapper using mlx-genkit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Sequence

import mlx.core as mx
import torch

from mlx_lm import load as mlx_load

from mlx_genkit import GenerationConfig, generate, sequence_logprob
from mlx_genkit.loader import auto_load

from .policy_base import PolicyStep, SequencePolicy
from .utils import ActionInfo


logger = logging.getLogger(__name__)


@dataclass
class MLXStepData:
    token_ids: mx.array
    prompt_length: int


class MLXPolicy(SequencePolicy):
    def __init__(
        self,
        model_id: str,
        cache_dir: str = "mlx_cache",
        quantize: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.quantize = quantize
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed

        _, _, local_path = auto_load(
            model_id,
            cache_dir=cache_dir,
            quantize=quantize,
            load_model=False,
        )
        logger.info("Loading MLX model from %s", local_path)
        self.model, self.tokenizer = mlx_load(local_path)

    def _prompt(self, observation: str) -> str:
        return (
            "You are a browsing agent. Read the action list and reply with only the number of the chosen action.\n\n"
            f"{observation}\n\nAnswer with the action index only:"
        )

    def act(
        self,
        observation: str,
        goal_url: str,
        actions: List[ActionInfo],
        *,
        sample: bool = True,
    ) -> PolicyStep:
        prompt = self._prompt(observation)
        cfg = GenerationConfig(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )
        output = generate(self.model, self.tokenizer, prompt, cfg)
        text = output["text"]
        action_index = self._parse_action(text, len(actions))
        token_ids = mx.array(output["tokens"], dtype=mx.int32)
        prompt_length = len(self.tokenizer.encode(prompt))
        step_data = MLXStepData(token_ids=token_ids, prompt_length=prompt_length)
        log_prob = float(self._sequence_log_prob(step_data).item())
        return PolicyStep(action_index=action_index, log_prob=log_prob, data=step_data)

    def _parse_action(self, text: str, num_actions: int) -> int:
        digits = ''.join(ch for ch in text if ch.isdigit())
        if digits:
            value = int(digits)
            if 1 <= value <= num_actions:
                return value - 1
        return max(num_actions - 1, 0)

    def _sequence_log_prob(self, data: MLXStepData) -> mx.array:
        # Build tokens and labels arrays for sequence_logprob
        token_ids = data.token_ids.reshape(1, -1)
        labels = mx.array(token_ids)
        # ignore prompt positions
        labels[:, : data.prompt_length] = -100
        return sequence_logprob(self.model, token_ids, labels, reduction="sum")

    def seq_log_prob(self, steps: Sequence[tuple[Any, int]]) -> torch.Tensor:
        lp = 0.0
        for data, _ in steps:
            assert isinstance(data, MLXStepData)
            lp += float(self._sequence_log_prob(data).item())
        return torch.tensor(lp, dtype=torch.float32)

    def clone_policy(self) -> "MLXPolicy":
        clone = MLXPolicy(
            model_id=self.model_id,
            cache_dir=self.cache_dir,
            quantize=self.quantize,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        return clone
