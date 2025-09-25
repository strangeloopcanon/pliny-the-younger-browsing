#!/usr/bin/env python3
"""HuggingFace causal LM policy for GSPO."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .policy_base import PolicyStep, SequencePolicy
from .utils import ActionInfo


logger = logging.getLogger(__name__)


@dataclass
class HFStepData:
    input_ids: List[int]
    prompt_length: int


class HFPolicy(SequencePolicy):
    def __init__(
        self,
        model_name: str = "sshleifer/tiny-gpt2",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 20,
        max_new_tokens: int = 4,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device or torch.device("cpu")

        logger.info("Loading policy model %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def _prompt(self, observation: str) -> str:
        return (
            "You are controlling a browsing agent."
            " Read the action list and reply with only the number of the chosen action.\n\n"
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
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = encoded.input_ids.shape[1]

        with torch.no_grad():
            generation = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated_ids = generation.sequences[0][prompt_length:]
        if len(generated_ids) == 0:
            generated_ids = torch.tensor([self.tokenizer.eos_token_id], device=self.device)

        log_probs = []
        with torch.no_grad():
            for step_scores, token_id in zip(generation.scores, generated_ids):
                log_probs.append(F.log_softmax(step_scores[0], dim=-1)[token_id])
        step_log_prob = torch.stack(log_probs).sum()

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        action_index = self._parse_action(text, len(actions))

        step_data = HFStepData(
            input_ids=(encoded.input_ids[0].tolist() + generated_ids.tolist()),
            prompt_length=prompt_length,
        )

        return PolicyStep(
            action_index=action_index,
            log_prob=float(step_log_prob.item()),
            data=step_data,
        )

    def _parse_action(self, text: str, num_actions: int) -> int:
        digits = ''.join(ch for ch in text if ch.isdigit())
        if digits:
            value = int(digits)
            if 1 <= value <= num_actions:
                return value - 1
        # fallback to STOP (last index)
        return max(num_actions - 1, 0)

    def seq_log_prob(self, steps: Sequence[tuple[Any, int]]) -> torch.Tensor:
        total = torch.zeros(1, device=self.device)
        for data, _ in steps:
            assert isinstance(data, HFStepData)
            input_ids = torch.tensor(data.input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :]
            target = input_ids[:, 1:]
            start = data.prompt_length - 1
            generated_length = len(data.input_ids) - data.prompt_length
            logits = logits[:, start : start + generated_length, :]
            target = target[:, start : start + generated_length]
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
            total = total + gathered.sum()
        return total.squeeze(0)

    def clone_policy(self) -> "HFPolicy":
        clone = copy.deepcopy(self)
        clone.model.to(self.device)
        return clone
