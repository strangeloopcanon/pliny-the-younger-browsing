#!/usr/bin/env python3
"""
Reward shaping for the browsing environment.

We provide step penalties, terminal success, and optional shaping for
reference-path alignment and action type consistency.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional


class RewardConfig:
    def __init__(
        self,
        step_penalty: float = -0.01,
        success_reward: float = 1.0,
        early_stop_penalty: float = -0.1,
        wrong_transition_penalty: float = -0.05,
        type_match_bonus: float = 0.02,
        on_path_bonus: float = 0.05,
        reward_module: Optional[str] = None,
        reward_class: Optional[str] = None,
        reward_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.early_stop_penalty = early_stop_penalty
        self.wrong_transition_penalty = wrong_transition_penalty
        self.type_match_bonus = type_match_bonus
        self.on_path_bonus = on_path_bonus
        self.reward_module = reward_module
        self.reward_class = reward_class
        self.reward_config = reward_config or {}


def compute_step_reward(
    cfg: RewardConfig,
    *,
    is_terminal: bool,
    reached_goal: bool,
    chose_stop: bool,
    chose_read: bool,
    chosen_edge: Optional[Dict[str, str]],
    ref_next_url: Optional[str],
    ref_next_type: Optional[str],
) -> float:
    # Base step penalty unless terminal success (we still apply penalty; caller may sum differently)
    r = cfg.step_penalty
    if is_terminal and reached_goal:
        r += cfg.success_reward
        return r
    if chose_stop and not reached_goal:
        r += cfg.early_stop_penalty
    if chosen_edge is None:
        # invalid transition
        r += cfg.wrong_transition_penalty
    else:
        if ref_next_url and chosen_edge.get("target") == ref_next_url:
            r += cfg.on_path_bonus
        if ref_next_type and chosen_edge.get("type") == ref_next_type:
            r += cfg.type_match_bonus
    return r


def load_custom_reward(cfg: RewardConfig):
    if not cfg.reward_module or not cfg.reward_class:
        return None
    module = importlib.import_module(cfg.reward_module)
    RewardClass = getattr(module, cfg.reward_class)
    return RewardClass(**cfg.reward_config)
