import json

import requests

from rl.policy_http import HTTPPolicy
from rl.utils import parse_actions


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_http_policy(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        assert "observation" in json
        return DummyResponse({"action_index": 0, "log_prob": -0.5})

    monkeypatch.setattr(requests, "post", fake_post)

    policy = HTTPPolicy(endpoint="https://example.com/act")
    observation = "ACTIONS:\n[1] internal_navigate -> foo\n[2] STOP"
    actions = parse_actions(observation)
    step = policy.act(observation, goal_url="http://example.com", actions=actions)
    assert step.action_index == 0
    assert step.log_prob == -0.5
