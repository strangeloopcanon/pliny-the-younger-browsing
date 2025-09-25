import torch

from rl.policy_hash import HashSoftmaxPolicy
from rl.utils import parse_actions


def test_hash_policy_act():
    policy = HashSoftmaxPolicy(device=torch.device("cpu"))
    observation = "ACTIONS:\n[1] internal_navigate -> foo\n[2] STOP"
    actions = parse_actions(observation)
    step = policy.act(observation, goal_url="http://example.com", actions=actions)
    assert step.action_index in {0, 1}
    assert isinstance(step.log_prob, float)
