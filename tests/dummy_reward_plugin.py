class ConstantReward:
    def __init__(self, value=0.0):
        self.value = value

    def calculate_step_reward(self, **kwargs):
        return self.value
