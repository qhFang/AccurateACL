from rlpyt.samplers.collections import TrajInfo


class RelocTrajInfo(TrajInfo):
    """Episode-wise info logging"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.SuccRatio = 0
        self.VarSuccRatio = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        if done:
            if getattr(env_info, "succ", 0):
                self.SuccRatio = 1
            if getattr(env_info, "var_succ", 0):
                self.VarSuccRatio = 1
