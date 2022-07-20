import os
from rlpyt.agents.pg.categorical import *
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network.network import PPORelocNet
import numpy as np
from random import sample
import torch

###
# the agent for rlpyt runtime
###

class PPOAgent(CategoricalPgAgent):
    def __init__(self, greedy_eval, ModelCls=PPORelocNet, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
        self.greedy_eval = greedy_eval

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            observation_shape=env_spaces.observation.shape, action_size=env_spaces.action.n,
        )

    def step(self, observation, prev_action, prev_reward):
        action, agent_info = super().step(observation, prev_action, prev_reward)
        if self._mode == "eval" and self.greedy_eval:
            action = torch.argmax(agent_info.dist_info.prob, dim=-1)
        return AgentStep(action=action, agent_info=agent_info)


class PPOLSTMAgent(RecurrentCategoricalPgAgent):
    def __init__(self, greedy_eval, ModelCls=PPORelocNet, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
        self.greedy_eval = greedy_eval

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            observation_shape=env_spaces.observation.shape, action_size=env_spaces.action.n,
        )

    def step(self, observation, prev_action, prev_reward):
        action, agent_info = super().step(observation, prev_action, prev_reward)
        if self._mode == "eval" and self.greedy_eval:
            action = torch.argmax(agent_info.dist_info.prob, dim=-1)
        return AgentStep(action=action, agent_info=agent_info)
