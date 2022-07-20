import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network import *

###
# assets for RL algorithm
###


device = 'cuda'

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class RelocCategoricalActor(Actor):
    def __init__(self, acnet, n_action):
        super().__init__()
        self.logits_net = acnet(out_dim=n_action).to(device)

    def _distribution(self, obs):
        obs = obs.to(device)
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class RelocCritic(nn.Module):
    def __init__(self, acnet):
        super().__init__()
        self.v_net = acnet(out_dim=1).to(device)

    def forward(self, obs):
        obs = obs.to(device)
        return torch.squeeze(self.v_net(obs), -1)   # Critical to ensure v has right shape.


class RelocActorCritic(nn.Module):
    def __init__(self, acnet, n_action):
        super().__init__()
        self.pi = RelocCategoricalActor(acnet, n_action)
        self.v = RelocCritic(acnet)

    def step(self, obs):
        """
        self.pi: Actor
        pi: Distribution
        """
        obs = obs.to(device)
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]
