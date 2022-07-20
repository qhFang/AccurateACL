from gym.envs.registration import register

register(
    id='EnvReloc-v0',
    entry_point='gym_foo.envs:EnvReloc',
    max_episode_steps=200,
)
register(
    id='EnvRelocUncertainty-v0',
    entry_point='gym_foo.envs:EnvRelocUncertainty',
    max_episode_steps=200,
)
