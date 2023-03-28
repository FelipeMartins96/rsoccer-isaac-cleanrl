import gym
import numpy as np


class SingleAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = env.device
        self._action_space = gym.spaces.Box(-1.0, 1.0, (env.num_actions,))
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (env.num_obs,))
        self.action_buf = env.dof_velocity_buf.clone()

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return {'obs': observations['obs'][:, 0, 0, :]}

    def step(self, action):
        self.action_buf[:, 0, 0, :] = action
        observations, rewards, dones, infos = super().step(self.action_buf)
        return (
            {'obs': observations['obs'][:, 0, 0, :]},
            rewards[:, 0, 0],
            dones,
            infos,
        )
