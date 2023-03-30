import gym
import numpy as np


class SingleAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (env.num_actions,))
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (env.num_obs,))
        self.action_buf = env.dof_velocity_buf.clone()
        self.act_view = self.action_buf[:, 0, 0, :]

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return {'obs': observations['obs'][:, 0, 0, :]}

    def step(self, action):
        self.act_view[:] = action
        observations, rewards, dones, infos = super().step(self.action_buf)
        infos['terminal_observation'] = infos['terminal_observation'][:, 0, 0, :]
        
        return (
            {'obs': observations['obs'][:, 0, 0, :]},
            rewards[:, 0, 0].sum(-1),
            dones,
            infos,
        )


class CMA(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = env.device
        num_actions = env.num_actions * 3
        self._action_space = gym.spaces.Box(-1.0, 1.0, (num_actions,))
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (env.num_obs,))
        self.action_buf = env.dof_velocity_buf.clone()
        self.act_view = self.action_buf[:, 0, :, :].view(-1, num_actions)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return {'obs': observations['obs'][:, 0, 0, :]}

    def step(self, action):
        self.act_view[:] = action
        observations, rewards, dones, infos = super().step(self.action_buf)
        infos['terminal_observation'] = infos['terminal_observation'][:, 0, 0, :]
        
        return (
            {'obs': observations['obs'][:, 0, 0, :]},
            rewards[:, 0, :].mean(1).sum(-1),
            dones,
            infos,
        )


class DMA(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        setattr(env, "num_environments", getattr(env, "num_envs", 1) * 3)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (env.num_actions,))
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (env.num_obs,))
        self.action_buf = env.dof_velocity_buf.clone()

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return {'obs': observations['obs'][:, 0, :, :].reshape(-1, self.env.num_obs)}

    def step(self, action):
        self.action_buf[:, 0, :, :] = action.view(-1, 3, 2)
        observations, rewards, dones, infos = super().step(self.action_buf)
        infos['terminal_observation'] = infos['terminal_observation'][:, 0, 0, :]
        infos['progress_buffer'] = infos['progress_buffer'].unsqueeze(1).repeat_interleave(3)
        infos['time_outs'] = infos['time_outs'].unsqueeze(1).repeat_interleave(3)
        
        return (
            {'obs': observations['obs'][:, 0, :, :].reshape(-1, self.env.num_obs)},
            rewards[:, 0, :].sum(-1).view(-1),
            dones.unsqueeze(1).repeat_interleave(3),
            infos,
        )
