import gym
import torch
from ppo_continuous_action_isaacgym import ExtractObsWrapper, Agent
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np


def random_ou(prev):
    ou_theta = 0.1
    ou_sigma = 0.15
    noise = (
        prev
        - ou_theta * prev
        + torch.normal(
            0.0,
            ou_sigma,
            size=prev.size(),
            device=prev.device,
            requires_grad=False,
        )
    )
    return noise.clamp(-1.0, 1.0)


class Team(ABC):
    def __init__(self, path=None, env_d=None):
        pass

    @abstractmethod
    def __call__(self, act, obs):
        pass


class TeamZero(Team):
    def __call__(self, act, obs):
        act[:] *= 0


class TeamOU(Team):
    def __call__(self, act, obs):
        act[:] = random_ou(act)


class TeamAgent(Team):
    def __init__(self, path, env_d):
        self.agent = Agent(env_d).to('cuda:0')
        self.agent.load_state_dict(torch.load(path))


class TeamSA(TeamAgent):
    def __call__(self, act, obs):
        act[:] = random_ou(act)
        act[:, 0, :] = self.agent.get_action_and_value(obs[:, 0, :])[0]


class TeamCMA(TeamAgent):
    def __call__(self, act, obs):
        act[:] = self.agent.get_action_and_value(obs[:, 0, :])[0].view(-1, 3, 2)


class TeamDMA(TeamAgent):
    def __call__(self, act, obs):
        act[:] = self.agent.get_action_and_value(obs)[0]


def get_team(algo, path=None):
    # create dummy env named tuple with single observation and action spaces
    dummy_env = namedtuple(
        'dummy_env', ['single_observation_space', 'single_action_space']
    )

    if algo == 'ppo-sa':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        return TeamSA(path, env_d)
    elif algo == 'ppo-sa-x3':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        return TeamDMA(path, env_d)
    elif algo == 'ppo-dma':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        return TeamDMA(path, env_d)
    elif algo == 'ppo-cma':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (6,)),
        )
        return TeamCMA(path, env_d)
    elif algo == 'zero':
        return TeamZero()
    elif algo == 'ou':
        return TeamOU()
    else:
        raise ValueError(f'Unknown algo: {algo}')


BASELINE_TEAMS = {
    'ppo-sa': {
        '10': get_team('ppo-sa', 'exp000/ppo-sa/agent-0apumq20.pt'),
        '20': get_team('ppo-sa', 'exp000/ppo-sa/agent-g7gguvub.pt'),
        '30': get_team('ppo-sa', 'exp000/ppo-sa/agent-x98y8vhs.pt'),
    },
    'ppo-sa-x3': {
        '10': get_team('ppo-sa-x3', 'exp000/ppo-sa/agent-0apumq20.pt'),
        '20': get_team('ppo-sa-x3', 'exp000/ppo-sa/agent-g7gguvub.pt'),
        '30': get_team('ppo-sa-x3', 'exp000/ppo-sa/agent-x98y8vhs.pt'),
    },
    'ppo-cma': {
        '10': get_team('ppo-cma', 'exp000/ppo-cma/agent-0m1lx8tj.pt'),
        '20': get_team('ppo-cma', 'exp000/ppo-cma/agent-yp5299oc.pt'),
        '30': get_team('ppo-cma', 'exp000/ppo-cma/agent-a8ogk7j9.pt'),
    },
    'ppo-dma': {
        '10': get_team('ppo-dma', 'exp000/ppo-dma/agent-mgeahqxu.pt'),
        '20': get_team('ppo-dma', 'exp000/ppo-dma/agent-qzy50f6t.pt'),
        '30': get_team('ppo-dma', 'exp000/ppo-dma/agent-ba6h8l2t.pt'),
    },
    'zero': {'00': get_team('zero')},
    'ou': {'00': get_team('ou')},
}


def play_matches(envs, blue_team, yellow_team, n_matches, video_path=None):
    envs.reset_buf[:] = 1
    envs.reset_dones()
    if video_path:
        envs.is_vector_env = True
        envs = gym.wrappers.RecordVideo(
            envs,
            video_path,
            step_trigger=lambda step: step == 0,
            video_length=300,
            name_prefix="video.000"
        )
    envs = ExtractObsWrapper(envs)

    ep_count = 0
    rew_sum = 0
    len_sum = 0
    action_buf = torch.zeros(
        (envs.cfg['env']['numEnvs'],) + envs.action_space.shape, device=envs.device
    )
    obs = envs.reset()
    while ep_count < n_matches:
        blue_team(action_buf[:, 0], obs[:, 0])
        # print(act)
        yellow_team(action_buf[:, 1], obs[:, 1])
        obs, rew, dones, info = envs.step(action_buf)

        env_ids = dones[:1065].nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids):
            ep_count += len(env_ids)
            rew_sum += rew[env_ids, 0, 0, 0].sum().item()
            len_sum += info['progress_buffer'][env_ids].sum().item()

    return rew_sum / ep_count, len_sum / ep_count
