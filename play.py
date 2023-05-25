import gym
import torch
from ppo_continuous_action_isaacgym import ExtractObsWrapper, Agent
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np

from envs.wrappers import OLD_Agent

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
    def __call__(self, act, obs, envs):
        pass


class TeamZero(Team):
    def __call__(self, act, obs, envs):
        act[:] *= 0


class TeamOU(Team):
    def __call__(self, act, obs, envs):
        act[:] = random_ou(act)


class TeamAgent(Team):
    def __init__(self, path, env_d, is_old):
        # TODO: remove is old
        agent_dict = torch.load(path)
        if is_old:
            self.agent = OLD_Agent(env_d).to('cuda:0')
            self.agent.load_state_dict(agent_dict)
        else:
            self.agent = Agent(
                n_obs=agent_dict['n_obs'],
                n_acts=agent_dict['n_acts'],
                h_units=agent_dict['h_units'],
                h_layers=agent_dict['h_layers'],
            ).to('cuda:0')
            self.agent.load_state_dict(agent_dict['state_dict'])


class TeamSA(TeamAgent):
    def __call__(self, act, obs, envs):
        act[:] = random_ou(act)
        act[:, 0, :] = self.agent.get_action_and_value(obs[:, 0, :])[0]


class TeamCMA(TeamAgent):
    def __call__(self, act, obs, envs):
        act[:] = self.agent.get_action_and_value(obs[:, 0, :])[0].view(-1, 3, 2)


class TeamDMA(TeamAgent):
    def __call__(self, act, obs, envs):
        act[:] = self.agent.get_action_and_value(obs)[0]


from isaacgym.torch_utils import get_euler_xyz


@torch.jit.script
def compute_goto_obs(tgt_pos, r_pos, r_vel, r_quats, r_w, r_acts):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    rbt_angles = get_euler_xyz(r_quats.reshape(-1, 4))[2].view(-1, 3, 1)
    obs = torch.cat(
        (
            tgt_pos,
            r_pos,
            r_vel,
            torch.cos(rbt_angles),
            torch.sin(rbt_angles),
            r_w,
            r_acts,
        ),
        dim=-1,
    ).squeeze()
    return obs


class TeamAgent_HRL(Team):
    def __init__(self, path, env_d):
        dummy_env = namedtuple(
            'dummy_env', ['single_observation_space', 'single_action_space']
        )
        self.last_manager_acts = None
        agent_dict = torch.load(path)
        self.manager = Agent(
            n_obs=agent_dict['n_obs'],
            n_acts=agent_dict['n_acts'],
            h_units=agent_dict['h_units'],
            h_layers=agent_dict['h_layers'],
        ).to('cuda:0')
        self.manager.load_state_dict(agent_dict['state_dict'])

        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (11,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        # TODO: Adaptar para novos param agent
        self.worker = OLD_Agent(env_d).to('cuda:0')
        self.worker.load_state_dict(torch.load('1-agent.pt'))
        self.field_size = torch.tensor(
            [0.85, 0.65], device='cuda:0', dtype=torch.float32, requires_grad=False
        )


class TeamSA_HRL(TeamAgent_HRL):
    def __call__(self, act, obs, envs):
        # Fill with random actions
        act[:] = random_ou(act)

        # Reset last manager actions if needed
        if self.last_manager_acts is None:
            self.last_manager_acts = torch.zeros_like(act)
        else:
            env_ids = envs.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.last_manager_acts[env_ids] *= 0

        # Infer manager actions, with clamping
        manager_acts = self.manager.get_action_and_value(obs)[0]
        # Clamp and scale manager acts
        manager_acts = torch.clamp(manager_acts, -1.0, 1.0) * self.field_size
        # Render manager acts
        if envs.view is not None:
            envs.view.set_targets(manager_acts[0, 0:1])

        # Get worker obs
        worker_obs = compute_goto_obs(
            manager_acts,
            envs.robots_pos[:, 0],
            envs.robots_vel[:, 0],
            envs.robots_quats[:, 0],
            envs.robots_ang_vel[:, 0],
            self.last_manager_acts,
        )

        # Infer worker actions
        act[:, 0] = self.worker.get_action_and_value(worker_obs)[0][:, 0]
        self.last_manager_acts[:] = manager_acts


class TeamCMA_HRL(TeamAgent_HRL):
    def __call__(self, act, obs, envs):
        # Fill with random actions
        act[:] = random_ou(act)

        # Reset last manager actions if needed
        if self.last_manager_acts is None:
            self.last_manager_acts = torch.zeros_like(act)
        else:
            env_ids = envs.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.last_manager_acts[env_ids] *= 0

        # Infer manager actions, with clamping
        manager_acts = self.manager.get_action_and_value(obs[:, 0])[0].view(-1, 3, 2)
        # Clamp and scale manager acts
        manager_acts = torch.clamp(manager_acts, -1.0, 1.0) * self.field_size
        # Render manager acts
        if envs.view is not None:
            envs.view.set_targets(manager_acts[0])

        # Get worker obs
        worker_obs = compute_goto_obs(
            manager_acts,
            envs.robots_pos[:, 0],
            envs.robots_vel[:, 0],
            envs.robots_quats[:, 0],
            envs.robots_ang_vel[:, 0],
            self.last_manager_acts,
        )

        # Infer worker actions
        act[:] = self.worker.get_action_and_value(worker_obs)[0]
        self.last_manager_acts[:] = manager_acts


class TeamDMA_HRL(TeamAgent_HRL):
    def __call__(self, act, obs, envs):
        # Fill with random actions
        act[:] = random_ou(act)

        # Reset last manager actions if needed
        if self.last_manager_acts is None:
            self.last_manager_acts = torch.zeros_like(act)
        else:
            env_ids = envs.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.last_manager_acts[env_ids] *= 0

        # Infer manager actions, with clamping
        manager_acts = self.manager.get_action_and_value(obs)[0]
        # Clamp and scale manager acts
        manager_acts = torch.clamp(manager_acts, -1.0, 1.0) * self.field_size
        # Render manager acts
        if envs.view is not None:
            envs.view.set_targets(manager_acts[0])

        # Get worker obs
        worker_obs = compute_goto_obs(
            manager_acts,
            envs.robots_pos[:, 0],
            envs.robots_vel[:, 0],
            envs.robots_quats[:, 0],
            envs.robots_ang_vel[:, 0],
            self.last_manager_acts,
        )

        # Infer worker actions
        act[:] = self.worker.get_action_and_value(worker_obs)[0]
        self.last_manager_acts[:] = manager_acts


def get_team(algo, path=None, hrl=False):
    # create dummy env named tuple with single observation and action spaces
    dummy_env = namedtuple(
        'dummy_env', ['single_observation_space', 'single_action_space']
    )

    if algo == 'ppo-sa':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        if not hrl:
            return TeamSA(path, env_d, is_old=True) # TODO: REMOVE IS OLD
        else:
            return TeamSA_HRL(path, env_d)
    elif algo == 'ppo-sa-x3':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        if not hrl:
            return TeamDMA(path, env_d, is_old=True) # TODO: REMOVE IS OLD
        else:
            return TeamDMA_HRL(path, env_d)
    elif algo == 'ppo-dma':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        if not hrl:
            return TeamDMA(path, env_d, is_old=True) # TODO: REMOVE IS OLD
        else:
            return TeamDMA_HRL(path, env_d)
    elif algo == 'ppo-cma':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (6,)),
        )
        if not hrl:
            return TeamCMA(path, env_d, is_old=True) # TODO: REMOVE IS OLD
        else:
            return TeamCMA_HRL(path, env_d)
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
            name_prefix="video.000",
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
        blue_team(action_buf[:, 0], obs[:, 0], envs)
        # print(act)
        yellow_team(action_buf[:, 1], obs[:, 1], envs)
        obs, rew, dones, info = envs.step(action_buf)

        env_ids = dones[:1065].nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids):
            ep_count += len(env_ids)
            rew_sum += rew[env_ids, 0, 0, 0].sum().item()
            len_sum += info['progress_buffer'][env_ids].sum().item()

    return rew_sum / ep_count, len_sum / ep_count
