import gym
import torch
from ppo_continuous_action_isaacgym import ExtractObsWrapper, Agent
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np


import torch.nn as nn
from ppo_continuous_action_isaacgym import layer_init
from torch.distributions.normal import Normal
class OLD_Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


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
    def __call__(self, act, obs, envs=None):
        return act * 0


class TeamOU(Team):
    def __call__(self, act, obs, envs=None):
        return random_ou(act)


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
            ).to('cuda:0')
            self.agent.load_state_dict(agent_dict['state_dict'])


class TeamSA(TeamAgent):
    def __call__(self, act, obs, envs=None):
        act[:] = random_ou(act)
        act[:, 0, :] = self.agent.get_action_and_value(obs[:, 0, :])[0]
        return act


class TeamCMA(TeamAgent):
    def __call__(self, act, obs, envs=None):
        return self.agent.get_action_and_value(obs[:, 0, :])[0].view(-1, 3, 2)


class TeamDMA(TeamAgent):
    def __call__(self, act, obs, envs=None):
        return self.agent.get_action_and_value(obs)[0]


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


def get_team(algo, path=None, hrl=False, is_old=False):
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
            return TeamSA(path, env_d, is_old=is_old) # TODO: REMOVE IS OLD
        else:
            return TeamSA_HRL(path, env_d)
    elif algo == 'ppo-sa-x3':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        if not hrl:
            return TeamDMA(path, env_d, is_old=is_old) # TODO: REMOVE IS OLD
        else:
            return TeamDMA_HRL(path, env_d)
    elif algo == 'ppo-dma':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
        )
        if not hrl:
            return TeamDMA(path, env_d, is_old=is_old) # TODO: REMOVE IS OLD
        else:
            return TeamDMA_HRL(path, env_d)
    elif algo == 'ppo-cma':
        env_d = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (52,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (6,)),
        )
        if not hrl:
            return TeamCMA(path, env_d, is_old=is_old) # TODO: REMOVE IS OLD
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
        '20': get_team('ppo-sa', 'exp102/SA/agent-yqq35j72.pt'), # TODO: remove is old
        '30': get_team('ppo-sa', 'exp102/SA/agent-frffb51f.pt'), # TODO: remove is old
        '50': get_team('ppo-sa', 'exp102/SA/agent-fs66dxjn.pt'), # TODO: remove is old
    },
    'ppo-sa-x3': {
        '20': get_team('ppo-sa-x3', 'exp102/SA/agent-yqq35j72.pt'), # TODO: remove is old
        '30': get_team('ppo-sa-x3', 'exp102/SA/agent-frffb51f.pt'), # TODO: remove is old
        '50': get_team('ppo-sa-x3', 'exp102/SA/agent-fs66dxjn.pt'), # TODO: remove is old
    },
    'ppo-cma': {
        '20': get_team('ppo-cma', 'exp102/JAL/agent-eiqacdzv.pt'), # TODO: remove is old
        '40': get_team('ppo-cma', 'exp102/JAL/agent-0arp1jm1.pt'), # TODO: remove is old
        '50': get_team('ppo-cma', 'exp102/JAL/agent-qps0le0n.pt'), # TODO: remove is old
    },
    'ppo-dma': {
        '10': get_team('ppo-dma', 'exp102/IL/agent-qdbxuop3.pt'), # TODO: remove is old
        '20': get_team('ppo-dma', 'exp102/IL/agent-jpjvv3nx.pt'), # TODO: remove is old
        '40': get_team('ppo-dma', 'exp102/IL/agent-49ca8j77.pt'), # TODO: remove is old
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

    results = {
        'matches': 0,
        'match_steps': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'atk_fouls': 0,
        'len_wins': 0,
        'len_losses': 0,
    }
    action_buf = torch.zeros(
        (envs.cfg['env']['numEnvs'],) + envs.action_space.shape, device=envs.device
    )
    obs = envs.reset()
    while results['matches'] < n_matches:
        action_buf[:, 0] = blue_team(action_buf[:, 0], obs[:, 0], envs)
        action_buf[:, 1] = yellow_team(action_buf[:, 1], obs[:, 1], envs)
        obs, rew, dones, info = envs.step(action_buf)

        env_ids = dones[:1065].nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids):
            results['matches'] += len(env_ids)
            goal_score = rew[env_ids, 0, 0, 0]
            atk_fouls = rew[env_ids, 0, 0, 4]

            win_ids = goal_score > 0
            loss_ids = goal_score < 0
            draw_ids = (goal_score == 0) & (atk_fouls == 0)

            results['wins'] += win_ids.sum().item()
            results['losses'] += loss_ids.sum().item()
            results['draws'] += draw_ids.sum().item()
            results['atk_fouls'] += atk_fouls.sum().item()

            done_lengths = info['progress_buffer'][env_ids]
            results['match_steps'] += done_lengths.sum().item()
            results['len_wins'] += done_lengths[win_ids].sum().item()
            results['len_losses'] += done_lengths[loss_ids].sum().item()

    return results
