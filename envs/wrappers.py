import gym
import numpy as np
import torch

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

def make_env(args):
    from hydra import compose, initialize
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    with initialize(config_path="."):
        cfg = compose(config_name="vss")
    cfg = omegaconf_to_dict(cfg)
    assert args.cuda
    cfg['env']['numEnvs'] = args.num_envs
    if args.env_id == 'dma':
        assert args.num_envs % 3 == 0
        cfg['env']['numEnvs'] = int(args.num_envs / 3)
    
    from envs.vss import VSS
    envs = VSS(
            cfg=cfg,
            rl_device="cuda:0",
            sim_device="cuda:0",
            graphics_device_id=0,
            headless=True,
            virtual_screen_capture=False,
            force_render=False,
        )
    if args.hierarchical:
        envs = HRL(envs)

    wrappers = {
        'sa': SingleAgent,
        'cma': CMA,
        'dma': DMA,
    }
    return envs, wrappers[args.env_id](envs)

def make_env_goto(args):
    from hydra import compose, initialize
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    with initialize(config_path="."):
        cfg = compose(config_name="vss_goto")
    cfg = omegaconf_to_dict(cfg)
    assert args.cuda
    cfg['env']['numEnvs'] = args.num_envs

    if not args.terminal_rw:
        cfg['env']['rew_weights']['terminal'] = 0.0
    if not args.check_angle:
        cfg['env']['thresholds']['angle'] = 5
    if not args.check_speed:
        cfg['env']['thresholds']['speed'] = 5

    
    from envs.vss_goto import VSSGoTo
    envs = VSSGoTo(
            cfg=cfg,
            rl_device="cuda:0",
            sim_device="cuda:0",
            graphics_device_id=0,
            headless=False if args.capture_video else True,
            virtual_screen_capture=args.capture_video,
            force_render=False,
        )
    return envs

class RecordEpisodeStatisticsTorchVSS(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self._num_envs = getattr(env, "num_environments", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self._num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self._num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos['rews']
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones.unsqueeze(1)
        self.episode_lengths *= 1 - dones
        infos["r"] = {
            'goal' : self.returned_episode_returns[:,0],
            'grad' : self.returned_episode_returns[:,1],
            'move' : self.returned_episode_returns[:,2],
            'energy' : self.returned_episode_returns[:,3],
            'return' : self.returned_episode_returns.sum(1),
        }
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self._num_envs = getattr(env, "num_environments", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self._num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self._num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(self._num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self._num_envs, dtype=torch.int32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

class SingleAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (env.num_actions,))
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (env.num_obs,))
        self.action_buf = torch.zeros((env.num_envs,) + env.action_space.shape, device=env.rl_device, dtype=torch.float32, requires_grad=False)
        self.act_view = self.action_buf[:, 0, 0, :]
        env.num_controlled = 1

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return {'obs': observations['obs'][:, 0, 0, :]}

    def step(self, action):
        self.action_buf[:] = random_ou(self.action_buf)
        self.act_view[:] = action
        observations, rewards, dones, infos = super().step(self.action_buf)
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.action_buf[env_ids] *= 0
        infos['rews'] = rewards[:, 0, 0]
        infos['terminal_observation'] = infos['terminal_observation'][:, 0, 0, :]
        return (
            {'obs': observations['obs'][:, 0, 0, :]},
            infos['rews'].sum(-1),
            dones,
            infos,
        )


class CMA(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._num_envs = getattr(env, "num_envs", 1)
        self.device = env.device
        num_actions = env.num_actions * 3
        self._action_space = gym.spaces.Box(-1.0, 1.0, (num_actions,))
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (env.num_obs,))
        self.action_buf = torch.zeros((env.num_envs,) + env.action_space.shape, device=env.rl_device, dtype=torch.float32, requires_grad=False)
        self.act_view = self.action_buf[:, 0, :, :].view(-1, num_actions)
        env.num_controlled = 3

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return {'obs': observations['obs'][:, 0, 0, :]}

    def step(self, action):
        self.action_buf[:] = random_ou(self.action_buf)
        self.act_view[:] = action
        observations, rewards, dones, infos = super().step(self.action_buf)
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.action_buf[env_ids] *= 0
        infos['rews'] = rewards[:, 0, :].mean(1)
        infos['terminal_observation'] = infos['terminal_observation'][:, 0, 0, :]
        
        return (
            {'obs': observations['obs'][:, 0, 0, :]},
            infos['rews'].sum(-1),
            dones,
            infos,
        )


class DMA(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_buf = torch.zeros((env.num_envs,) + env.action_space.shape, device=env.rl_device, dtype=torch.float32, requires_grad=False)
        setattr(env, "num_environments", getattr(env, "num_envs", 1) * 3)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (env.num_actions,))
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (env.num_obs,))
        env.num_controlled = 3

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        return {'obs': observations['obs'][:, 0, :, :].reshape(-1, self.env.num_obs)}

    def step(self, action):
        self.action_buf[:] = random_ou(self.action_buf)
        self.action_buf[:, 0, :, :] = action.view(-1, 3, 2)
        observations, rewards, dones, infos = super().step(self.action_buf)
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.action_buf[env_ids] *= 0
        infos['terminal_observation'] = infos['terminal_observation'][:, 0, :, :].reshape(-1, self.env.num_obs)
        infos['progress_buffer'] = infos['progress_buffer'].unsqueeze(1).repeat_interleave(3)
        infos['time_outs'] = infos['time_outs'].unsqueeze(1).repeat_interleave(3)
        infos['rews'] = rewards[:, 0, :].reshape(-1, 4)
        
        return (
            {'obs': observations['obs'][:, 0, :, :].reshape(-1, self.env.num_obs)},
            infos['rews'].sum(-1),
            dones.unsqueeze(1).repeat_interleave(3),
            infos,
        )

from ppo_continuous_action_isaacgym import Agent
from collections import namedtuple
dummy_env = namedtuple(
        'dummy_env', ['single_observation_space', 'single_action_space']
    )
class HRL(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        manager_act_size = 2
        worker_obs_size = 11
        self.field_size = torch.tensor([env.field_width + env.goal_width*2, env.field_height], device=env.rl_device, dtype=torch.float32, requires_grad=False) / 2
        self._action_space = gym.spaces.Box(-1.0, 1.0, env.action_space.shape[:-1] + (manager_act_size,))
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (env.num_obs,))
        self._worker_act_buf = torch.zeros((env.num_envs,) + env.action_space.shape, device=env.rl_device, dtype=torch.float32, requires_grad=False)
        self._manager_act_buf = torch.zeros((env.num_envs,) + self._action_space.shape, device=env.rl_device, dtype=torch.float32, requires_grad=False)
        self._worker_obs_buf = torch.zeros((env.num_envs, 2, 3, worker_obs_size), device=env.rl_device, dtype=torch.float32, requires_grad=False)
        
        # LOAD AGENT
        d_env = dummy_env(
            single_observation_space=gym.spaces.Box(-np.inf, np.inf, (worker_obs_size,)),
            single_action_space=gym.spaces.Box(-1.0, 1.0, (manager_act_size,)),
        )
        agent = Agent(d_env)
        self.agent = agent
        self.agent.load_state_dict(torch.load('1-agent.pt'))
        self.agent.to(env.device)

        self.num_controlled = None
    
    def step(self, actions):
        assert self.num_controlled is not None # TODO: case if none

        # Reset previous actions
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self._manager_act_buf[env_ids] *= 0
        
        # Clamp actions and scale to field size
        self._worker_act_buf[:] = actions
        
        actions = torch.clamp(actions, -1.0, 1.0) * self.field_size
        
        self._worker_obs_buf[:] = compute_goto_obs(
            actions,
            self.env.robots_pos,
            self.env.robots_vel,
            self.env.robots_quats,
            self.env.robots_ang_vel,
            self._manager_act_buf
        )
        self._manager_act_buf[:] = actions

        # Get worker actions
        self._worker_act_buf[:, 0:1, 0:self.num_controlled] = self.agent.get_action_and_value(self._worker_obs_buf)[0][:, 0:1, 0:self.num_controlled]
        return super().step(self._worker_act_buf)

    def render(self, mode="rgb_array"):
        if self.view is not None:
            self.view.set_targets(self._manager_act_buf[0, 0, 0:self.num_controlled])
        
        return super().render(mode)

from isaacgym.torch_utils import get_euler_xyz
@torch.jit.script
def compute_goto_obs(tgt_pos, r_pos, r_vel, r_quats, r_w, r_acts):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    rbt_angles = get_euler_xyz(r_quats.reshape(-1, 4))[2].view(-1, 2, 3, 1)
    mirror_tensor = torch.tensor(
        [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        dtype=torch.float,
        device="cuda:0",
        requires_grad=False,
    )
    obs = torch.cat(
        (
            tgt_pos,
            r_pos,
            r_vel,
            torch.cos(rbt_angles),
            torch.sin(rbt_angles),
            r_w,
            r_acts
        ), dim=-1
    ).squeeze()
    obs[:, 1] *= mirror_tensor
    return obs