import gym
import isaacgym  # noqa
import isaacgymenvs
import torch

from envs.vss import VSS

# env setup
from hydra import compose, initialize
from isaacgymenvs.utils.reformat import omegaconf_to_dict

from envs.wrappers import SingleAgent, CMA, DMA

with initialize(config_path="envs"):
    cfg = compose(config_name="vss")
cfg = omegaconf_to_dict(cfg)

cfg['env']['numEnvs'] = 3

envs = VSS(
    cfg=cfg,
    rl_device="cuda:0",
    sim_device="cuda:0",
    graphics_device_id=0,
    headless=False,
    virtual_screen_capture=False,
    force_render=True,
)

# envs = SingleAgent(envs)
# envs = CMA(envs)
envs = DMA(envs)
envs.reset()
act = torch.ones((envs.num_envs,) + envs.action_space.shape, device=envs.device)
act[..., 0] = -2.0

while 1:
    obs, rew, dones, info = envs.step(act)
