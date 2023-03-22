import gym
import isaacgym  # noqa
import isaacgymenvs
import torch

from envs.base import BASE

# env setup
from hydra import compose, initialize
from isaacgymenvs.utils.reformat import omegaconf_to_dict

with initialize(config_path="envs"):
    cfg = compose(config_name="base")
cfg = omegaconf_to_dict(cfg)
envs = BASE(
    cfg=cfg,
    rl_device="cuda:0",
    sim_device="cuda:0",
    graphics_device_id=0,
    headless=False,
    virtual_screen_capture=False,
    force_render=True,
)

while 1:
    envs.step(torch.ones((envs.num_envs,) + envs.act_space.shape, device=envs.device))
