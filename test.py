import gym
import isaacgym  # noqa
import isaacgymenvs
import torch
from envs.vss_goto import VSSGoTo
# env setup
from hydra import compose, initialize
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from play import play_matches, get_team
import argparse

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="2k8d2fxm")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--net-path", type=str, default="base_nets/exp000_ppo-sa_10/agent.pt")
    parser.add_argument("--algo", type=str, default="ppo-sa")
    args = parser.parse_args()
    # fmt: on
    return args

if __name__ == "__main__":
    args = parse_args()
    N_GAMES = 10000
    SEED = args.seed
    NET_PATH = args.net_path
    RUN_ID = args.run_id
    ALGO = args.algo

    with initialize(config_path="envs"):
        cfg = compose(config_name="vss_goto")
    cfg = omegaconf_to_dict(cfg)

    cfg['env']['numEnvs'] = 9
    envs = VSSGoTo(
        cfg=cfg,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=False,
        virtual_screen_capture=False,
        force_render=True,
    )

    import pdb; pdb.set_trace()
    
    actions = torch.zeros_like(envs.dof_velocity_buf)
    while True:
        envs.step(actions)