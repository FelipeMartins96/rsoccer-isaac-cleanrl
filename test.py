import gym
import isaacgym  # noqa
import isaacgymenvs
import torch
from envs.vss_goto import VSSGoTo
from envs.vss import VSS
# env setup
from hydra import compose, initialize
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from play import play_matches, get_team
import argparse
from envs.wrappers import HRL, SingleAgent, CMA, DMA

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
        # cfg = compose(config_name="vss_goto")
        cfg = compose(config_name="vss")
    cfg = omegaconf_to_dict(cfg)

    cfg['env']['numEnvs'] = 3
    # envs = VSSGoTo(
    envs = VSS(
        cfg=cfg,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )
    envs = HRL(envs)
    envs = SingleAgent(envs)
    # envs = CMA(envs)
    # envs = DMA(envs)

    actions = torch.ones((envs.num_environments,) + envs.action_space.shape, device=envs.rl_device) * 2
    while True:
        envs.set_speed_factor(0)
        envs.step(actions)
        envs.render()