import gym
import isaacgym  # noqa
import isaacgymenvs
import torch
from envs.vss import VSS
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
    N_GAMES = 15000
    SEED = args.seed
    NET_PATH = args.net_path
    RUN_ID = args.run_id
    ALGO = args.algo

    with initialize(config_path="envs"):
        cfg = compose(config_name="vss")
    cfg = omegaconf_to_dict(cfg)

    cfg['env']['numEnvs'] = 1065
    envs = VSS(
        cfg=cfg,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=False,
        virtual_screen_capture=True,
        force_render=False,
    )
    envs.w_goal = 1.0
    envs.w_grad = 0.0
    envs.w_move = 0.0
    envs.w_energy = 0.0

    from play import BASELINE_TEAMS
    import wandb

    blue_team = get_team(ALGO, NET_PATH)
    run = wandb.init(
        project='ppo-isaac-cleanrl',
        monitor_gym=False,
        id=RUN_ID,
        resume=True,
    )
    new_config = dict(wandb.config)
    wandb.finish()

    total_score = 0
    total_len = 0
    total_count = 0

    for team in BASELINE_TEAMS:
        team_score = 0
        team_len = 0
        for seed in BASELINE_TEAMS[team]:
            print(f"Playing {team} {seed}")
            yellow_team = BASELINE_TEAMS[team][seed]
            rews, lens = play_matches(envs, blue_team, yellow_team, N_GAMES, f"runs/{team}_{seed}")
            team_score += rews
            team_len += lens
            total_score += rews
            total_len += lens
            total_count += 1
            run.log({f"Validation/Media/{team}/{seed}": wandb.Video(f"runs/{team}_{seed}/video.000-step-0.mp4")})
        run.summary[f"Validation/Score/{team}"] = team_score / len(BASELINE_TEAMS[team])
        run.summary[f"Validation/Length/{team}"] = team_len / len(BASELINE_TEAMS[team])
    run.summary["Validation/Score Mean"] = total_score / total_count
    run.summary["Validation/Length Mean"] = total_len / total_count
    # run.summary.update()

    wandb.finish()
    ############################
    if ALGO == "ppo-sa":
        blue_team = get_team('ppo-sa-x3', NET_PATH)
        new_config['env_id'] = 'sa-x3'
        run = wandb.init(
            project='ppo-isaac-cleanrl',
            monitor_gym=False,
            name=f"exp000_ppo-sa-x3_{seed}",
            group="exp000",
            config=new_config
        )
        total_score = 0
        total_len = 0
        total_count = 0

        for team in BASELINE_TEAMS:
            team_score = 0
            team_len = 0
            for seed in BASELINE_TEAMS[team]:
                print(f"Playing {team} {seed}")
                yellow_team = BASELINE_TEAMS[team][seed]
                rews, lens = play_matches(envs, blue_team, yellow_team, N_GAMES, f"runs/{team}_{seed}")
                team_score += rews
                team_len += lens
                total_score += rews
                total_len += lens
                total_count += 1
                run.log({f"Validation/Media/{team}/{seed}": wandb.Video(f"runs/{team}_{seed}/video.000-step-0.mp4")})
            run.summary[f"Validation/Score/{team}"] = team_score / len(BASELINE_TEAMS[team])
            run.summary[f"Validation/Length/{team}"] = team_len / len(BASELINE_TEAMS[team])
        run.summary["Validation/Score Mean"] = total_score / total_count
        run.summary["Validation/Length Mean"] = total_len / total_count
    
    wandb.finish()