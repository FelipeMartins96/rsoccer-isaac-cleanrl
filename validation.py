import argparse
from hydra import compose, initialize
import isaacgym
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import wandb
from envs.vss import VSS
from play import play_matches, get_team, BASELINE_TEAMS


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str,
        help="Wandb run id")
    parser.add_argument("--env-id", type=str,
        help="run env id")
    parser.add_argument("--model-path", type=str,
        help="model path")
    args = parser.parse_args()
    # fmt: on
    return args

def make_env():
    with initialize(config_path="envs/"):
        cfg = compose(config_name="vss_validation")
    cfg = omegaconf_to_dict(cfg)
    return VSS(
        cfg=cfg,
        rl_device="cuda:0",
        sim_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

if __name__ == "__main__":
    args = parse_args()
    run_name = f'ppo-{args.env_id}'

    envs = make_env()

    api = wandb.Api()
    bt_run = api.run(f"ppo-isaac-sfe/{args.run_id}")
    
    config = dict(bt_run.config)
    save_path = f"runs/validation/{config['exp_name']}/{args.env_id}/{args.run_id}"
    
    config.update({
        "env_id": args.env_id,
        "num_envs": envs.num_envs,
    })
    run = wandb.init(
        project='ppo-june-net-validation',
        monitor_gym=False,
        name=f"{config['exp_name']}-{args.env_id}",
        group=f"{config['exp_name']}-{args.env_id}",
        config=config,
    )

    #TODO: remove is_old
    blue_team = get_team(run_name, args.model_path, bt_run.config['hierarchical'], is_old=False)
    total_score = 0
    total_len = 0
    total_count = 0
    for team in BASELINE_TEAMS:
        team_score = 0
        team_len = 0
        for seed in BASELINE_TEAMS[team]:
            print(f"Playing {team} {seed}")
            yellow_team = BASELINE_TEAMS[team][seed]
            rews, lens = play_matches(envs, blue_team, yellow_team, 10000, f"{save_path}/val_{team}_{seed}")
            team_score += rews
            team_len += lens
            total_score += rews
            total_len += lens
            total_count += 1
            if team == 'zero' or team == 'ou':
                run.log({f"Media/Deterministic/{team}": wandb.Video(f"{save_path}/val_{team}_{seed}/video.000-step-0.mp4")})
            else:
                run.log({f"Media/{team}/{seed}": wandb.Video(f"{save_path}/val_{team}_{seed}/video.000-step-0.mp4")})
        run.summary[f"Score/{team}"] = team_score / len(BASELINE_TEAMS[team])
        run.summary[f"Length/{team}"] = team_len / len(BASELINE_TEAMS[team])
    run.summary["Score/Mean"] = total_score / total_count
    run.summary["Length/Mean"] = total_len / total_count

    wandb.finish()
