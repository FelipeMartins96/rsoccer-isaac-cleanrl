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
    bt_run = api.run(f"ppo-june-exps/{args.run_id}")
    
    config = dict(bt_run.config)
    save_path = f"runs/validation/{config['exp_name']}/{args.env_id}/{args.run_id}"
    
    config.update({
        "env_id": args.env_id
    })
    run = wandb.init(
        project='ppo-june-exps-validation',
        monitor_gym=False,
        name=f"{config['exp_name']}-{args.env_id}",
        group=f"{config['exp_name']}-{args.env_id}",
        config=config,
    )

    #TODO: remove is_old
    blue_team = get_team(run_name, args.model_path, bt_run.config['hierarchical'], is_old=False)

    # 3 done cases, len by case:
    totals = {
        'matches': 0,
        'match_steps': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'len_wins': 0,
        'len_losses': 0,
        'len_draws': 0,
    }

    for team in BASELINE_TEAMS:
        team_totals = {
            'matches': 0,
            'match_steps': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'len_wins': 0,
            'len_losses': 0,
            'len_draws': 0,
        }
        for seed in BASELINE_TEAMS[team]:
            print(f"Playing {team} {seed}")
            yellow_team = BASELINE_TEAMS[team][seed]
            results = play_matches(envs, blue_team, yellow_team, 10000, f"{save_path}/val_{team}_{seed}")

            team_totals['matches'] += results['matches']
            team_totals['match_steps'] += results['match_steps']
            team_totals['wins'] += results['wins']
            team_totals['losses'] += results['losses']
            team_totals['draws'] += results['draws']
            team_totals['len_wins'] += results['len_wins']
            team_totals['len_losses'] += results['len_losses']
            team_totals['len_draws'] += results['len_draws']
            
            if team == 'zero' or team == 'ou':
                run.log({f"Media/Deterministic/{team}": wandb.Video(f"{save_path}/val_{team}_{seed}/video.000-step-0.mp4")})
            else:
                run.log({f"Media/{team}/{seed}": wandb.Video(f"{save_path}/val_{team}_{seed}/video.000-step-0.mp4")})

        
        run.summary[f"{team}/Score"] = (team_totals['wins'] - team_totals['losses']) / team_totals['matches']
        run.summary[f"{team}/Lenght"] = team_totals['match_steps'] / team_totals['matches']
        run.summary[f"{team}/Win Rate"] = team_totals['wins'] / team_totals['matches']
        run.summary[f"{team}/Loss Rate"] = team_totals['losses'] / team_totals['matches']
        run.summary[f"{team}/Draw Rate"] = team_totals['draws'] / team_totals['matches']
        run.summary[f"{team}/Lenght Wins"] = team_totals['len_wins'] / team_totals['wins']
        run.summary[f"{team}/Lenght Losses"] = team_totals['len_losses'] / team_totals['losses']
        run.summary[f"{team}/Lenght Draws"] = team_totals['len_draws'] / team_totals['draws']
    
        totals['matches'] += team_totals['matches']
        totals['match_steps'] += team_totals['match_steps']
        totals['wins'] += team_totals['wins']
        totals['losses'] += team_totals['losses']
        totals['draws'] += team_totals['draws']
        totals['len_wins'] += team_totals['len_wins']
        totals['len_losses'] += team_totals['len_losses']
        totals['len_draws'] += team_totals['len_draws']

    run.summary["Score"] = (totals['wins'] - totals['losses']) / totals['matches']
    run.summary["Lenght"] = totals['match_steps'] / totals['matches']
    run.summary["Win Rate"] = totals['wins'] / totals['matches']
    run.summary["Loss Rate"] = totals['losses'] / totals['matches']
    run.summary["Draw Rate"] = totals['draws'] / totals['matches']
    run.summary["Lenght Wins"] = totals['len_wins'] / totals['wins']
    run.summary["Lenght Losses"] = totals['len_losses'] / totals['losses']
    run.summary["Lenght Draws"] = totals['len_draws'] / totals['draws']

    wandb.finish()
