import argparse
from distutils.util import strtobool
from hydra import compose, initialize
import isaacgym
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import wandb
from envs.vss import VSS
from play import play_matches, get_team, BASELINE_TEAMS
import os

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str,
        help="Wandb run id")
    parser.add_argument("--env-id", type=str,
        help="run env id")
    parser.add_argument("--model-path", type=str,
        help="model path")
    parser.add_argument("--atk-foul", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    args = parser.parse_args()
    # fmt: on
    return args

def make_env():
    with initialize(config_path="envs/"):
        cfg = compose(config_name="vss_validation")
    cfg = omegaconf_to_dict(cfg)
    if args.atk_foul:
        cfg['env']['rew_weights']['atk_foul'] = 1.0
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
    bt_run = api.run(f"isaac-reducing-move-bias/{args.run_id}")
    
    config = dict(bt_run.config)
    save_path = f"runs/validation/{config['exp_name']}/{args.env_id}/{args.run_id}"
    
    config.update({
        "env_id": args.env_id
    })
    run = wandb.init(
        project='isaac--validation-2',
        monitor_gym=False,
        name=f"{config['exp_name']}-{args.env_id}",
        group=f"{config['exp_name']}-{args.env_id}",
        config=config,
    )
    
    intermediate = [m for m in os.listdir(os.path.join(os.path.dirname(args.model_path), args.run_id)) if '.pt' in m]
    
    # sort intermediate_models by integer before .pt suffix
    intermediate.sort(key=lambda x: int(x.split('.')[0]))

    for im in intermediate:
        blue_team = get_team(run_name, os.path.join(os.path.dirname(args.model_path), args.run_id, im), bt_run.config['hierarchical'], is_old=False)
        results_zero = play_matches(envs, blue_team, BASELINE_TEAMS['zero']['00'], 1000, None)
        results_rsa = play_matches(envs, blue_team, BASELINE_TEAMS['ppo-sa-x3']['40'], 1000, None)

        zero_mean_len_wins = results_zero['len_wins'] / results_zero['wins'] if results_zero['wins'] else 0
        zero_mean_len_losses = results_zero['len_losses'] / results_zero['losses'] if results_zero['losses'] else 0

        rsa_mean_len_wins = results_rsa['len_wins'] / results_rsa['wins'] if results_rsa['wins'] else 0
        rsa_mean_len_losses = results_rsa['len_losses'] / results_rsa['losses'] if results_rsa['losses'] else 0

        log = {
            'global_step': int(im.split('.')[0]),
            'RATING-zero': (results_zero['wins'] - results_zero['losses']) / results_zero['matches'],
            'RATING2-zero': (((results_zero['wins'] * (1 - (zero_mean_len_wins/600)) - (results_zero['losses'] * (1 - (zero_mean_len_losses/600)))) / results_zero['matches']) + 1)/2,
            'RATING-rsa': (results_rsa['wins'] - results_rsa['losses']) / results_rsa['matches'],
            'RATING2-rsa': (((results_rsa['wins'] * (1 - (rsa_mean_len_wins/600)) - (results_rsa['losses'] * (1 - (rsa_mean_len_losses/600)))) / results_rsa['matches']) + 1)/2
        }
        wandb.log(log)

    #TODO: remove is_old
    blue_team = get_team(run_name, args.model_path, bt_run.config['hierarchical'], is_old=False)

    # 3 done cases, len by case:
    totals = {
        'matches': 0,
        'match_steps': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'atk_fouls': 0,
        'len_wins': 0,
        'len_losses': 0,
    }

    for team in BASELINE_TEAMS:
        team_totals = {
            'matches': 0,
            'match_steps': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'atk_fouls': 0,
            'len_wins': 0,
            'len_losses': 0,
        }
        for seed in BASELINE_TEAMS[team]:
            print(f"Playing {team} {seed}")
            yellow_team = BASELINE_TEAMS[team][seed]
            results = play_matches(envs, blue_team, yellow_team, 5000, f"{save_path}/val_{team}_{seed}")

            team_totals['matches'] += results['matches']
            team_totals['match_steps'] += results['match_steps']
            team_totals['wins'] += results['wins']
            team_totals['losses'] += results['losses']
            team_totals['draws'] += results['draws']
            team_totals['atk_fouls'] += results['atk_fouls']
            team_totals['len_wins'] += results['len_wins']
            team_totals['len_losses'] += results['len_losses']

            score = (results['wins'] - results['losses']) / results['matches']
            afp = results['atk_fouls'] / results['matches']
            
            run.summary[f"Rating/{team}/{seed}"] = score
            run.summary[f'Atk-Foul/{team}/{seed}'] = afp
            if team == 'zero' or team == 'ou':
                run.log({f"Media/Deterministic/{team}": wandb.Video(f"{save_path}/val_{team}_{seed}/video.000-step-0.mp4")})
            else:
                run.log({f"Media/{team}/{seed}": wandb.Video(f"{save_path}/val_{team}_{seed}/video.000-step-0.mp4")})

        
        # run.summary[f"results/Score/{team}"] = (team_totals['wins'] - team_totals['losses']) / team_totals['matches']
        # run.summary[f"results/Lenght/{team}"] = team_totals['match_steps'] / team_totals['matches']
        # run.summary[f"results/Rate Win/{team}"] = team_totals['wins'] / team_totals['matches']
        # run.summary[f"results/Rate Loss/{team}"] = team_totals['losses'] / team_totals['matches']
        # run.summary[f"results/Rate Draw/{team}"] = team_totals['draws'] / team_totals['matches']
        # run.summary[f'results/Rate Atk Foul/{team}'] = team_totals['atk_fouls'] / team_totals['matches']
        # run.summary[f"results/Lenght Wins/{team}"] = (team_totals['len_wins'] / team_totals['wins']) if team_totals['wins'] else 0
        # run.summary[f"results/Lenght Losses/{team}"] = (team_totals['len_losses'] / team_totals['losses']) if team_totals['losses'] else 0
    
        totals['matches'] += team_totals['matches']
        totals['match_steps'] += team_totals['match_steps']
        totals['wins'] += team_totals['wins']
        totals['losses'] += team_totals['losses']
        totals['draws'] += team_totals['draws']
        totals['atk_fouls'] += team_totals['atk_fouls']
        totals['len_wins'] += team_totals['len_wins']
        totals['len_losses'] += team_totals['len_losses']

    mean_len_wins = totals['len_wins'] / totals['wins'] if totals['wins'] else 0
    mean_len_losses = totals['len_losses'] / totals['losses'] if totals['losses'] else 0

    run.summary["Rating/Score"] = (totals['wins'] - totals['losses']) / totals['matches']
    # run.summary["1-Score 2"] = (((totals['wins'] * (1 - (mean_len_wins/600)) - (totals['losses'] * (1 - (mean_len_losses/600)))) / totals['matches']) + 1)/2
    run.summary["Len/2-Lenght"] = totals['match_steps'] / totals['matches']
    run.summary["Len/3-Lenght Wins"] = mean_len_wins
    run.summary["Len/4-Lenght Losses"] = mean_len_losses
    run.summary["Total/5-Total Wins"] = totals['wins']
    run.summary["Total/6-Total Losses"] = totals['losses']
    run.summary["Total/7-Total Draws"] = totals['draws']
    run.summary['Total/8-Total Atk Fouls'] = totals['atk_fouls']
    run.summary['Total/9-Matches'] = totals['matches']


    # run.summary["results/zz-all/Score"] = (totals['wins'] - totals['losses']) / totals['matches']
    # run.summary["results/zz-all/Lenght"] = totals['match_steps'] / totals['matches']
    # run.summary["results/zz-all/Rate Win"] = totals['wins'] / totals['matches']
    # run.summary["results/zz-all/Rate Loss"] = totals['losses'] / totals['matches']
    # run.summary["results/zz-all/Rate Draw"] = totals['draws'] / totals['matches']
    # run.summary['results/zz-all/Rate Atk Foul'] = totals['atk_fouls'] / totals['matches']
    # run.summary["results/zz-all/Lenght Wins"] = (totals['len_wins'] / totals['wins']) if totals['wins'] else 0
    # run.summary["results/zz-all/Lenght Losses"] = (totals['len_losses'] / totals['losses']) if totals['losses'] else 0
    
    wandb.finish()
