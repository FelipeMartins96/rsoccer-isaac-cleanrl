# TODO MOVE TO VALIDATION SCRIPT
    # if args.track and args.env_id != 'goto':
    #     with torch.no_grad():
    #         # envs.close()
    #         writer.close()
    #         wandb_id = wandb.run.id
    #         wandb.finish()
    #         print("Training finished")
    #         print("Running evaluation")
    #         unwrapped_env.w_goal = 1.0
    #         unwrapped_env.w_grad = 0.0
    #         unwrapped_env.w_move = 0.0
    #         unwrapped_env.w_energy = 0.0
    #         from play import play_matches, get_team, BASELINE_TEAMS
    #         blue_team = get_team(f'ppo-{args.env_id}', f"{save_path}/{run_name}/{run_name}-agent.pt") 
    #         run = wandb.init(
    #             project=args.wandb_project_name,
    #             monitor_gym=False,
    #             id=wandb_id,
    #             resume=True,
    #         )
        
    #         total_score = 0
    #         total_len = 0
    #         total_count = 0
        
    #         for team in BASELINE_TEAMS:
    #             team_score = 0
    #             team_len = 0
    #             for seed in BASELINE_TEAMS[team]:
    #                 print(f"Playing {team} {seed}")
    #                 yellow_team = BASELINE_TEAMS[team][seed]
    #                 rews, lens = play_matches(unwrapped_env, blue_team, yellow_team, 10000, f"{save_path}/{run_name}/val_{team}_{seed}" if args.capture_video else None)
    #                 team_score += rews
    #                 team_len += lens
    #                 total_score += rews
    #                 total_len += lens
    #                 total_count += 1
    #                 if args.capture_video:
    #                     run.log({f"Validation/Media/{team}/{seed}": wandb.Video(f"{save_path}/{run_name}/val_{team}_{seed}/video.000-step-0.mp4")})
    #             run.summary[f"Validation/Score/{team}"] = team_score / len(BASELINE_TEAMS[team])
    #             run.summary[f"Validation/Length/{team}"] = team_len / len(BASELINE_TEAMS[team])
    #         run.summary["Validation/Score Mean"] = total_score / total_count
    #         run.summary["Validation/Length Mean"] = total_len / total_count
        
    #         wandb.finish()
    #         if args.env_id == "sa":
    #             print("Running evaluation for sa-x3")
        
    #             blue_team = get_team('ppo-sa-x3', f"{save_path}/{run_name}/{run_name}-agent.pt")
    #             run_name = f"{args.exp_name}_ppo-{args.env_id}-x3_{args.seed}"
    #             new_config = vars(args)
    #             new_config["env_id"] = "sa-x3"
    #             run = wandb.init(
    #                 project=args.wandb_project_name,
    #                 monitor_gym=False,
    #                 name=run_name,
    #                 group=args.exp_name,
    #                 config=new_config,
    #             )
    #             total_score = 0
    #             total_len = 0
    #             total_count = 0
        
    #             for team in BASELINE_TEAMS:
    #                 team_score = 0
    #                 team_len = 0
    #                 for seed in BASELINE_TEAMS[team]:
    #                     print(f"Playing {team} {seed}")
    #                     yellow_team = BASELINE_TEAMS[team][seed]
    #                     rews, lens = play_matches(unwrapped_env, blue_team, yellow_team, 10000, f"{save_path}/{run_name}/val_{team}_{seed}" if args.capture_video else None)
    #                     team_score += rews
    #                     team_len += lens
    #                     total_score += rews
    #                     total_len += lens
    #                     total_count += 1
    #                     if args.capture_video:
    #                         run.log({f"Validation/Media/{team}/{seed}": wandb.Video(f"{save_path}/{run_name}/val_{team}_{seed}/video.000-step-0.mp4")})
    #                 run.summary[f"Validation/Score/{team}"] = team_score / len(BASELINE_TEAMS[team])
    #                 run.summary[f"Validation/Length/{team}"] = team_len / len(BASELINE_TEAMS[team])
    #             run.summary["Validation/Score Mean"] = total_score / total_count
    #             run.summary["Validation/Length Mean"] = total_len / total_count