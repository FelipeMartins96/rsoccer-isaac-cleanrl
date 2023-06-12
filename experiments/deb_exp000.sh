EXP_NAME="exp000"
EXP_NOTES="debug"
EXP_ARGS=""
case $1 in
0) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name 256 --wandb-notes="$EXP_NOTES" --num-envs 256 --env-id sa  --seed 10 --track --capture-video;;
1) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name 250 --wandb-notes="$EXP_NOTES" --env-id sa --seed 10 --track --capture-video;;
*) echo "Opcao Invalida!" ;;
esac