EXP_NAME="exp012"
EXP_NOTES="vf-coef 2"
EXP_ARGS="--vf-coef 2"
case $1 in
0) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id dma --seed 10 --track --capture-video;;
1) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id dma --seed 20 --track --capture-video;;
2) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id dma --seed 30 --track --capture-video;;
*) echo "Opcao Invalida!" ;;
esac