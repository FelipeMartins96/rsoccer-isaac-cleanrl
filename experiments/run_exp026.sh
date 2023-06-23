EXP_NAME="exp026"
EXP_NOTES="5 epochs"
EXP_ARGS="--update-epochs 5"
case $1 in
0) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id sa  --seed 10 --track --capture-video;;
1) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id cma --seed 10 --track --capture-video;;
2) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id dma --seed 10 --track --capture-video;;
3) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id sa  --seed 20 --track --capture-video;;
4) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id cma --seed 20 --track --capture-video;;
5) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id dma --seed 20 --track --capture-video;;
6) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id sa  --seed 30 --track --capture-video;;
7) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id cma --seed 30 --track --capture-video;;
8) xvfb-run -a python ppo_continuous_action_isaacgym.py --exp-name $EXP_NAME --wandb-notes="$EXP_NOTES" $EXP_ARGS --env-id dma --seed 30 --track --capture-video;;
*) echo "Opcao Invalida!" ;;
esac