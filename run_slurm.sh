case $1 in
0) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id sa --seed 10 --track --capture-video;;
1) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id sa --seed 20 --track --capture-video;;
2) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id sa --seed 30 --track --capture-video;;
3) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id cma --seed 10 --track --capture-video;;
4) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id cma --seed 20 --track --capture-video;;
5) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id cma --seed 30 --track --capture-video;;
6) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id dma --seed 10 --track --capture-video;;
7) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id dma --seed 20 --track --capture-video;;
8) python ppo_continuous_action_isaacgym.py --exp-name exp002 --env-id dma --seed 30 --track --capture-video;;
*) echo "Opcao Invalida!" ;;
esac