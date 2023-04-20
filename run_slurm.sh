case $1 in
0) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 10 --env-id sa  --track;;
1) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 10 --env-id cma --track;;
2) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 10 --env-id dma --track;;
3) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 20 --env-id sa  --track;;
4) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 20 --env-id cma --track;;
5) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 20 --env-id dma --track;;
6) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 30 --env-id sa  --track;;
7) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 30 --env-id cma --track;;
8) python ppo_continuous_action_isaacgym.py --exp-name exp003 --seed 30 --env-id dma --track;;
*) echo "Opcao Invalida!" ;;
esac