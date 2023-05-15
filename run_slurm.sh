case $1 in
0) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 10 --env-id sa  --track --hierarchical;;
1) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 10 --env-id cma --track --hierarchical;;
2) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 10 --env-id dma --track --hierarchical;;
3) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 20 --env-id sa  --track --hierarchical;;
4) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 20 --env-id cma --track --hierarchical;;
5) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 20 --env-id dma --track --hierarchical;;
6) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 30 --env-id sa  --track --hierarchical;;
7) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 30 --env-id cma --track --hierarchical;;
8) python ppo_continuous_action_isaacgym.py --exp-name exp006 --seed 30 --env-id dma --track --hierarchical;;
*) echo "Opcao Invalida!" ;;
esac