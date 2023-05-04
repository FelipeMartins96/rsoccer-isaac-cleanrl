case $1 in
0) ppo_continuous_action_isaacgym.py --exp-name exp000 -seed 10 --track ;;
1) ppo_continuous_action_isaacgym.py --exp-name exp001 -seed 10 --track --check-angle ;;
2) ppo_continuous_action_isaacgym.py --exp-name exp002 -seed 10 --track --check-speed ;;
3) ppo_continuous_action_isaacgym.py --exp-name exp003 -seed 10 --track --check-angle --check-speed ;;
4) ppo_continuous_action_isaacgym.py --exp-name exp004 -seed 10 --track --terminal-rw ;;
5) ppo_continuous_action_isaacgym.py --exp-name exp005 -seed 10 --track --terminal-rw --check-angle ;;
6) ppo_continuous_action_isaacgym.py --exp-name exp006 -seed 10 --track --terminal-rw --check-speed ;;
7) ppo_continuous_action_isaacgym.py --exp-name exp007 -seed 10 --track --terminal-rw --check-angle --check-speed ;;
*) echo "Opcao Invalida!" ;;
esac