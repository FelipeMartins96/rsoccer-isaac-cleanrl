case $1 in
0) python test.py --algo sa --seed 20 --net-path base_nets/exp000_ppo-sa_20/agent.pt --run-id 1qt6s043;;
1) python test.py --algo sa --seed 30 --net-path base_nets/exp000_ppo-sa_30/agent.pt --run-id 3b2ab74c;;
2) python test.py --algo cma --seed 10 --net-path base_nets/exp000_ppo-cma_10/agent.pt --run-id 31vaen3f;;
3) python test.py --algo cma --seed 20 --net-path base_nets/exp000_ppo-cma_20/agent.pt --run-id ofadaqvy;;
4) python test.py --algo cma --seed 30 --net-path base_nets/exp000_ppo-cma_30/agent.pt --run-id 3hvw26wk;;
5) python test.py --algo dma --seed 10 --net-path base_nets/exp000_ppo-dma_10/agent.pt --run-id 4rqx0yhf;;
6) python test.py --algo dma --seed 20 --net-path base_nets/exp000_ppo-dma_20/agent.pt --run-id ajxulk8j;;
7) python test.py --algo dma --seed 30 --net-path base_nets/exp000_ppo-dma_30/agent.pt --run-id 2qk7rzlz;;
*) echo "Opcao Invalida!" ;;
esac