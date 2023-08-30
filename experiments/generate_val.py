import os

base_path = '../runs'
# iterate over folders inside runs folder
experiments = [exp for exp in os.listdir(base_path) if 'exp' in exp]
for experiment in experiments:
    with open(f'val_{experiment}.sh', 'w') as f:
        f.write('case $1 in\n')
        experiment_path = os.path.join(base_path, experiment)
        case_counter = 0
        for run_name in os.listdir(experiment_path):
            run_name_path = os.path.join(experiment_path, run_name)
            for model in [m for m in os.listdir(run_name_path) if '.pt' in m]:
                model_path = os.path.join('runs', experiment, run_name, model)
                env_id = run_name.split('-')[-1]
                run_id = model.split('.')[0].split('-')[-1]

                f.write(f'{case_counter}) xvfb-run -a python validation.py --env-id {env_id} --run-id {run_id} --model-path {model_path} --atk-foul;;\n')
                case_counter += 1

                if env_id == 'sa':
                    f.write(f'{case_counter}) xvfb-run -a python validation.py --env-id {env_id}-x3 --run-id {run_id} --model-path {model_path} --atk-foul;;\n')
                    case_counter += 1
        f.write('*) echo "Opcao Invalida!" ;;\n')
        f.write('esac\n')