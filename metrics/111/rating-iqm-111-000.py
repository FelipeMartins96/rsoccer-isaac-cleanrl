from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import numpy as np
import seaborn as sns

algorithms = [
    'IL',
    'JAL',
    'RSA',
    'SA',
    ]

colors = sns.color_palette('colorblind')


import pandas as pd

experiments = [(000, 'OLD WEIGHTS'), (111,'NEW WEIGHTS')]
color_idxs = [5,0,3]
COLOR_DICT = dict(zip([n[1] for n in experiments], [colors[idx] for idx in color_idxs]))

aggregate_scores, aggregate_score_cis = {}, {}

score_dict = {}
for alg in algorithms:
    score_dict[alg] = []

# csvs = ['111/data.csv', '111/data2.csv']

df = pd.read_csv('../000/data.csv', header=None)
for i in range(len(df)):
    run = df.iloc[i]

    alg = None
    if run[0] == 'IL-DDPG':
        continue
        alg = 'IL-DDPG'
    if run[0] == 'IL-MADDPG':
        alg = 'IL'
    if run[0] == 'JAL-DDPG':
        alg = 'JAL'
    if run[0] == 'SA-DDPG':
        alg = 'SA'
    if run[0] == 'RSA-DDPG':
        alg = 'RSA'

    score_dict[alg].append([
        run[1],
        run[4],
        run[7],
        run[10],
        run[13],
        run[16],
        run[17]
    ])

    score_dict[alg].append([
        run[2],
        run[5],
        run[8],
        run[11],
        run[14],
        run[16],
        run[17]
    ])

    score_dict[alg].append([
        run[3],
        run[6],
        run[9],
        run[12],
        run[15],
        run[16],
        run[17]
    ])
for alg in algorithms:
    score_dict[alg] = np.array(score_dict[alg])

aggregate_func = lambda x: np.array([
  metrics.aggregate_iqm(x)
  ])
aggregate_scores['OLD WEIGHTS'], aggregate_score_cis['OLD WEIGHTS'] = rly.get_interval_estimates(
  score_dict, aggregate_func, reps=50000)


for exp_id, exp_name in experiments[1:]:

    score_dict = {}
    for alg in algorithms:
        score_dict[alg] = []

    csvs = [f'../{exp_id}/data.csv', f'../{exp_id}/data2.csv']

    for fi in csvs:
        df = pd.read_csv(fi)
        for i in range(len(df)):
            run = df.iloc[i]

            alg = None
            if run.env_id == 'dma':
                alg = 'IL'
            if run.env_id == 'cma':
                alg = 'JAL'
            if run.env_id == 'sa':
                alg = 'SA'
            if run.env_id == 'sa-x3':
                alg = 'RSA'

            score_dict[alg].append([
                run['Rating/ppo-sa-x3/20'],
                run['Rating/zero/00'],
                run['Rating/ou/00'],
                run['Rating/ppo-sa/20'],
                run['Rating/ppo-cma/10'],
                run['Rating/ppo-dma/10']
            ])

            score_dict[alg].append([
                run['Rating/ppo-sa-x3/40'],
                run['Rating/zero/01'],
                run['Rating/ou/01'],
                run['Rating/ppo-sa/40'],
                run['Rating/ppo-cma/30'],
                run['Rating/ppo-dma/20']
            ])

            score_dict[alg].append([
                run['Rating/ppo-sa-x3/50'],
                run['Rating/zero/02'],
                run['Rating/ou/02'],
                run['Rating/ppo-sa/50'],
                run['Rating/ppo-cma/50'],
                run['Rating/ppo-dma/40']
            ])

    for alg in algorithms:
        score_dict[alg] = np.array(score_dict[alg])

    # Load ALE scores as a dictionary mapping algorithms to their human normalized
    # score matrices, each of which is of size `(num_runs x num_games)`.
    aggregate_func = lambda x: np.array([
    #   metrics.aggregate_median(x),
    metrics.aggregate_iqm(x)
    #   metrics.aggregate_mean(x)
    ])
    aggregate_scores[exp_name], aggregate_score_cis[exp_name] = rly.get_interval_estimates(
    score_dict, aggregate_func, reps=50000)
# fig, axes = plot_utils.plot_interval_estimates(
#   aggregate_scores, aggregate_score_cis,
#   metric_names=['MEDIAN', 'IQM'],
#   algorithms=algorithms)

# fig.savefig('fig.png', bbox_inches='tight')
import matplotlib.pyplot as plt
from matplotlib import patches

fig, ax = plt.subplots(figsize=(8,4))
h = 0.6
algorithm_labels = []

for exp_id, exp_name in experiments:
    for i, alg in enumerate(algorithms):
        (l, u) = aggregate_score_cis[exp_name][alg]
        ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT[exp_name], alpha=0.75)
        ax.vlines(x=aggregate_scores[exp_name][alg], ymin=i-7.5 * h/16, ymax=i+(7.5*h/16), color='k', alpha=0.85)
        # ax.text(aggregate_scores[exp_name][alg], i+(h/2), f'{aggregate_scores[exp_name][alg][0]:.3f}', ha='center', va='bottom', size='large')

fake_patches = [patches.Patch(color=COLOR_DICT[e[1]], 
                               alpha=0.75) for e in experiments]
legend = fig.legend(fake_patches, [e[1] for e in experiments], loc='upper center', 
                    fancybox=True, ncol=len(algorithms), 
                    fontsize='x-large',
                    bbox_to_anchor=(0.50, 1.1))

ax.set_yticks(range(len(algorithms)))
ax.set_yticklabels(algorithms)
# ax.set_xlim(-0.5,1)
plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
# ax.set_title('RATING IQM', size='xx-large')
# ax.set_ylabel('PARADIGM', size='xx-large')
ax.set_xlabel('RATING(IQM)', size='xx-large')
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# fig.subplots_adjust(wspace=0.25, hspace=0.45)

fig.tight_layout()
fig.savefig('rating-iqm-000-111.png', bbox_inches='tight')
fig.savefig('rating-iqm-000-111.pdf', bbox_inches='tight')
# ax.xaxis.set_major_locator(MaxNLocator(4))