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

RED    =    (201/255,  33/255,  30/255)
ORANGE =    (229/255, 116/255,  57/255)
PURPLE =    (125/255,  84/255, 178/255)
GREEN  =    ( 76/255, 147/255, 103/255)
BLUE   =    ( 83/255, 135/255, 221/255)

colors = [RED, ORANGE, PURPLE, GREEN, BLUE]
color_idxs = [2,3,1,0]
ATARI_100K_COLOR_DICT = dict(zip(algorithms, [colors[idx] for idx in color_idxs]))


import pandas as pd

score_dict = {}
for alg in algorithms:
    score_dict[alg] = []

csvs = ['121/data.csv', '121/data2.csv']

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
        
        # score_dict[alg].append([
        #     run['Rating/ppo-sa-x3/20'],
        #     run['Rating/ppo-sa-x3/40'],
        #     run['Rating/ppo-sa-x3/50'],
        #     run['Rating/zero/00'],
        #     # run['Rating/zero/01'],
        #     # run['Rating/zero/02'],
        #     run['Rating/ou/00'],
        #     # run['Rating/ou/01'],
        #     # run['Rating/ou/02'],
        #     run['Rating/ppo-sa/20'],
        #     run['Rating/ppo-sa/40'],
        #     run['Rating/ppo-sa/50'],
        #     run['Rating/ppo-cma/10'],
        #     run['Rating/ppo-cma/30'],
        #     run['Rating/ppo-cma/50'],
        #     run['Rating/ppo-dma/10'],
        #     run['Rating/ppo-dma/20'],
        #     run['Rating/ppo-dma/40']
        # ])

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
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
  score_dict, aggregate_func, reps=50000)
# fig, axes = plot_utils.plot_interval_estimates(
#   aggregate_scores, aggregate_score_cis,
#   metric_names=['MEDIAN', 'IQM'],
#   algorithms=algorithms)

# fig.savefig('fig.png', bbox_inches='tight')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
h = 0.6
algorithm_labels = []

for i, alg in enumerate(algorithms):
    (l, u) = aggregate_score_cis[alg]
    ax.barh(y=i, width=u-l, height=h, left=l, color=ATARI_100K_COLOR_DICT[alg], alpha=0.75)
    ax.vlines(x=aggregate_scores[alg], ymin=i-7.5 * h/16, ymax=i+(7.5*h/16), color='k', alpha=0.85)
    ax.text(aggregate_scores[alg], i+(h/2), f'{aggregate_scores[alg][0]:.3f}', ha='center', va='bottom', size='large')

ax.set_yticks(range(len(algorithms)))
ax.set_yticklabels(algorithms)
# ax.set_xlim(-0.5,1)
plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
# ax.set_title('RATING IQM', size='xx-large')
# ax.set_ylabel('PARADIGM', size='xx-large')
ax.set_xlabel('RATING (IQM)', size='xx-large')
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# fig.subplots_adjust(wspace=0.25, hspace=0.45)

fig.tight_layout()
fig.savefig('rating-iqm.png', bbox_inches='tight')
fig.savefig('rating-iqm.pdf', bbox_inches='tight')
# ax.xaxis.set_major_locator(MaxNLocator(4))