from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import numpy as np
import seaborn as sns

algorithms = [
    'IL-DDPG',
    'IL-MADDPG',
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
color_idxs = [4, 2, 3, 1, 0]
COLOR_DICT = dict(zip(algorithms, [colors[idx] for idx in color_idxs]))


import pandas as pd

score_dict = {}
for alg in algorithms:
    score_dict[alg] = []

# csvs = ['111/data.csv', '111/data2.csv']

df = pd.read_csv('000/data.csv', header=None)
for i in range(len(df)):
    run = df.iloc[i]

    alg = None
    if run[0] == 'IL-DDPG':
        alg = 'IL-DDPG'
    if run[0] == 'IL-MADDPG':
        alg = 'IL-MADDPG'
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
    ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT[alg], alpha=0.75)
    ax.vlines(x=aggregate_scores[alg], ymin=i-7.5 * h/16, ymax=i+(7.5*h/16), color='k', alpha=0.85)
    ax.text(aggregate_scores[alg], i+(h/2), f'{aggregate_scores[alg][0]:.3f}', ha='center', va='bottom', size='large')

ax.set_yticks(range(len(algorithms)))
ax.set_yticklabels(algorithms)
# ax.set_xlim(-0.5,1)
plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
# ax.set_title('RATING IQM', size='xx-large')
# ax.set_ylabel('PARADIGM', size='xx-large')
ax.set_xlabel('Rating (IQM)', size='xx-large')
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# fig.subplots_adjust(wspace=0.25, hspace=0.45)

fig.tight_layout()
fig.savefig('rating-iqm-rsoccer.pdf', bbox_inches='tight')
fig.savefig('rating-iqm-rsoccer.png', bbox_inches='tight')
# ax.xaxis.set_major_locator(MaxNLocator(4))