# fig.savefig('fig.png', bbox_inches='tight')
import matplotlib.pyplot as plt
import seaborn as sns
from rliable import plot_utils
from matplotlib import ticker, patches

RED    =    (201/255,  33/255,  30/255)
ORANGE =    (229/255, 116/255,  57/255)
PURPLE =    (125/255,  84/255, 178/255)
GREEN  =    ( 76/255, 147/255, 103/255)
BLUE   =    ( 83/255, 135/255, 221/255)
colors = [RED, ORANGE, PURPLE, GREEN, BLUE]
OUTCOMES = ['WIN', 'DRAW', 'LOSS']
color_idxs = [3, 4, 0]
COLOR_DICT = dict(zip(OUTCOMES, [colors[idx] for idx in color_idxs]))

algorithms = [
    'IL-DDPG',
    'IL-MADDPG',
    'JAL',
    'RSA',
    'SA',
    ]

scores_dict = {
    'JAL': {
        'WINS': 19433,
        'DRAWS': 4397,
        'LOSSES': 7670,
    },
    'IL-DDPG': {
        'WINS': 5953,
        'DRAWS': 13129,
        'LOSSES': 12418,
    },
    'IL-MADDPG': {
        'WINS': 10179,
        'DRAWS': 10367,
        'LOSSES': 10954,
    },
    'RSA': {
        'WINS': 23080,
        'DRAWS': 5837,
        'LOSSES': 2583,
    },
    'SA': {
        'WINS': 18691,
        'DRAWS': 3390,
        'LOSSES': 9419,
    }
}

fig, ax = plt.subplots()
h = 0.8

for i, alg in enumerate(algorithms):
    total_matches = scores_dict[alg]['WINS'] + scores_dict[alg]['DRAWS'] + scores_dict[alg]['LOSSES']
    win_rate = scores_dict[alg]['WINS'] / total_matches
    draw_rate = scores_dict[alg]['DRAWS'] / total_matches
    loss_rate = scores_dict[alg]['LOSSES'] / total_matches
    l, u = 0, win_rate
    ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT['WIN'], alpha=0.75)
    ax.text((l+u)/2, i, f'{int(win_rate*100)}%', ha='center', va='center', size='large')
    l, u = u, u+draw_rate
    ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT['DRAW'], alpha=0.75)
    ax.text((l+u)/2, i, f'{int(draw_rate*100)}%', ha='center', va='center', size='large')
    l, u = u, u+loss_rate
    ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT['LOSS'], alpha=0.75)
    ax.text((l+u)/2, i, f'{int(loss_rate*100)}%', ha='center', va='center', size='large')
    # ax.vlines(x=aggregate_scores[alg], ymin=i-7.5 * h/16, ymax=i+(7.5*h/16), color='k', alpha=0.85)

fake_patches = [patches.Patch(color=COLOR_DICT[o], 
                               alpha=0.75) for o in OUTCOMES]
legend = fig.legend(fake_patches, OUTCOMES, loc='upper center', 
                    fancybox=True, ncol=len(algorithms), 
                    fontsize='x-large',
                    bbox_to_anchor=(0.45, 1.1))

ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks(range(len(algorithms)))
ax.set_yticklabels(algorithms)

# ax.set_xlim(-0.5,1)
plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
# ax.set_title('RATING IQM', size='xx-large')
# ax.set_ylabel('PARADIGM', size='xx-large')
ax.set_xlabel('Outcomes', size='xx-large')
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# fig.subplots_adjust(wspace=0.25, hspace=0.45)
ax.tick_params(labelleft = False)
fig.tight_layout()
fig.savefig('outcomes-rsoccer.pdf', bbox_inches='tight')
fig.savefig('outcomes-rsoccer.png', bbox_inches='tight')
# ax.xaxis.set_major_locator(MaxNLocator(4))