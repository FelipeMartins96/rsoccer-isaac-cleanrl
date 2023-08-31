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
OUTCOMES = ['GS', 'BM', 'RM', 'RE']
color_idxs = [3, 1, 4, 0]
COLOR_DICT = dict(zip(OUTCOMES, [colors[idx] for idx in color_idxs]))

algorithms = [
    'Previous\nMean',
    'IL',
    # 'IL-MADDPG',
    'JAL',
    # 'RSA',
    'SA',
    ]

prev_dict = {
    'JAL': {
        'GS': 9.895627705627706,
        'BM': 46.99431596148108,
        'RM': 27.206296743008483,
        'RE': 2.0461271716078837,
    },
    'IL-DDPG': {
        'GS': 5.169928959928966,
        'BM': 21.066964387748147,
        'RM': 37.668747376288245,
        'RE': 8.763220941365681,
    },
    'IL-MADDPG': {
        'GS': 7.030517075517073,
        'BM': 32.994164375069026,
        'RM': 41.479216544168736,
        'RE': 7.055969270018965,
    },
    'SA': {
        'GS': 9.93994227994228,
        'BM': 47.885367175826644,
        'RM': 42.45611826348676,
        'RE': 2.04815179794282,
    },
}

scores_dict = {
    'Previous\nMean': {
        'GS':(prev_dict['JAL']['GS'] + prev_dict['IL-MADDPG']['GS'] + prev_dict['IL-DDPG']['GS'] + prev_dict['SA']['GS'])/4,
        'BM':(prev_dict['JAL']['BM'] + prev_dict['IL-MADDPG']['BM'] + prev_dict['IL-DDPG']['BM'] + prev_dict['SA']['BM'])/4,
        'RM':(prev_dict['JAL']['RM'] + prev_dict['IL-MADDPG']['RM'] + prev_dict['IL-DDPG']['RM'] + prev_dict['SA']['RM'])/4,
        'RE':(prev_dict['JAL']['RE'] + prev_dict['IL-MADDPG']['RE'] + prev_dict['IL-DDPG']['RE'] + prev_dict['SA']['RE'])/4,
    },
    'JAL': {
        'GS':9.72663492063492,
        'BM':2.6678080559098514,
        'RM':1.2567009498483355,
        'RE':0.27335481804952266,
    },
    'IL': {
        'GS':9.616222222222222,
        'BM':2.68733164069842,
        'RM':1.5192956631276942,
        'RE':0.20409760636460214,
    },
    'SA': {
        'GS':9.059396825396824,
        'BM':2.755729823060264,
        'RM':1.6134715028708644,
        'RE':0.2463373139175572,
    },
}

fig, ax = plt.subplots(figsize=(10, 4))
h = 0.8

for i, alg in enumerate(algorithms):
    total_rwds = scores_dict[alg]['GS'] + scores_dict[alg]['BM'] + scores_dict[alg]['RM'] + scores_dict[alg]['RE']
    gs_rate = scores_dict[alg]['GS'] / total_rwds
    bm_rate = scores_dict[alg]['BM'] / total_rwds
    rm_rate = scores_dict[alg]['RM'] / total_rwds
    re_rate = scores_dict[alg]['RE'] / total_rwds

    l, u = 0, gs_rate
    ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT['GS'], alpha=0.75)
    ax.text((l+u)/2, i, f'{int(gs_rate*100)}%', ha='center', va='center', size='large')
    l, u = u, u+bm_rate
    ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT['BM'], alpha=0.75)
    ax.text((l+u)/2, i, f'{int(bm_rate*100)}%', ha='center', va='center', size='large')
    l, u = u, u+rm_rate
    ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT['RM'], alpha=0.75)
    ax.text((l+u)/2, i, f'{int(rm_rate*100)}%', ha='center', va='center', size='large')
    l, u = u, u+re_rate
    ax.barh(y=i, width=u-l, height=h, left=l, color=COLOR_DICT['RE'], alpha=0.75)
    if re_rate>0.03:
        ax.text((l+u)/2, i, f'{int(re_rate*100)}%', ha='right' if re_rate<0.03 else 'center', va='center', size='large')
    
    # ax.vlines(x=aggregate_scores[alg], ymin=i-7.5 * h/16, ymax=i+(7.5*h/16), color='k', alpha=0.85)

fake_patches = [patches.Patch(color=COLOR_DICT[o], 
                               alpha=0.75) for o in OUTCOMES]
legend = fig.legend(fake_patches, OUTCOMES, loc='upper center', 
                    fancybox=True, ncol=len(OUTCOMES), 
                    fontsize='x-large',
                    bbox_to_anchor=(0.55, 1.1))

ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks(range(len(algorithms)))
ax.set_yticklabels(algorithms)

# ax.set_xlim(-0.5,1)
plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
# ax.set_title('RATING IQM', size='xx-large')
# ax.set_ylabel('PARADIGM', size='xx-large')
ax.set_xlabel('Reward Component Ratio From Episode Total', size='xx-large')
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# fig.subplots_adjust(wspace=0.25, hspace=0.45)
# ax.tick_params(labelleft = False)
fig.tight_layout()
fig.savefig('rwds-percents.pdf', bbox_inches='tight')
fig.savefig('rwds-percents.png', bbox_inches='tight')
# ax.xaxis.set_major_locator(MaxNLocator(4))