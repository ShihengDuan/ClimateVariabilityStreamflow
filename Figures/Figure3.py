import pickle
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os

feature_to_plot = 11
fontsize = 12 
with open('r2s_include_eof_smooth.p', 'rb') as pfile:
        r2s_include = pickle.load(pfile)
r2s_linear, r2s_lasso, r2s_ridge, r2s_lod, r2s_ml = r2s_include

# Sort according to r2_linear:
sorted_items = sorted(r2s_linear.items(), key=lambda x: np.median(x[1]), reverse=True)
result_keys = [key for key, value in sorted_items]
print(result_keys)
labels = result_keys.copy()
labels.remove('full')
key_labels = []
for p in labels:
    if 'eof' in p:
        key_labels.append(p[:3]+'-'+p[-1])
    else:
        key_labels.append(p)
print(key_labels)

modes_colors = {'PNA_eof_5':'tab:brown', 'PDO_eof_5':'tab:gray', 
                'PDO_eof_3':'tab:orange', 'NAM_eof_2':'tab:red',
                'NAO_eof_5':'tab:pink'}

def plot_ax(ax, r2s, model):
    full_r2 = np.median(r2s['full'])
    for i, key in enumerate(result_keys[:feature_to_plot+1]):
        if not key =='full':
            color=modes_colors[key] if key in modes_colors else 'tab:blue'
            ax.bar(i, np.median(r2s[key])/full_r2, color=color)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8])
    base_r2 = 0
    ax2 = ax.twinx()
    for l in range(5):
        file = '/p/lustre2/shiduan/'+model.upper()+'-median-results/level-'+str(l)+'.p'
        if os.path.exists(file):
            with open('/p/lustre2/shiduan/'+model.upper()+'-median-results/level-'+str(l)+'.p', 'rb') as pfile:
                results = pickle.load(pfile)
            mode_max = results['max_mode']
            print(mode_max)
            r2 = results[mode_max]
            # print(r2/full_r2)
            color=modes_colors[mode_max] if mode_max in modes_colors else 'tab:blue'
            ax2.bar(feature_to_plot+1, (r2-base_r2)/full_r2, bottom=base_r2/full_r2, color=color)
            base_r2 = r2
    y_max = np.ceil(base_r2/full_r2 * 2) / 2
    ax2.set_ylim([0, y_max])

fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(8, 8))
ax = axes.flatten()[0]
ax.set_title('LR', fontsize=fontsize)
plot_ax(ax, r2s=r2s_linear, model='LR')

ax = axes.flatten()[1]
ax.set_title('Ridge', fontsize=fontsize)
plot_ax(ax, r2s=r2s_ridge, model='Ridge')

ax = axes.flatten()[2]
ax.set_title('Lasso', fontsize=fontsize)
plot_ax(ax, r2s=r2s_lasso, model='Lasso')

ax = axes.flatten()[3]
ax.set_title('LOD', fontsize=fontsize)
plot_ax(ax, r2s=r2s_lod, model='LOD')

ax = axes.flatten()[4]
ax.set_title('AutoML', fontsize=fontsize)
plot_ax(ax, r2s=r2s_ml, model='AutoML')


ticks = key_labels[:feature_to_plot]
ticks.append('Cumul')
for i in range(5):
    ax = axes.flatten()[i]
    # ax.set_xticks(np.arange(38))
    ax.set_xticks(np.arange(1, feature_to_plot+1+1), ticks, 
                  rotation=45, fontsize=fontsize, 
                  ha='right', rotation_mode='anchor')
    # ax.set_xticklabels(names, rotation=90)
plt.tight_layout()
plt.savefig('Cal-median-result-includeFigure3.png', dpi=150, bbox_inches='tight')
print('Done')

