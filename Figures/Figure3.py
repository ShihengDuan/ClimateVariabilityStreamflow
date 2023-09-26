import pickle
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import cartopy
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import pandas as pd
from matplotlib import patches
station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]

lats = []
lons = []
camel_topo = pd.read_csv(
    '/usr/workspace/shiduan/neuralhydrology/data/camels_us/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_topo.txt', delimiter=';')
for station in station_ids:
    record = camel_topo[camel_topo['gauge_id']==int(station)]
    lats.append(record['gauge_lat'])
    lons.append(record['gauge_lon'])



fig = plt.figure(figsize=(11, 8))
# Grid Design
gs = GridSpec(6, 3, figure=fig)

feature_to_plot = 11
fontsize = 12 
with open('r2s_include_eof_smooth.p', 'rb') as pfile:
        r2s_include = pickle.load(pfile)
r2s_linear, r2s_lasso, r2s_ridge, r2s_lod, r2s_ml, r2s_pls = r2s_include

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
                'NAO_eof_5':'tab:pink', 
                'PNA_eof_6':'tab:green', 'PNA_eof_3':'tab:olive', 'PDO_eof_4':'tab:purple'}

def plot_ax(ax, r2s, model):
    full_r2 = np.median(r2s['full'])
    for i, key in enumerate(result_keys[:feature_to_plot+1]):
        if not key =='full':
            color=modes_colors[key] if key in modes_colors else 'tab:blue'
            ax.bar(i, np.median(r2s[key])/full_r2, color=color)
    if model=='LOD':
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_yticklabels([0, 0.2, 0.4, 0.6])
    elif model=='AutoML':
        ax.set_yticks([0, 0.2, 0.4])
        ax.set_yticklabels([0, 0.2, 0.4])
    else:
        ax.set_yticks([0, 0.4, 0.8])
        ax.set_yticklabels([0, 0.4, 0.8])
    base_r2 = 0
    ax2 = ax.twinx()
    for l in range(5):
        file = '/p/lustre2/shiduan/'+model.upper()+'-median-results/level-'+str(l)+'.p'
        if os.path.exists(file):
            with open('/p/lustre2/shiduan/'+model.upper()+'-median-results/level-'+str(l)+'.p', 'rb') as pfile:
                results = pickle.load(pfile)
            mode_max = results['max_mode']
            # print(mode_max)
            r2 = results[mode_max]
            print(r2/full_r2)
            color=modes_colors[mode_max] if mode_max in modes_colors else 'tab:blue'
            ax2.bar(feature_to_plot+1, (r2-base_r2)/full_r2, bottom=base_r2/full_r2, color=color)
            base_r2 = r2
    y_max = np.ceil(base_r2/full_r2 * 2) / 2
    ax2.set_ylim([0, y_max])
    print(full_r2, base_r2, model)
# fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True, sharey=False, figsize=(8, 8))
# ax = axes.flatten()[0]
axes = []
ax1 = fig.add_subplot(gs[5, :2])
ax1.set_title('AutoML', fontsize=fontsize)
plot_ax(ax1, r2s=r2s_ml, model='AutoML')
ticks = key_labels[:feature_to_plot]
ticks.append('Cumul')
ax1.set_xticks(np.arange(1, feature_to_plot+1+1), ticks, 
                  rotation=45, fontsize=fontsize, 
                  ha='right', rotation_mode='anchor')
axes.append(ax1)

ax = fig.add_subplot(gs[0, :2], sharex=ax1)
ax.set_title('LR', fontsize=fontsize)
plot_ax(ax, r2s=r2s_linear, model='LR')
axes.append(ax)
ax.annotate('A)', xy=(-0.08, 1.15), xycoords='axes fraction', 
            fontsize=14, weight='bold')

ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
ax2.set_title('Ridge', fontsize=fontsize)
plot_ax(ax2, r2s=r2s_ridge, model='Ridge')
axes.append(ax2)

ax3 = fig.add_subplot(gs[2, :2], sharex=ax1)
ax3.set_title('Lasso', fontsize=fontsize)
plot_ax(ax3, r2s=r2s_lasso, model='Lasso')
axes.append(ax3)

ax = fig.add_subplot(gs[3, :2], sharex=ax1)
ax.set_title('PLS', fontsize=fontsize)
plot_ax(ax, r2s=r2s_pls, model='PLS')
axes.append(ax)

ax = fig.add_subplot(gs[4, :2], sharex=ax1)
ax.set_title('LOD', fontsize=fontsize)
plot_ax(ax, r2s=r2s_lod, model='LOD')
axes.append(ax)

for i in range(1, 6):
    plt.setp(axes[i].get_xticklabels(), visible=False)

# Spatial map
results = pd.read_csv('~/MyWorkSpace/hydro/include/resultsPass/results.csv')

ax = fig.add_subplot(gs[:3, 2:3], projection=cartopy.crs.PlateCarree())
labels = []
for j in range(len(station_peaks)):
    station_slice = results[results['station']==int(station_ids[j])]
    f1 = station_slice[['model', 'F1']]
    f1_unique = f1['F1'].unique()
    if len(f1_unique)==1: # all model agree
        color=modes_colors[f1_unique[0]]
        labels.append(f1_unique[0])
    else:
        color='tab:blue'
    peak = station_peaks[j]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    ax.scatter(lons[j], lats[j],  color=color,
            transform=cartopy.crs.PlateCarree(), marker=marker)
ax.add_feature(cartopy.feature.STATES)
ax.set_title('Rank 1', fontsize=fontsize)
labels = np.unique(labels)
handles = []
label_name = []
for label in labels:
    handles.append(patches.Rectangle((0,0), .8, 1, facecolor=modes_colors[label]))
    label_name.append(label[:3]+'-'+label[-1])
ax.legend(handles, label_name)
ax.annotate('B)', xy=(-0.12, 1.03), xycoords='axes fraction', 
            fontsize=14, weight='bold')

ax = fig.add_subplot(gs[3:, 2:3], projection=cartopy.crs.PlateCarree())
labels = []
for j in range(len(station_peaks)):
    peak = station_peaks[j]
    station_slice = results[results['station']==int(station_ids[j])]
    f2 = station_slice[['model', 'F2']]
    f2_unique = f2['F2'].unique()
    if len(f2_unique)==1: # all model agree
        color=modes_colors[f2_unique[0]]
        labels.append(f2_unique[0])
        if peak>3 and peak<12:
            marker='^'
        else:
            marker='o'
        ax.scatter(lons[j], lats[j], color=color,
                transform=cartopy.crs.PlateCarree(), marker=marker)
    else:
        pass
    
ax.add_feature(cartopy.feature.STATES)
ax.set_title('Rank 2', fontsize=fontsize)
labels = np.unique(labels)
handles = []
label_name = []
for label in labels:
    handles.append(patches.Rectangle((0,0), .8, 1, facecolor=modes_colors[label]))
    label_name.append(label[:3]+'-'+label[-1])
ax.legend(handles, label_name)


plt.tight_layout()
plt.savefig('Cal-median-result-includeFigure3.png', dpi=150, bbox_inches='tight')
print('Done')

