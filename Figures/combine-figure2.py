import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.metrics import r2_score
import os
import pandas as pd
import cartopy
import matplotlib as mpl
import pickle
from matplotlib.gridspec import GridSpec

fontsize=14
subsize=12
station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]

# Load r2
scores_LOD_lag3 = np.load('../dataResult/scores_LOD_smooth_lag3.npy')
scores_LA_lag3 = np.load('../dataResult/scores_LA_smooth_lag3.npy')
scores_RD_lag3 = np.load('../dataResult/scores_RD_smooth_lag3.npy')
scores_LR_lag3  = np.load('../dataResult/scores_LR_smooth_lag3.npy')
scores_ML_lag3 = np.load('../dataResult/scores_AutoLR_smooth_lag3.npy')
scores_PLS_lag3 = np.load('../dataResult/scores_PLS_smooth_lag3.npy')

scores_LOD = np.load('../dataResult/scores_LOD_smooth.npy')
scores_LA = np.load('../dataResult/scores_LA_smooth.npy')
scores_RD = np.load('../dataResult/scores_RD_smooth.npy')
scores_LR  = np.load('../dataResult/scores_LR_smooth.npy')
scores_ML = np.load('../dataResult/scores_AutoLR_smooth.npy')
scores_PLS = np.load('../dataResult/scores_PLS_smooth.npy')


lats = []
lons = []
camel_topo = pd.read_csv(
    '/usr/workspace/shiduan/neuralhydrology/data/camels_us/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_topo.txt', delimiter=';')
for station in station_ids:
    record = camel_topo[camel_topo['gauge_id']==int(station)]
    lats.append(record['gauge_lat'])
    lons.append(record['gauge_lon'])

norm = colors.Normalize(vmin=0, vmax=0.6)
norm_re = colors.Normalize(vmin=0, vmax=0.6)

fig = plt.figure(figsize=(11, 15))
# Grid Design
gs = GridSpec(8, 7, figure=fig)
for i in range(6):
    ind = i
    # ax = fig.add_subplot(5, 7, ind+1, projection=cartopy.crs.PlateCarree())
    ax = fig.add_subplot(gs[0, ind], projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
            score = scores_LR_lag3[j, i]
        else:
            marker='o'
            score = scores_LR[j, i]
        ax.scatter(lons[j], lats[j], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('LR-EOF-'+str(i+1), fontsize=subsize)
for i in range(6):
    ind = i
    # ax = fig.add_subplot(5, 7, ind+7+1, projection=cartopy.crs.PlateCarree())
    ax = fig.add_subplot(gs[2, ind], projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
            score = scores_RD_lag3[j, i]
        else:
            marker='o'
            score = scores_RD[j, i]
        ax.scatter(lons[j], lats[j], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('Ridge-EOF-'+str(i+1), fontsize=subsize)
for i in range(6):
    ind = i
    # ax = fig.add_subplot(5, 7, ind+14+1, projection=cartopy.crs.PlateCarree())
    ax = fig.add_subplot(gs[1, ind], projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
            score = scores_LA_lag3[j, i]
        else:
            marker='o'
            score = scores_LA[j, i]
        ax.scatter(lons[j], lats[j], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('Lasso-EOF-'+str(i+1), fontsize=subsize)
for i in range(6):
    ind = i
    # ax = fig.add_subplot(5, 7, ind+21+1, projection=cartopy.crs.PlateCarree())
    ax = fig.add_subplot(gs[4, ind], projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
            score = scores_LOD_lag3[j, i]
        else:
            marker='o'
            score = scores_LOD[j, i]
        ax.scatter(lons[j], lats[j], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('LOD-EOF-'+str(i+1), fontsize=subsize)

for i in range(6):
    ind = i
    # ax = fig.add_subplot(5, 7, ind+21+1, projection=cartopy.crs.PlateCarree())
    ax = fig.add_subplot(gs[3, ind], projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
            score = scores_PLS_lag3[j, i]
        else:
            marker='o'
            score = scores_PLS[j, i]
        ax.scatter(lons[j], lats[j], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('PLS-EOF-'+str(i+1), fontsize=subsize)

for i in range(6):
    ind = i
    # ax = fig.add_subplot(5, 7, ind+28+1, projection=cartopy.crs.PlateCarree())
    ax = fig.add_subplot(gs[5, ind], projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
            score = scores_ML_lag3[j, i]
        else:
            marker='o'
            score = scores_ML[j, i]
        ax.scatter(lons[j], lats[j], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('AutoML-EOF-'+str(i+1), fontsize=subsize)

# reanalysis
with open('../dataResult/Reanalysis/results.p', 'rb') as pfile:
    results = pickle.load(pfile)

# ax_lr = fig.add_subplot(5, 7, 1, projection=cartopy.crs.PlateCarree())
ax_lr = fig.add_subplot(gs[0, 6], projection=cartopy.crs.PlateCarree())

for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['LR']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm_re, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('LR-Obs', fontsize=subsize)

ax_lr = fig.add_subplot(gs[1, 6], projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['Lasso']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm_re, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('Lasso-Obs', fontsize=subsize)

ax_lr = fig.add_subplot(gs[2, 6], projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['Ridge']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm_re, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('Ridge-Obs', fontsize=subsize)


ax_lr = fig.add_subplot(gs[3, 6], projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['PLS']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm_re, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('PLS-Obs', fontsize=subsize)

# ax_lr = fig.add_subplot(5, 7, 22, projection=cartopy.crs.PlateCarree())
ax_lr = fig.add_subplot(gs[4, 6], projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['LOD']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm_re, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('LOD-Obs', fontsize=subsize)

# ax_lr = fig.add_subplot(5, 7, 29, projection=cartopy.crs.PlateCarree())
ax_lr = fig.add_subplot(gs[5, 6], projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['AutoML']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm_re, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('AutoML-Obs', fontsize=subsize)

cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.5])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="plasma"), 
                  cax=cbar_ax,
                  # ax=ax,
                  shrink=.5)
cb.set_label(label='$R^2$ score', fontsize=fontsize)

'''cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_re, cmap="plasma"), 
                  cax=cbar_ax)
cbar_ax.yaxis.set_ticks_position('left')
cbar_ax.yaxis.set_label_position('left')
cbar_ax.set_ylabel(ylabel='Observation', fontsize=fontsize)'''

# subplot2
scores_LOD = np.load('../dataResult/scores_LOD_smooth.npy')
scores_LA = np.load('../dataResult/scores_LA_smooth.npy')
scores_RD = np.load('../dataResult/scores_RD_smooth.npy')
scores_LR = np.load('../dataResult/scores_LR_smooth.npy')
scores_AutoML= np.load('../dataResult/scores_AutoML_smooth.npy')
scores_AutoLR = np.load('../dataResult/scores_AutoLR_smooth.npy')

scores_LOD_lag3 = np.load('../dataResult/scores_LOD_smooth_lag3.npy')
scores_LA_lag3 = np.load('../dataResult/scores_LA_smooth_lag3.npy')
scores_RD_lag3 = np.load('../dataResult/scores_RD_smooth_lag3.npy')
scores_LR_lag3 = np.load('../dataResult/scores_LR_smooth_lag3.npy')
scores_AutoML_lag3 = np.load('../dataResult/scores_AutoML_smooth_lag3.npy')
scores_AutoLR_lag3 = np.load('../dataResult/scores_AutoLR_smooth_lag3.npy')

LODs = np.zeros((25, 6))
LAs = np.zeros((25, 6))
RDs = np.zeros((25, 6))
LRs = np.zeros((25, 6))
PLSs = np.zeros((25, 6))
MLs = np.zeros((25, 6))
MLLRs = np.zeros((25, 6))

station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]
for i in range(len(station_peaks)):
    peak = station_peaks[i]
    if peak>3 and peak<12: # summer peaks
        LODs[i, :] = scores_LOD_lag3[i, :]
        LAs[i, :] = scores_LA_lag3[i, :]
        RDs[i, :] = scores_RD_lag3[i, :]
        LRs[i, :] = scores_LR_lag3[i, :]
        MLs[i, :] = scores_AutoML_lag3[i, :]
        MLLRs[i, :] = scores_AutoLR_lag3[i, :]
        PLSs[i, :] = scores_PLS_lag3[i, :]
    else:
        LODs[i, :] = scores_LOD[i, :]
        LAs[i, :] = scores_LA[i, :]
        RDs[i, :] = scores_RD[i, :]
        LRs[i, :] = scores_LR[i, :]
        MLs[i, :] = scores_AutoML[i, :]
        MLLRs[i, :] = scores_AutoLR[i, :]
        PLSs[i, :] = scores_PLS[i, :]

def plot_lines(ax, LRs, RDs, LAs, PLSs, LODs, MLLRs, file_name='smooth-line.png'):
    space = 3.2
    for i in range(6):
        bplot = ax.boxplot(LRs[:, i], positions=[.2+i*space], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:red')
        bplot['medians'][0].set_color('black')
        bplot = ax.boxplot(RDs[:, i], positions=[.6+i*space], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:green')
        bplot['medians'][0].set_color('black')
        bplot = ax.boxplot(LAs[:, i], positions=[1+i*space], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:blue')
        bplot['medians'][0].set_color('black')
        bplot = ax.boxplot(PLSs[:, i], positions=[1.4+i*space], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:brown')
        bplot['medians'][0].set_color('black')
        bplot = ax.boxplot(LODs[:, i], positions=[1.8+i*space], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:purple')
        bplot['medians'][0].set_color('black')
        bplot = ax.boxplot(MLLRs[:, i], positions=[2.2+i*space], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:orange')
        bplot['medians'][0].set_color('black')
    ax.set_xticks(np.arange(1.2, 19, space), np.arange(1, 7))
    ax.set_xticklabels(['EOF1', 'EOF1-2', 'EOF1-3', 
                        'EOF1-4', 'EOF1-5', 'EOF1-6'], fontsize=fontsize)
    # [1, 4, 7, 10, 13, 16]
    main_linewidth = 1
    sub_linewidth = .5
    for i in range(5):
        # Line connect two EOFs
        if i==0:
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(LRs[:, i:i+2], axis=0), 
                    label='LR', color='tab:red', linestyle='-', 
                    linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(LAs[:, i:i+2], axis=0), 
                    label='Lasso', color='tab:blue', linestyle='-', 
                    linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(RDs[:, i:i+2], axis=0), 
                    label='Ridge', color='tab:green', linestyle='-', 
                    linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(PLSs[:, i:i+2], axis=0), 
                    label='PLS', color='tab:brown', linestyle='-', 
                    linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(LODs[:, i:i+2], axis=0), 
                    label='LOD', color='tab:purple', linestyle='-', 
                    linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(MLLRs[:, i:i+2], axis=0), 
                    label='AutoML', color='tab:orange', linestyle='-', 
                    linewidth=main_linewidth)
        else:
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(LRs[:, i:i+2], axis=0), 
                    color='tab:red', linestyle='-', linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(LAs[:, i:i+2], axis=0), 
                    color='tab:blue', linestyle='-', linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(RDs[:, i:i+2], axis=0), 
                    color='tab:green', linestyle='-', linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(LODs[:, i:i+2], axis=0), 
                    color='tab:purple', linestyle='-', linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(PLSs[:, i:i+2], axis=0), 
                    color='tab:brown', linestyle='-', linewidth=main_linewidth)
            ax.plot([2.5+i*space, -0.1+i*space+space], np.median(MLLRs[:, i:i+2], axis=0), 
                    color='tab:orange', linestyle='-', linewidth=main_linewidth)
        # previous line for median 
        ax.plot([.2+i*space, 2.5+i*space], 
                [np.median(LRs[:, i]), np.median(LRs[:, i])],
                color='tab:red', linestyle='--', linewidth=sub_linewidth)
        ax.plot([.6+i*space, 2.5+i*space], 
                [np.median(RDs[:, i]), np.median(RDs[:, i])],
                color='tab:green', linestyle='--', linewidth=sub_linewidth)
        ax.plot([1+i*space, 2.5+i*space], 
                [np.median(LAs[:, i]), np.median(LAs[:, i])],
                color='tab:blue', linestyle='--', linewidth=sub_linewidth)
        ax.plot([1.8+i*space, 2.5+i*space], 
                [np.median(LODs[:, i]), np.median(LODs[:, i])],
                color='tab:purple', linestyle='--', linewidth=sub_linewidth)
        ax.plot([1.4+i*space, 2.5+i*space], 
                [np.median(PLSs[:, i]), np.median(PLSs[:, i])],
                color='tab:brown', linestyle='--', linewidth=sub_linewidth)
        ax.plot([2.2+i*space, 2.5+i*space], 
                [np.median(MLLRs[:, i]), np.median(MLLRs[:, i])],
                color='tab:orange', linestyle='--', linewidth=sub_linewidth)
        # After line for median. 
        ax.plot([-0.1+i*space+space, 0.2+i*space+space], 
                [np.median(LRs[:, i+1]), np.median(LRs[:, i+1])],
                color='tab:red', linestyle='--', linewidth=sub_linewidth)
        ax.plot([-0.1+i*space+space, 0.6+i*space+space], 
                [np.median(RDs[:, i+1]), np.median(RDs[:, i+1])],
                color='tab:green', linestyle='--', linewidth=sub_linewidth)
        ax.plot([-0.1+i*space+space, 1+i*space+space], 
                [np.median(LAs[:, i+1]), np.median(LAs[:, i+1])],
                color='tab:blue', linestyle='--', linewidth=sub_linewidth)
        ax.plot([-0.1+i*space+space, 1.4+i*space+space], 
                [np.median(PLSs[:, i+1]), np.median(PLSs[:, i+1])],
                color='tab:brown', linestyle='--', linewidth=sub_linewidth)
        ax.plot([-0.1+i*space+space, 1.8+i*space+space], 
                [np.median(LODs[:, i+1]), np.median(LODs[:, i+1])],
                color='tab:purple', linestyle='--', linewidth=sub_linewidth)
        ax.plot([-0.1+i*space+space, 2.2+i*space+space], 
                [np.median(MLLRs[:, i+1]), np.median(MLLRs[:, i+1])],
                color='tab:orange', linestyle='--', linewidth=sub_linewidth)


    ax.legend(fontsize=fontsize, ncols=2)
    ax.set_xlabel('EOFs used in ML models', fontsize=fontsize)
    ax.set_ylabel('Median $R^2$ Score', fontsize=fontsize)

    
ax = fig.add_subplot(gs[6:, :])
# ax.set_title('B', fontsize=15, weight='bold', loc='left')

ax.annotate('B)', xy=(-0.05, 1.0), xycoords='axes fraction', 
            fontsize=14, weight='bold')
ax.annotate('A)', xy=(-0.05, 4.25), xycoords='axes fraction', 
            fontsize=14, weight='bold')
plot_lines(ax, LRs, RDs, LAs, PLSs, LODs, MLs)

plt.savefig('Figure2.png', bbox_inches='tight', dpi=180)
