import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.metrics import r2_score
import os
import pandas as pd
import cartopy
import matplotlib as mpl
import pickle

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
scores_ML_lag3 = np.load('../dataResult/scores_ML_smooth_lag3.npy')

scores_LOD = np.load('../dataResult/scores_LOD_smooth.npy')
scores_LA = np.load('../dataResult/scores_LA_smooth.npy')
scores_RD = np.load('../dataResult/scores_RD_smooth.npy')
scores_LR  = np.load('../dataResult/scores_LR_smooth.npy')
scores_ML = np.load('../dataResult/scores_ML_smooth.npy')



lats = []
lons = []
camel_topo = pd.read_csv(
    '/usr/workspace/shiduan/neuralhydrology/data/camels_us/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_topo.txt', delimiter=';')
for station in station_ids:
    record = camel_topo[camel_topo['gauge_id']==int(station)]
    lats.append(record['gauge_lat'])
    lons.append(record['gauge_lon'])

norm = colors.Normalize(vmin=0, vmax=0.5)

fig = plt.figure(figsize=(12, 12))
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 7, ind, projection=cartopy.crs.PlateCarree())
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
    ax.set_title('LR-EOF-'+str(ind), fontsize=12)
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 7, ind+7, projection=cartopy.crs.PlateCarree())
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
    ax.set_title('Ridge-EOF-'+str(ind), fontsize=12)
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 7, ind+14, projection=cartopy.crs.PlateCarree())
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
    ax.set_title('Lasso-EOF-'+str(ind), fontsize=12)
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 7, ind+21, projection=cartopy.crs.PlateCarree())
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
    ax.set_title('LOD-EOF-'+str(ind), fontsize=12)
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 7, ind+28, projection=cartopy.crs.PlateCarree())
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
    ax.set_title('AutoML-EOF-'+str(ind), fontsize=12)

# reanalysis
with open('../dataResult/Reanalysis/results.p', 'rb') as pfile:
    results = pickle.load(pfile)

ax_lr = fig.add_subplot(5, 7, 7, projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['LR']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('LR-Reanalysis', fontsize=12)

ax_lr = fig.add_subplot(5, 7, 14, projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['Ridge']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('Ridge-Reanalysis', fontsize=12)

ax_lr = fig.add_subplot(5, 7, 21, projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['Lasso']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('Lasso-Reanalysis', fontsize=12)

ax_lr = fig.add_subplot(5, 7, 28, projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['LOD']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('LOD-Reanalysis', fontsize=12)

ax_lr = fig.add_subplot(5, 7, 35, projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    score, eof_number = results[station]['AutoML']
    ax_lr.scatter(lons[ind], lats[ind], c=score, norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
ax_lr.add_feature(cartopy.feature.STATES)
ax_lr.set_title('AutoML-Reanalysis', fontsize=12)

plt.tight_layout()
fig.subplots_adjust(right=0.94)
cbar_ax = fig.add_axes([0.949, 0.05, 0.015, 0.9])
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="plasma"), cax=cbar_ax)

plt.savefig('performance-spatial-combine.png', bbox_inches='tight', dpi=180)
