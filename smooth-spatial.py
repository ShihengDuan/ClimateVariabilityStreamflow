import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from sklearn.metrics import r2_score
import os
import pandas as pd
import cartopy
import matplotlib as mpl

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]

# Load Data
path = '/p/lustre2/shiduan/LOD-predictions-smooth/'
if not os.path.exists('dataResult/scores_LOD_smooth.npy'):
    scores_LOD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real.npy')
                print(station, ' ', eof, ' ', seed)
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LOD[i, k] = r2
    print(np.max(scores_LOD, axis=1)) # 25 stations. 
    print('LOD', np.median(np.max(scores_LOD, axis=1)))
    print('LOD', np.median(scores_LOD[:, -2]), np.mean(scores_LOD[:, -2]))
    np.save('dataResult/scores_LOD_smooth', scores_LOD)
else:
    scores_LOD = np.load('dataResult/scores_LOD_smooth.npy')

path = '/p/lustre2/shiduan/Lasso-predictions-smooth/'
if not os.path.exists('dataResult/scores_LA_smooth.npy'):
    scores_LA = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real.npy')
                print(station, ' ', eof, ' ', seed)
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LA[i, k] = r2
    print(np.max(scores_LA, axis=1)) # 25 stations. 
    print('Lasso', np.median(np.max(scores_LA, axis=1)))
    print('Lasso', np.median(scores_LA[:, -2]), np.mean(scores_LA[:, -2]))
    np.save('dataResult/scores_LA_smooth', scores_LA)
else:
    scores_LA = np.load('dataResult/scores_LA_smooth.npy')

path = '/p/lustre2/shiduan/Ridge-predictions-smooth/'
if not os.path.exists('dataResult/scores_RD_smooth.npy'):
    scores_RD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real.npy')
                print(station, ' ', eof, ' ', seed)
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_RD[i, k] = r2
    print(np.max(scores_RD, axis=1)) # 25 stations. 
    print('Ridge', np.median(np.max(scores_RD, axis=1)))
    print('Ridge', np.median(scores_RD[:, -2]), np.mean(scores_RD[:, -2]))
    np.save('dataResult/scores_RD_smooth', scores_RD)
else:
    scores_RD = np.load('dataResult/scores_RD_smooth.npy')

path = '/p/lustre2/shiduan/Linear-predictions-smooth/'
if not os.path.exists('dataResult/scores_LR_smooth.npy'):
    scores_LR = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LR[i, k] = r2
    print(np.max(scores_LR, axis=1)) # 25 stations. 
    print('LR', np.median(np.max(scores_LR, axis=1)))
    print('LR', np.median(scores_LR[:, -2]), np.mean(scores_LR[:, -2]))
    np.save('dataResult/scores_LR_smooth', scores_LR)
else:
    scores_LR = np.load('dataResult/scores_LR_smooth.npy')

path = '/p/lustre2/shiduan/AutoML-predictions-smooth/'
if not os.path.exists('dataResult/scores_ML_smooth.npy'):
    scores_ML = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real.npy')
                pred = np.load(
                    path+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_ML[i, k] = r2
    print(np.max(scores_ML, axis=1)) # 25 stations. 
    print('ML', np.median(np.max(scores_ML, axis=1)))
    print('ML', np.median(scores_ML[:, -2]), np.mean(scores_ML[:, -2]))
    np.save('dataResult/scores_ML_smooth', scores_ML)
else:
    scores_ML = np.load('dataResult/scores_ML_smooth.npy')

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
    ax = fig.add_subplot(5, 6, ind, projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
        else:
            marker='o'
        ax.scatter(lons[j], lats[j], c=scores_LR[j, i], norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('LR-EOF-'+str(ind), fontsize=12)
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 6, ind+6, projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
        else:
            marker='o'
        ax.scatter(lons[j], lats[j], c=scores_RD[j, i], norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('Ridge-EOF-'+str(ind), fontsize=12)
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 6, ind+12, projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
        else:
            marker='o'
        ax.scatter(lons[j], lats[j], c=scores_LA[j, i], norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('Lasso-EOF-'+str(ind), fontsize=12)
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 6, ind+18, projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
        else:
            marker='o'
        ax.scatter(lons[j], lats[j], c=scores_LOD[j, i], norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('LOD-EOF-'+str(ind), fontsize=12)
for i in range(6):
    ind = i+1
    ax = fig.add_subplot(5, 6, ind+24, projection=cartopy.crs.PlateCarree())
    for j in range(len(station_peaks)):
        peak = station_peaks[j]
        if peak>3 and peak<12:
            marker='^'
        else:
            marker='o'
        ax.scatter(lons[j], lats[j], c=scores_ML[j, i], norm=norm, 
                   transform=cartopy.crs.PlateCarree(), cmap='plasma', marker=marker)
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title('AutoML-EOF-'+str(ind), fontsize=12)
plt.tight_layout()
fig.subplots_adjust(right=0.94)
cbar_ax = fig.add_axes([0.945, 0.05, 0.015, 0.9])
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="plasma"), cax=cbar_ax)

plt.savefig('smooth-spatial.png', bbox_inches='tight', dpi=180)
