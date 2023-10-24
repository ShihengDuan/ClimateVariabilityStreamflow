# feature importance by removing only 1 input variable. 
# ```remove```
import numpy as np
from sklearn.metrics import r2_score
import glob
from matplotlib import pyplot as plt
import pickle
import cartopy
import pandas as pd
import matplotlib.patches as mpatches

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
path = '/p/lustre2/shiduan/'
mods = ['PNA_eof_1', 'PNA_eof_2', 'PNA_eof_3', 'PNA_eof_4', 'PNA_eof_5', 'PNA_eof_6', 
        'PDO_eof_1', 'PDO_eof_2', 'PDO_eof_3', 'PDO_eof_4', 'PDO_eof_5', 'PDO_eof_6', 
        'AMO_eof_1', 'AMO_eof_2', 'AMO_eof_3', 'AMO_eof_4', 'AMO_eof_5', 'AMO_eof_6', 
        'NAM_eof_1', 'NAM_eof_2', 'NAM_eof_3', 'NAM_eof_4', 'NAM_eof_5', 'NAM_eof_6', 
        'SAM_eof_1', 'SAM_eof_2', 'SAM_eof_3', 'SAM_eof_4', 'SAM_eof_5', 'SAM_eof_6', 
        'NAO_eof_1', 'NAO_eof_2', 'NAO_eof_3', 'NAO_eof_4', 'NAO_eof_5', 'NAO_eof_6', 
        'co2', 'nino34']
station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]
names = mods
eof = 6
r2s_lasso = {}
r2s_ridge = {}
r2s_lod = {}
r2s_ml = {}
r2s_linear = {}
def get_prediction(model_type, station, mod, lag3, seed):
    if lag3:
        pred_path = '/p/lustre2/shiduan/'+model_type.upper()+'-predictions-smooth/remove-'+mod+'/'+station+'/'+station+'-EOF-6-seed-'+str(seed)+'-pred_lag3.npy'
        real_path = '/p/lustre2/shiduan/'+model_type.upper()+'-predictions-smooth/remove-'+mod+'/'+station+'/'+station+'-EOF-6-seed-'+str(seed)+'-real_lag3.npy'
    else:
        pred_path = '/p/lustre2/shiduan/'+model_type.upper()+'-predictions-smooth/remove-'+mod+'/'+station+'/'+station+'-EOF-6-seed-'+str(seed)+'-pred_lag3.npy'
        real_path = '/p/lustre2/shiduan/'+model_type.upper()+'-predictions-smooth/remove-'+mod+'/'+station+'/'+station+'-EOF-6-seed-'+str(seed)+'-real_lag3.npy'
    pred = np.load(pred_path)
    real = np.load(real_path)
    return real, pred

re_load = True
if re_load:
    for ind, station in enumerate(station_ids):
        peak = station_peaks[ind]
        if peak>3 and peak<12:
            lag3 = True
        else:
            lag3 = False
        r2s = []
        print(station, peak, lag3)
        for mod in mods:
            reals = []
            preds = []
            for seed in range(6):
                real, pred = get_prediction(model_type='Ridge', station=station, 
                                            mod=mod, lag3=lag3, seed=seed)
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            r2s.append(r2)
            if station=='10336645':
                r2s_ridge[mod] = [r2]
            else:
                r2s_ridge[mod].append(r2)
        reals = []
        preds = []
        for seed in range(6):
            if lag3:
                real_file = path+'RIDGE-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real_lag3.npy'
                pred_file = path+'RIDGE-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred_lag3.npy'
            else:
                real_file = path+'RIDGE-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real.npy'
                pred_file = path+'RIDGE-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred.npy'
            real = np.load(real_file)
            pred = np.load(pred_file)
            reals.append(real.reshape(-1, 1))
            preds.append(pred.reshape(-1, 1))
        reals = np.concatenate(reals)
        preds = np.concatenate(preds)
        r2_full = r2_score(reals, preds)
        print('Ridge full r2: ', "{:.4f}".format(r2_full))
        if station =='10336645':
            r2s_ridge['full'] = [r2_full]
        else:
            r2s_ridge['full'].append(r2_full)
        second = np.argsort(r2s)[-2]
        third = np.argsort(r2s)[-3]
        Ridge_output = 'Ridge: '+mods[np.argmax(r2s)]+' '+"{:.4f}".format(np.max(r2s))+' '+mods[second]+' '+"{:.4f}".format(r2s[second])+' '+mods[third]+' '+"{:.4f}".format(r2s[third])
        # Linear
        r2s = []
        for mod in mods:
            reals = []
            preds = []
            for seed in range(6):
                real, pred = get_prediction(model_type='LR', station=station, 
                                            mod=mod, lag3=lag3, seed=seed)
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            r2s.append(r2)
            if station=='10336645':
                r2s_linear[mod] = [r2]
            else:
                r2s_linear[mod].append(r2)
        reals = []
        preds = []
        for seed in range(6):
            if lag3:
                real_file = path+'LR-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real_lag3.npy'
                pred_file = path+'LR-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred_lag3.npy'
            else:
                real_file = path+'LR-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real.npy'
                pred_file = path+'LR-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred.npy'
            real = np.load(real_file)
            pred = np.load(pred_file)
            reals.append(real.reshape(-1, 1))
            preds.append(pred.reshape(-1, 1))
        reals = np.concatenate(reals)
        preds = np.concatenate(preds)
        r2_full = r2_score(reals, preds)
        print('Linear full r2: ', "{:.4f}".format(r2_full))
        if station=='10336645':
            r2s_linear['full'] = [r2_full]
        else:
            r2s_linear['full'].append(r2_full)
        second = np.argsort(r2s)[-2]
        third = np.argsort(r2s)[-3]
        linear_output = 'Linear: '+mods[np.argmax(r2s)]+' '+"{:.4f}".format(np.max(r2s))+' '+mods[second]+' '+"{:.4f}".format(r2s[second])+' '+mods[third]+' '+"{:.4f}".format(r2s[third])
        ## LOD
        r2s = []
        for mod in mods:
            reals = []
            preds = []
            for seed in range(6):
                real, pred = get_prediction(model_type='LOD', station=station, 
                                            mod=mod, lag3=lag3, seed=seed)
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            r2s.append(r2)
            if station=='10336645':
                r2s_lod[mod] = [r2]
            else:
                r2s_lod[mod].append(r2)
        reals = []
        preds = []
        for seed in range(6):
            if lag3:
                real_file = path+'LOD-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real_lag3.npy'
                pred_file = path+'LOD-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred_lag3.npy'
            else:
                real_file = path+'LOD-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real.npy'
                pred_file = path+'LOD-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred.npy'
            real = np.load(real_file)
            pred = np.load(pred_file)
            reals.append(real.reshape(-1, 1))
            preds.append(pred.reshape(-1, 1))
        reals = np.concatenate(reals)
        preds = np.concatenate(preds)
        r2_full = r2_score(reals, preds)
        print('LOD full r2: ', "{:.4f}".format(r2_full))
        if station=='10336645':
            r2s_lod['full'] = [r2_full]
        else:
            r2s_lod['full'].append(r2_full)
        second = np.argsort(r2s)[-2]
        third = np.argsort(r2s)[-3]
        lod_output = 'LOD: '+mods[np.argmax(r2s)]+' '+"{:.4f}".format(np.max(r2s))+' '+mods[second]+' '+"{:.4f}".format(r2s[second])+' '+mods[third]+' '+"{:.4f}".format(r2s[third])
        # AutoML
        r2s = []
        for mod in mods:
            reals = []
            preds = []
            for seed in range(6):
                real, pred = get_prediction(model_type='AutoML', station=station, 
                                            mod=mod, lag3=lag3, seed=seed)
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
                print(real.shape, pred.shape)
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            if station=='10336645':
                r2s_ml[mod] = [r2]
            else:
                r2s_ml[mod].append(r2)
            r2s.append(r2)
        reals = []
        preds = []
        for seed in range(6):
            if lag3:
                real_file = path+'AUTOML-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real_lag3.npy'
                pred_file = path+'AUTOML-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred_lag3.npy'
            else:
                real_file = path+'AUTOML-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real.npy'
                pred_file = path+'AUTOML-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred.npy'
            real = np.load(real_file)
            pred = np.load(pred_file)
            reals.append(real.reshape(-1, 1))
            preds.append(pred.reshape(-1, 1))
        reals = np.concatenate(reals)
        preds = np.concatenate(preds)
        r2_full = r2_score(reals, preds)
        print('AutoML full r2: ', "{:.4f}".format(r2_full))
        if station=='10336645':
            r2s_ml['full'] = [r2_full]
        else:
            r2s_ml['full'].append(r2_full)
        second = np.argsort(r2s)[-2]
        third = np.argsort(r2s)[-3]
        auto_output = 'AutoML: '+mods[np.argmax(r2s)]+' '+"{:.4f}".format(np.max(r2s))+' '+mods[second]+' '+"{:.4f}".format(r2s[second])+' '+mods[third]+' '+"{:.4f}".format(r2s[third])
        ## Lasso
        r2s = []
        for mod in mods:
            reals = []
            preds = []
            for seed in range(6):
                real, pred = get_prediction(model_type='LASSO', station=station, 
                                            mod=mod, lag3=lag3, seed=seed)
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            r2s.append(r2)
            if station=='10336645':
                r2s_lasso[mod] = [r2]
            else:
                r2s_lasso[mod].append(r2)
        reals = []
        preds = []
        for seed in range(6):
            if lag3:
                real_file = path+'LASSO-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real_lag3.npy'
                pred_file = path+'LASSO-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred_lag3.npy'
            else:
                real_file = path+'LASSO-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-real.npy'
                pred_file = path+'LASSO-predictions-smooth/'+station+'/'+station+'-EOF-6'+'-seed-'+str(seed)+'-pred.npy'
            real = np.load(real_file)
            pred = np.load(pred_file)
            reals.append(real.reshape(-1, 1))
            preds.append(pred.reshape(-1, 1))
        reals = np.concatenate(reals)
        preds = np.concatenate(preds)
        r2_full = r2_score(reals, preds)
        print('Lasso full r2: ', "{:.4f}".format(r2_full))
            # print(mod,' r2: ', r2)
        if station=='10336645':
            r2s_lasso['full'] = [r2_full]
        else:
            r2s_lasso['full'].append(r2_full)
        second = np.argsort(r2s)[-2]
        third = np.argsort(r2s)[-3]
        Lasso_output = 'Lasso: '+mods[np.argmax(r2s)]+' '+"{:.4f}".format(np.max(r2s))+' '+mods[second]+' '+"{:.4f}".format(r2s[second])+' '+mods[third]+' '+"{:.4f}".format(r2s[third])
        print(Ridge_output)
        print(linear_output)
        print(lod_output)
        print(auto_output)
        print(Lasso_output)
        print()
    r2s_remove = [r2s_linear, r2s_lasso, r2s_ridge, r2s_lod, r2s_ml]
    with open('r2s_remove_eof_smooth.p', 'wb') as pfile:
        pickle.dump(r2s_remove, pfile)
else:
    with open('r2s_remove_eof_smooth.p', 'rb') as pfile:
        r2s_remove = pickle.load(pfile)
    r2s_linear, r2s_lasso, r2s_ridge, r2s_lod, r2s_ml = r2s_remove


lats = []
lons = []
camel_topo = pd.read_csv(
    '/usr/workspace/shiduan/neuralhydrology/data/camels_us/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_topo.txt', delimiter=';')
for station in station_ids:
    record = camel_topo[camel_topo['gauge_id']==int(station)]
    lats.append(record['gauge_lat'])
    lons.append(record['gauge_lon'])
# print top predictors for each station
fig = plt.figure(figsize=(5, 5))
ax1 = fig.add_subplot(121, projection=cartopy.crs.PlateCarree())
ax2 = fig.add_subplot(122, projection=cartopy.crs.PlateCarree())
for ind, station in enumerate(station_ids):
    print(station)
    peak = station_peaks[ind]
    if peak>3 and peak<12:
        marker='^'
    else:
        marker='o'
    r2s = []
    for mod in mods:
        r2_linear = r2s_linear[mod][ind]
        r2s.append(r2_linear)
    mod_index = np.argmax(r2s)
    second = np.argsort(r2s)[-2]
    third = np.argsort(r2s)[-3]
    print('top linear: ', mods[mod_index], ' ',r2s[mod_index], 
          ' ', mods[second], ' ',r2s[second], 
          ' ', mods[third], ' ', r2s[third])
    if mods[mod_index]=='PNA_eof_5':
        ax1.scatter(lons[ind], lats[ind], marker=marker, color='red', label='PNA-5')
    elif mods[mod_index]=='PDO_eof_3':
        ax1.scatter(lons[ind], lats[ind], marker=marker, color='blue', label='PDO-3')
    elif mods[mod_index]=='PDO_eof_5':
        ax1.scatter(lons[ind], lats[ind], marker=marker, color='green', label='PDO-5')
    else:
        print('No such index')
    if mods[second]=='PNA_eof_5':
        ax2.scatter(lons[ind], lats[ind], marker=marker, color='red', label='PNA-5')
    elif mods[second]=='PDO_eof_3':
        ax2.scatter(lons[ind], lats[ind], marker=marker, color='blue', label='PDO-3')
    elif mods[second]=='PDO_eof_5':
        ax2.scatter(lons[ind], lats[ind], marker=marker, color='green', label='PDO-5')
    else:
        print('No such index')
    
    r2s = []
    for mod in mods:
        r2 = r2s_ridge[mod][ind]
        r2s.append(r2)
    mod_index = np.argmax(r2s)
    second = np.argsort(r2s)[-2]
    third = np.argsort(r2s)[-3]
    print('top Ridge: ', mods[mod_index], ' ',r2s[mod_index], 
          ' ', mods[second], ' ',r2s[second], 
          ' ', mods[third], ' ', r2s[third])

    r2s = []
    for mod in mods:
        r2 = r2s_lasso[mod][ind]
        r2s.append(r2)
    mod_index = np.argmax(r2s)
    second = np.argsort(r2s)[-2]
    third = np.argsort(r2s)[-3]
    print('top Lasso: ', mods[mod_index], ' ',r2s[mod_index], 
          ' ', mods[second], ' ',r2s[second], 
          ' ', mods[third], ' ', r2s[third])

    r2s = []
    for mod in mods:
        r2 = r2s_lod[mod][ind]
        r2s.append(r2)
    mod_index = np.argmax(r2s)
    second = np.argsort(r2s)[-2]
    third = np.argsort(r2s)[-3]
    print('top LOD: ', mods[mod_index], ' ',r2s[mod_index], 
          ' ', mods[second], ' ',r2s[second], 
          ' ', mods[third], ' ', r2s[third])

    r2s = []
    for mod in mods:
        r2 = r2s_ml[mod][ind]
        r2s.append(r2)
    mod_index = np.argmax(r2s)
    second = np.argsort(r2s)[-2]
    third = np.argsort(r2s)[-3]
    print('top ML: ', mods[mod_index], ' ',r2s[mod_index], 
          ' ', mods[second], ' ',r2s[second], 
          ' ', mods[third], ' ', r2s[third], '\n')

ax1.add_feature(cartopy.feature.STATES)
ax2.add_feature(cartopy.feature.STATES)
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# fig.legend(by_label.values(), by_label.keys(), fontsize=12, 
#            loc='upper center', ncol=3, bbox_to_anchor=(0.5, .9))
PNA5_patch = mpatches.Patch(color='red', label='PNA-5')
PDO5_patch = mpatches.Patch(color='green', label='PDO-5')
PDO3_patch = mpatches.Patch(color='blue', label='PDO-3')
fig.legend(handles=[PNA5_patch, PDO5_patch, PDO3_patch], fontsize=12, 
           loc='upper center', ncol=3, bbox_to_anchor=(0.5, .9))
plt.tight_layout()
plt.savefig('Cal-firstVariable-spatial-remove.png', dpi=150, bbox_inches='tight')


fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=False, figsize=(8, 8))
ax = axes.flatten()[0]
ax.set_title('Linear Regression')
# print('Linear: ')
for i, key in enumerate(r2s_linear):
    # print(key, ' ', np.median(r2s_linear[key]))
    if not key =='full':
        ax.bar(i, np.median(r2s_linear[key]))
    else:
        ax.hlines(np.median(r2s_linear[key]), 0, 38, color='black', linestyle='dashed')
        # ax.set_ylim([0, np.median(r2s_linear[key])+0.05])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_yticklabels([0, 0.1, 0.2, 0.3])

ax = axes.flatten()[1]
ax.set_title('Ridge')
# print('\nRidge: ')
for i, key in enumerate(r2s_linear):
    # print(key, ' ', np.median(r2s_ridge[key]))
    if not key =='full':
        ax.bar(i, np.median(r2s_ridge[key]))
    else:
        ax.hlines(np.median(r2s_ridge[key]), 0, 38,  color='black', linestyle='dashed')
        # ax.set_ylim([0, np.median(r2s_ridge[key])+0.05])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_yticklabels([0, 0.1, 0.2, 0.3])

ax = axes.flatten()[2]
ax.set_title('Lasso')
# print('\nLasso: ')
for i, key in enumerate(r2s_linear):
    # print(key, ' ', np.median(r2s_lasso[key]))
    if not key =='full':
        ax.bar(i, np.median(r2s_lasso[key]))
    else:
        ax.hlines(np.median(r2s_lasso[key]), 0, 38, color='black', linestyle='dashed')
        # ax.set_ylim([0, np.median(r2s_lasso[key])+0.05])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.set_yticklabels([0, 0.1, 0.2, 0.3])

ax = axes.flatten()[3]
ax.set_title('LOD')
# print('\nLOD: ')
for i, key in enumerate(r2s_linear):
    # print(key, ' ', np.median(r2s_lod[key]))
    if not key =='full':
        ax.bar(i, np.median(r2s_lod[key]))
    else:
        ax.hlines(np.median(r2s_lod[key]), 0, 38, color='black', linestyle='dashed')
        # ax.set_ylim([0, np.median(r2s_lod[key])+0.05])
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])

ax = axes.flatten()[4]
ax.set_title('AutoML')
# print('\nAutoML: ')
for i, key in enumerate(r2s_linear):
    # print(key, ' ', np.median(r2s_ml[key]))
    if not key =='full':
        ax.bar(i, np.median(r2s_ml[key]))
    else:
        ax.hlines(np.median(r2s_ml[key]), 0, 38, color='black', linestyle='dashed')
        # ax.set_ylim([0, np.median(r2s_ml[key])+0.05])
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])

# fig.delaxes(axes.flatten()[-1])
for i in range(5):
    ax = axes.flatten()[i]
    ax.set_xticks(np.arange(38))
    ax.set_xticklabels(names, rotation=90)
    # ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    # ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])
    # ax.set_ylim([0, 0.4])
plt.tight_layout()
plt.savefig('Cal-median-result-remove.png', dpi=150, bbox_inches='tight')
print('Done')




