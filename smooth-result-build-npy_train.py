import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import os

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 

reload = True

path = '/p/lustre2/shiduan/LOD-predictions-smooth/'
if not os.path.exists('dataResult/scores_LOD_smooth.npy') or reload:
    scores_LOD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_train.npy')
                # print(station, ' ', eof, ' ', seed)
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LOD[i, k] = r2
    # print(np.max(scores_LOD, axis=1)) # 25 stations. 
    # print('LOD', np.median(np.max(scores_LOD, axis=1)))
    # print('LOD', np.median(scores_LOD[:, -2]), np.mean(scores_LOD[:, -2]))
    print('LOD loaded')
    np.save('dataResult/scores_LOD_smooth_train', scores_LOD)
else:
    scores_LOD = np.load('dataResult/scores_LOD_smooth_train.npy')

path = '/p/lustre2/shiduan/LASSO-predictions-smooth/'
if not os.path.exists('dataResult/scores_LA_smooth_train.npy') or reload:
    scores_LA = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_train.npy')
                # print(station, ' ', eof, ' ', seed)
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LA[i, k] = r2
    # print(np.max(scores_LA, axis=1)) # 25 stations. 
    # print('Lasso', np.median(np.max(scores_LA, axis=1)))
    # print('Lasso', np.median(scores_LA[:, -2]), np.mean(scores_LA[:, -2]))
    print('Lasso loaded')
    np.save('dataResult/scores_LA_smooth_train', scores_LA)
else:
    scores_LA = np.load('dataResult/scores_LA_smooth_train.npy')

path = '/p/lustre2/shiduan/RIDGE-predictions-smooth/'
if not os.path.exists('dataResult/scores_RD_smooth_train.npy') or reload:
    scores_RD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_train.npy')
                # print(station, ' ', eof, ' ', seed)
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_RD[i, k] = r2
    print('Ridge loaded')
    np.save('dataResult/scores_RD_smooth_train', scores_RD)
else:
    scores_RD = np.load('dataResult/scores_RD_smooth_train.npy')

path = '/p/lustre2/shiduan/LR-predictions-smooth/'
if not os.path.exists('dataResult/scores_LR_smooth_train.npy') or reload:
    scores_LR = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_train.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LR[i, k] = r2
    print('LR loaded ')
    np.save('dataResult/scores_LR_smooth_train', scores_LR)
else:
    scores_LR = np.load('dataResult/scores_LR_smooth_train.npy')
    print('LR', np.median(np.median(scores_LR[:, -1])))

path = '/p/lustre2/shiduan/AUTOML-predictions-smooth/'
if not os.path.exists('dataResult/scores_AutoML_smooth_train.npy') or reload:
    scores_ML = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_train.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_ML[i, k] = r2
    print('AutoML loaded')
    np.save('dataResult/scores_AutoML_smooth_train', scores_ML)
else:
    scores_ML = np.load('dataResult/scores_AutoML_smooth_train.npy')

path = '/p/lustre2/shiduan/AUTOLR-predictions-smooth/'
if not os.path.exists('dataResult/scores_AutoLR_smooth_train.npy') or reload:
    scores_ML = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_train.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_ML[i, k] = r2
    print('AutoLR loaded')
    np.save('dataResult/scores_AutoLR_smooth_train', scores_ML)
else:
    scores_ML = np.load('dataResult/scores_AutoLR_smooth_train.npy')

###
### Lag3
###
path = '/p/lustre2/shiduan/LOD-predictions-smooth/'
if not os.path.exists('dataResult/scores_LOD_smooth_lag3_train.npy') or reload:
    scores_LOD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3_train.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LOD[i, k] = r2
    print('LOD lag3')
    np.save('dataResult/scores_LOD_smooth_lag3_train', scores_LOD)
else:
    scores_LOD = np.load('dataResult/scores_LOD_smooth_lag3_train.npy')

path = '/p/lustre2/shiduan/LASSO-predictions-smooth/'
if not os.path.exists('dataResult/scores_LA_smooth_lag3_train.npy') or reload:
    scores_LA = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3_train.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LA[i, k] = r2
    print('Lasso lag3')
    np.save('dataResult/scores_LA_smooth_lag3_train', scores_LA)
else:
    scores_LA = np.load('dataResult/scores_LA_smooth_lag3_train.npy')

path = '/p/lustre2/shiduan/RIDGE-predictions-smooth/'
if not os.path.exists('dataResult/scores_RD_smooth_lag3_train.npy') or reload:
    scores_RD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3_train.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_RD[i, k] = r2
    print('Ridge lag3')
    np.save('dataResult/scores_RD_smooth_lag3_train', scores_RD)
else:
    scores_RD = np.load('dataResult/scores_RD_smooth_lag3_train.npy')

path = '/p/lustre2/shiduan/LR-predictions-smooth/'
if not os.path.exists('dataResult/scores_LR_smooth_lag3_train.npy') or reload:
    scores_LR = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3_train.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3_train.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LR[i, k] = r2
    print('LR lag3')
    np.save('dataResult/scores_LR_smooth_lag3_train', scores_LR)
else:
    scores_LR = np.load('dataResult/scores_LR_smooth_lag3_train.npy')


path = '/p/lustre2/shiduan/AUTOML-predictions-smooth/'
if not os.path.exists('dataResult/scores_AutoML_smooth_lag3_train.npy') or reload:
    scores_ML = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                file = path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3_train.npy'
                if not os.path.exists(file):
                    print(i, station, eof)
                else:
                    real = np.load(
                        path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3_train.npy')
                    pred = np.load(
                        path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3_train.npy')
                    reals.append(real.reshape(-1, 1))
                    preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_ML[i, k] = r2
    print('AutoML lag3')
    np.save('dataResult/scores_AutoML_smooth_lag3_train', scores_ML)
else:
    scores_ML = np.load('dataResult/scores_AutoML_smooth_lag3_train.npy')

path = '/p/lustre2/shiduan/AUTOLR-predictions-smooth/'
if not os.path.exists('dataResult/scores_AutoLR_smooth_lag3_train.npy') or reload:
    scores_ML = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                file = path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3_train.npy'
                if not os.path.exists(file):
                    print(i, station, eof)
                else:
                    real = np.load(
                        path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3_train.npy')
                    pred = np.load(
                        path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3_train.npy')
                    reals.append(real.reshape(-1, 1))
                    preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_ML[i, k] = r2
    print('AutoML lag3')
    np.save('dataResult/scores_AutoLR_smooth_lag3_train', scores_ML)
else:
    scores_ML = np.load('dataResult/scores_AutoLR_smooth_lag3_train.npy')