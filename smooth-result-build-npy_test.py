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

'''# AutoML high
path = '/p/lustre2/shiduan/AutoML-predictions-smooth-high/'
if not os.path.exists('dataResult/scores_ML_smooth_high.npy'):
    scores_ML_high = np.zeros((25, 6))
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
            scores_ML_high[i, k] = r2
    print(np.max(scores_ML_high, axis=1)) # 25 stations. 
    print('ML', np.median(np.max(scores_ML_high, axis=1)))
    print('ML', np.median(scores_ML_high[:, -2]), np.mean(scores_ML_high[:, -2]))
    np.save('dataResult/scores_ML_smooth_high', scores_ML_high)
else:
    scores_ML_high = np.load('dataResult/scores_ML_smooth_high.npy')
    print('ML-high', np.median(np.max(scores_ML_high, axis=1)))
    print(np.median(scores_ML_high[:, -1]), np.mean(scores_ML_high[:, -1]))'''

'''# AutoML best
path = '/p/lustre2/shiduan/AutoML-predictions-smooth-best/'
if not os.path.exists('dataResult/scores_ML_smooth_best.npy'):
    scores_ML_smooth_best = np.zeros((25, 6))
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
            scores_ML_smooth_best[i, k] = r2
    print(np.max(scores_ML_smooth_best, axis=1)) # 25 stations. 
    print('ML-best', np.median(np.max(scores_ML_smooth_best, axis=1)))
    print('ML', np.median(scores_ML_smooth_best[:, -2]), np.mean(scores_ML_smooth_best[:, -2]))
    np.save('dataResult/scores_ML_smooth_best', scores_ML_smooth_best)
else:
    scores_ML_smooth_best = np.load('dataResult/scores_ML_smooth_best.npy')
    print('ML-best', np.median(np.max(scores_ML_smooth_best, axis=1)))
    print(np.median(scores_ML_smooth_best[:, -1]), np.mean(scores_ML_smooth_best[:, -1]))
'''
path = '/p/lustre2/shiduan/Lasso-predictions-smooth/'
if not os.path.exists('dataResult/scores_LA_smooth.npy') or reload:
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
if not os.path.exists('dataResult/scores_RD_smooth.npy') or reload:
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
if not os.path.exists('dataResult/scores_LR_smooth.npy') or reload:
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
    print('LR', np.median(np.median(scores_LR[:, -1])))

'''path = '/p/lustre2/shiduan/AutoML-predictions-smooth-LR/'
if not os.path.exists('dataResult/scores_ML_smooth_LR.npy') or reload:
    scores_ML = np.zeros((25, 6))
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
            scores_ML[i, k] = r2
    print(np.max(scores_ML, axis=1)) # 25 stations. 
    print('ML', np.median(np.max(scores_ML, axis=1)))
    print('ML', np.median(scores_ML[:, -2]), np.mean(scores_ML[:, -2]))
    np.save('dataResult/scores_ML_smooth_LR', scores_ML)
else:
    scores_ML = np.load('dataResult/scores_ML_smooth_LR.npy')
    print('ML-medium', np.median(np.max(scores_ML, axis=1)))
    print(np.median(scores_ML[:, -1]), np.mean(scores_ML[:, -1]))'''

###
### Lag3
###
path = '/p/lustre2/shiduan/LOD-predictions-smooth/'
if not os.path.exists('dataResult/scores_LOD_smooth_lag3.npy') or reload:
    scores_LOD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred_lag3.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LOD[i, k] = r2
    print(np.max(scores_LOD, axis=1)) # 25 stations. 
    print('LOD', np.median(np.max(scores_LOD, axis=1)))
    print('LOD', np.median(scores_LOD[:, -2]), np.mean(scores_LOD[:, -2]))
    np.save('dataResult/scores_LOD_smooth_lag3', scores_LOD)
else:
    scores_LOD = np.load('dataResult/scores_LOD_smooth_lag3.npy')

path = '/p/lustre2/shiduan/Lasso-predictions-smooth/'
if not os.path.exists('dataResult/scores_LA_smooth_lag3.npy') or reload:
    scores_LA = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred_lag3.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LA[i, k] = r2
    print(np.max(scores_LA, axis=1)) # 25 stations. 
    print('Lasso', np.median(np.max(scores_LA, axis=1)))
    print('Lasso', np.median(scores_LA[:, -2]), np.mean(scores_LA[:, -2]))
    np.save('dataResult/scores_LA_smooth_lag3', scores_LA)
else:
    scores_LA = np.load('dataResult/scores_LA_smooth_lag3.npy')

path = '/p/lustre2/shiduan/Ridge-predictions-smooth/'
if not os.path.exists('dataResult/scores_RD_smooth_lag3.npy') or reload:
    scores_RD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred_lag3.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_RD[i, k] = r2
    print(np.max(scores_RD, axis=1)) # 25 stations. 
    print('Ridge', np.median(np.max(scores_RD, axis=1)))
    print('Ridge', np.median(scores_RD[:, -2]), np.mean(scores_RD[:, -2]))
    np.save('dataResult/scores_RD_smooth_lag3', scores_RD)
else:
    scores_RD = np.load('dataResult/scores_RD_smooth_lag3.npy')

path = '/p/lustre2/shiduan/Linear-predictions-smooth/'
if not os.path.exists('dataResult/scores_LR_smooth_lag3.npy') or reload:
    scores_LR = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred_lag3.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_LR[i, k] = r2
    print(np.max(scores_LR, axis=1)) # 25 stations. 
    print('LR', np.median(np.max(scores_LR, axis=1)))
    print('LR', np.median(scores_LR[:, -2]), np.mean(scores_LR[:, -2]))
    np.save('dataResult/scores_LR_smooth_lag3', scores_LR)
else:
    scores_LR = np.load('dataResult/scores_LR_smooth_lag3.npy')

path = '/p/lustre2/shiduan/AutoML-predictions-smooth/'
if not os.path.exists('dataResult/scores_ML_smooth_lag3.npy') or reload:
    scores_ML = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred_lag3.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_ML[i, k] = r2
    print(np.max(scores_ML, axis=1)) # 25 stations. 
    print('ML', np.median(np.max(scores_ML, axis=1)))
    print('ML', np.median(scores_ML[:, -2]), np.mean(scores_ML[:, -2]))
    np.save('dataResult/scores_ML_smooth_lag3', scores_ML)
else:
    scores_ML = np.load('dataResult/scores_ML_smooth_lag3.npy')


'''path = '/p/lustre2/shiduan/AutoML-predictions-smooth-LR/'
if not os.path.exists('dataResult/scores_ML_smooth_LR_lag3.npy') or reload:
    scores_ML = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            reals = []
            preds = []
            for seed in range(6):
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-pred_lag3.npy')
                reals.append(real.reshape(-1, 1))
                preds.append(pred.reshape(-1, 1))
            reals = np.concatenate(reals)
            preds = np.concatenate(preds)
            r2 = r2_score(reals, preds)
            scores_ML[i, k] = r2
    print(np.max(scores_ML, axis=1)) # 25 stations. 
    print('ML', np.median(np.max(scores_ML, axis=1)))
    print('ML', np.median(scores_ML[:, -2]), np.mean(scores_ML[:, -2]))
    np.save('dataResult/scores_ML_smooth_LR_lag3', scores_ML)
else:
    scores_ML = np.load('dataResult/scores_ML_smooth_LR_lag3.npy')'''






