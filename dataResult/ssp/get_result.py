import numpy as np
from sklearn.metrics import r2_score
import os

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2] 
reload = True
path = '/p/lustre2/shiduan/LOD-predictions-smooth-ssp/'
if not os.path.exists('scores_LOD_smooth_ssp.npy') or reload:
    scores_LOD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        peak = station_peaks[i]
        if peak>3 and peak<12:
            lag3=True
        else:
            lag3=False
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            seed=42
            if lag3:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3.npy')
            else:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred.npy')
            r2 = r2_score(real, pred)
            scores_LOD[i, k] = r2
    np.save('scores_LOD_smooth_ssp', scores_LOD)
else:
    scores_LOD = np.load('scores_LOD_smooth_ssp.npy')

path = '/p/lustre2/shiduan/LR-predictions-smooth-ssp/'
if not os.path.exists('scores_LR_smooth_ssp.npy') or reload:
    scores_LR = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        peak = station_peaks[i]
        if peak>3 and peak<12:
            lag3=True
        else:
            lag3=False
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            seed=42
            if lag3:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3.npy')
            else:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred.npy')
            r2 = r2_score(real, pred)
            scores_LR[i, k] = r2
    np.save('scores_LR_smooth_ssp', scores_LR)
else:
    scores_LR = np.load('scores_LR_smooth_ssp.npy')


path = '/p/lustre2/shiduan/LASSO-predictions-smooth-ssp/'
if not os.path.exists('scores_LA_smooth_ssp.npy') or reload:
    scores_LA = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        peak = station_peaks[i]
        if peak>3 and peak<12:
            lag3=True
        else:
            lag3=False
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            seed=42
            if lag3:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3.npy')
            else:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred.npy')
            r2 = r2_score(real, pred)
            scores_LA[i, k] = r2
    np.save('scores_LA_smooth_ssp', scores_LA)
else:
    scores_LA = np.load('scores_LA_smooth_ssp.npy')

path = '/p/lustre2/shiduan/RIDGE-predictions-smooth-ssp/'
if not os.path.exists('scores_RD_smooth_ssp.npy') or reload:
    scores_RD = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        peak = station_peaks[i]
        if peak>3 and peak<12:
            lag3=True
        else:
            lag3=False
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            seed=42
            if lag3:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3.npy')
            else:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred.npy')
            r2 = r2_score(real, pred)
            scores_RD[i, k] = r2
    np.save('scores_RD_smooth_ssp', scores_RD)
else:
    scores_RD = np.load('scores_RD_smooth_ssp.npy')

path = '/p/lustre2/shiduan/AUTOML-predictions-smooth-ssp/'
if not os.path.exists('scores_AutoML_smooth_ssp.npy') or reload:
    scores_ML = np.zeros((25, 6))
    for i, station in enumerate(station_ids):
        peak = station_peaks[i]
        if peak>3 and peak<12:
            lag3=True
        else:
            lag3=False
        lag = 0
        for k, eof in enumerate(range(1, 7)):
            seed=42
            if lag3:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real_lag3.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred_lag3.npy')
            else:
                real = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-real.npy')
                pred = np.load(
                    path+station+'/'+station+'-EOF-'+str(eof)+'-seed-'+str(seed)+'-pred.npy')
            r2 = r2_score(real, pred)
            scores_ML[i, k] = r2
    np.save('scores_AutoML_smooth_ssp', scores_ML)
else:
    scores_ML = np.load('scores_AutoML_smooth_ssp.npy')

# include
