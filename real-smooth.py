import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from tools import build_df, get_peak_month
from iteration import get_seasonal_data, get_prediction

import pickle
import argparse
from matplotlib import pyplot as plt
from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eof', type=int)
    args = vars(parser.parse_args())
    return args

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total.
station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]

args = get_args()
eof = args['eof']
lag = 0
mode_smooth = True
output = 'EOF: ' + str(eof) + ' lag: ' + str(lag) 
print(output)
fig = plt.figure(figsize=(12, 3))

model_path = '/p/lustre2/shiduan/AutogluonModels-smooth-LR/ag-'
# model_path = '/p/lustre2/shiduan/AutogluonModels-smooth/ag-'
hist_co2 = pd.read_csv('historical_co2.csv', index_col=['wy', 'year', 'month'])


r2_all_ml = []
r2_all_lasso = []
r2_all_ridge = []
r2_all_linear = []
r2_all_lod = []

acc_all_ml = []
acc_all_lasso = []
acc_all_ridge = []
acc_all_linear = []
acc_all_lod = []

for ind, station in enumerate(station_ids):
    real_hist_dfs = []
    path = 'Reanalysis-csv/hist_q_csv_monthly_usgs.csv'
    real_q_df = pd.read_csv(path, index_col=['wy', 'year', 'month'])
    real_q_df[real_q_df<0]=0
    real_q_df = real_q_df.sort_index(level=0)
    # hist_modes_df = pd.read_csv('IPSL-Modes-csv-high/r'+str(member)+'-hist_modes_csv_monthly.csv', index_col=['wy', 'year', 'month'])
    real_modes_df = pd.read_csv('Reanalysis-csv-CBF/hist_modes_csv_monthly.csv', index_col=['wy', 'year', 'month'])
    real_modes_df['modesWY']=real_modes_df.index
    real_modes_df_co2 = pd.concat((real_modes_df, (hist_co2)), axis=1)
    real_modes_df_co2 = real_modes_df_co2.sort_index(level=0)
    start_wy = 1980
    if ind not in [0, 1, 19]:
        start_wy = 1979
    real_df = build_df(station_ids, real_q_df, 0, 12, 
                real_modes_df_co2, start_wy, 7, 2014, 12, norm=False)
    real_df['lat'] = pd.to_numeric(real_df['lat'])
    real_df['lon'] = pd.to_numeric(real_df['lon'])
    real_df['ele'] = pd.to_numeric(real_df['ele'])
    _, peak = get_peak_month(station, real_df=real_df)
    print(ind, station, peak)
    if peak==1:
        months = [12, peak, peak+1]
    elif peak==12:
        months = [peak-1, peak, 1]
    else:
        months = [peak-1, peak, peak+1]
    peak_2 = station_peaks[ind]
    if peak_2>3 and peak_2<12:
        lag3=True
    else:
        lag3=False
    print(peak_2, peak)
    predictor = ['PDO_eof', 'AMO_eof', 'PNA_eof', 
                     'NAM_eof', 'NAO_eof', 'SAM_eof']
    predictor_high = []
    for i in range(1, eof+1):
        for p in predictor:
            if lag3:
                predictor_high.append(p+'_'+str(i)+'_lag3')
            else:
                predictor_high.append(p+'_'+str(i))
    if lag3:
        predictor_high.append('nino34_lag3')
        predictor_high.append('co2_lag3')
    else:
        predictor_high.append('nino34')
        predictor_high.append('co2')
    aux = []
    if lag>0:
        for k in range(1, lag+1):
            predictor_aux = [m+'_lag'+str(k) for m in predictor_high]
            aux = aux+predictor_aux
    all_predictor = predictor_high+aux
    print(len(all_predictor), all_predictor[0])
    ml_predictor = all_predictor.copy()
    ml_predictor.append('Q_sim') # for AutoGluon

    mam_dfs_real_hist = get_seasonal_data([real_df], months=months, smooth_mode=mode_smooth)
    test_dfs = []
    for df in mam_dfs_real_hist:
        df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
        test_dfs.append(df)
    test_dfs = test_dfs[0] # only reanalysis dataframe. 
    station_test_dfs = test_dfs.xs(station, level=1)
    # autoML
    test_x = station_test_dfs[ml_predictor]
    true_y = test_x['Q_sim'].values.reshape(-1, 1)
    pred_y = []
    for seed in range(0, 6):
        if lag3:
            path = model_path+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'-lag3'
        else:
            path = model_path+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)
        model = TabularPredictor.load(path, require_py_version_match=False)
        y_pred = model.predict(test_x)
        pred_y.append(y_pred.values.reshape(-1, 1))
    pred_y_all = np.concatenate(pred_y, axis=-1)
    pred_y_all = np.mean(pred_y_all, axis=-1)
    r2 = r2_score(true_y, pred_y_all.reshape(-1, 1))
    acc = accuracy_score(true_y>0, pred_y_all.reshape(-1, 1)>0)
    print(station, ' AutoML: ', r2, ' peak: ', peak)
    print('MSE: ', mean_squared_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' MAE: ', mean_absolute_error(true_y, pred_y_all.reshape(-1, 1)),
          ' ACC: ', acc) 
    acc_all_ml.append(acc)
    r2_all_ml.append(r2)
    if station=='11528700':
        plt.plot(pred_y_all.reshape(-1, 1), label='AutoML', color='tab:green')
        for y in pred_y:
            plt.plot(y.reshape(-1, 1), color='tab:green', alpha=.3)
    # LOD
    pred_y = []
    for seed in range(0, 6):
        if lag3:
            path = '/p/lustre2/shiduan/LOD-smooth/model-records-EOF-'+str(eof)+'-lag-'+str(lag)+'-lag3-seed-'+str(seed)
        else:
            path = '/p/lustre2/shiduan/LOD-smooth/model-records-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)
        with open(path, 'rb') as pfile:
            records_all = pickle.load(pfile)
        record = records_all[ind]
        all_prediction, predictors = get_prediction(records=record, station_df=test_x)
        pred_y.append(all_prediction.reshape(-1, 1))
    pred_y_all = np.concatenate(pred_y, axis=-1)
    pred_y_all = np.mean(pred_y_all, axis=-1)
    r2 = r2_score(true_y, pred_y_all.reshape(-1, 1))
    acc = accuracy_score(true_y>0, pred_y_all.reshape(-1, 1)>0)
    print(station, ' LOD: ', r2)
    print('MSE: ', mean_squared_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' MAE: ', mean_absolute_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' ACC: ', acc)
    r2_all_lod.append(r2)
    acc_all_lod.append(acc)
    if station=='11528700':
        plt.plot(pred_y_all.reshape(-1, 1), label='LOD', color='tab:purple')
        for y in pred_y:
            plt.plot(y.reshape(-1, 1), color='tab:purple', alpha=.3)
    # Lasso
    test_x = station_test_dfs[all_predictor].copy()
    co2_predictors = []
    for predct in all_predictor:
        if 'co2' in predct:
            co2_predictors.append(predct)
    test_x.loc[:, co2_predictors] = test_x.loc[:, co2_predictors].div(300)
    pred_y = []
    for seed in range(0, 6):
        if lag3:
            path = '/p/lustre2/shiduan/Lasso-smooth/LS-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'_lag3'
        else:
            path = '/p/lustre2/shiduan/Lasso-smooth/LS-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)
        with open(path, 'rb') as pfile:
            model = pickle.load(pfile)
        y_pred = model.predict(test_x)
        pred_y.append(y_pred.reshape(-1, 1))
    pred_y_all = np.concatenate(pred_y, axis=-1)
    pred_y_all = np.mean(pred_y_all, axis=-1)
    r2 = r2_score(true_y, pred_y_all.reshape(-1, 1))
    r2_all_lasso.append(r2)
    acc = accuracy_score(true_y>0, pred_y_all.reshape(-1, 1)>0)
    acc_all_lasso.append(acc)
    print(station, ' Lasso: ', r2)
    print('MSE: ', mean_squared_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' MAE: ', mean_absolute_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' ACC: ', acc)
    if station=='11528700':
        plt.plot(pred_y_all.reshape(-1, 1), label='Lasso', color='tab:red')
        for y in pred_y:
            plt.plot(y.reshape(-1, 1), color='tab:red', alpha=.3)
    # Ridge
    pred_y = []
    for seed in range(0, 6):
        if lag3:
            path = '/p/lustre2/shiduan/Ridge-smooth/RD-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'_lag3'
        else:
            path = '/p/lustre2/shiduan/Ridge-smooth/RD-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)
        with open(path, 'rb') as pfile:
            model = pickle.load(pfile)
        y_pred = model.predict(test_x)
        pred_y.append(y_pred.reshape(-1, 1))
    pred_y_all = np.concatenate(pred_y, axis=-1)
    pred_y_all = np.mean(pred_y_all, axis=-1)
    r2 = r2_score(true_y, pred_y_all.reshape(-1, 1))
    r2_all_ridge.append(r2)
    acc = accuracy_score(true_y>0, pred_y_all.reshape(-1, 1)>0)
    acc_all_ridge.append(acc)
    print(station, ' Ridge: ', r2)
    print('MSE: ', mean_squared_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' MAE: ', mean_absolute_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' ACC: ', acc)
    if station=='11528700':
        plt.plot(pred_y_all.reshape(-1, 1), label='Ridge', color='blue')
        for y in pred_y:
            plt.plot(y.reshape(-1, 1), color='blue', alpha=.3)
    # Linear
    pred_y = []
    for seed in range(0, 6):
        if lag3:
            path = '/p/lustre2/shiduan/Linear-smooth/LR-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)+'_lag3'
        else:
            path = '/p/lustre2/shiduan/Linear-smooth/LR-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed)
        with open(path, 'rb') as pfile:
            model = pickle.load(pfile)
        y_pred = model.predict(test_x)
        pred_y.append(y_pred.reshape(-1, 1))
    pred_y_all = np.concatenate(pred_y, axis=-1)
    pred_y_all = np.mean(pred_y_all, axis=-1)
    r2 = r2_score(true_y, pred_y_all.reshape(-1, 1))
    r2_all_linear.append(r2)
    acc = accuracy_score(true_y>0, pred_y_all.reshape(-1, 1)>0)
    acc_all_linear.append(acc)
    print(station, ' LR: ', r2)
    print('MSE: ', mean_squared_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' MAE: ', mean_absolute_error(true_y, pred_y_all.reshape(-1, 1)), 
          ' ACC: ', acc)
    print()
    if station=='11528700':
        plt.plot(pred_y_all.reshape(-1, 1), label='LR', color='tab:orange')
        for y in pred_y:
            plt.plot(y.reshape(-1, 1), color='tab:orange', alpha=.3)
        plt.plot(true_y, label='USGS', linewidth=2, color='black')
plt.legend()
plt.savefig('real-11528700-ML.png')
print('AutoML: ', np.median(r2_all_ml), np.mean(r2_all_ml), 
      np.median(acc_all_ml), np.mean(acc_all_ml))
print('LOD: ', np.median(r2_all_lod), np.mean(r2_all_lod), 
      np.median(acc_all_lod), np.mean(acc_all_lod))
print('Ridge: ', np.median(r2_all_ridge), np.mean(r2_all_ridge), 
      np.median(acc_all_ridge), np.mean(acc_all_ridge))
print('Lasso: ', np.median(r2_all_lasso), np.mean(r2_all_lasso), 
      np.median(acc_all_lasso), np.mean(acc_all_lasso))
print('LR: ', np.median(r2_all_linear), np.mean(r2_all_linear), 
      np.median(acc_all_linear), np.mean(acc_all_linear))

