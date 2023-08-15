import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from tools import build_df
from iteration import station_iteration, get_seasonal_data
from sklearn.linear_model import LinearRegression, Lasso
from iteration import get_prediction
from sklearn.model_selection import train_test_split
import pickle
import argparse
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eof', type=int)
    parser.add_argument('--lag', type=int)
    args = vars(parser.parse_args())
    return args

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 

args = get_args()
eof = args['eof']
lag = args['lag']
lag = 0
mode_smooth = True
output = 'EOF: ' + str(eof) + ' lag: ' + str(lag) 
print(output)

model_path = '/p/lustre2/shiduan/AutogluonModels-smooth/ag-'
hist_co2 = pd.read_csv('historical_co2.csv', index_col=['wy', 'year', 'month'])
def get_peak_month(station):
    peaks = []
    for hist_df in [real_df]:
        station_q = hist_df[hist_df['station_id']==station]['Q_sim']
        station_q = station_q.groupby('month').mean()
        peak = station_q.argmax()+1
        peaks.append(peak)
    values, counts = np.unique(peaks, return_counts=True)
    ind = np.argmax(counts)
    return values, values[ind]
def calculate_adj_r2(target, pred, p):
    r2 = r2_score(target, pred)
    n = len(target)
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return r2, adj_r2

real_hist_dfs = []
path = 'Reanalysis-csv/hist_q_csv_monthly.csv'
real_q_df = pd.read_csv(path, index_col=['wy', 'year', 'month'])
real_q_df[real_q_df<0]=0
real_q_df = real_q_df.sort_index(level=0)
# hist_modes_df = pd.read_csv('IPSL-Modes-csv-high/r'+str(member)+'-hist_modes_csv_monthly.csv', index_col=['wy', 'year', 'month'])
real_modes_df = pd.read_csv('Reanalysis-csv-CBF/hist_modes_csv_monthly.csv', index_col=['wy', 'year', 'month'])
real_modes_df['modesWY']=real_modes_df.index
real_modes_df_co2 = pd.concat((real_modes_df, (hist_co2)), axis=1)
real_modes_df_co2 = real_modes_df_co2.sort_index(level=0)
start_wy = 1980
real_df = build_df(station_ids, real_q_df, 0, 12, 
               real_modes_df_co2, 1981, 1, 2014, 12, norm=False)
real_df['lat'] = pd.to_numeric(real_df['lat'])
real_df['lon'] = pd.to_numeric(real_df['lon'])
real_df['ele'] = pd.to_numeric(real_df['ele'])

def load_data(model, ensembles, scenario='hist', start_wy=1951):
    hist_dfs = []
    for member in ensembles:
        path = 'data/'+model+'-Streamflow-csv/r'+str(member)+'-'+scenario+'_q_csv_monthly.csv'
        if os.path.exists(path):
            hist_q_df = pd.read_csv(path, index_col=['wy', 'year', 'month'])
            hist_q_df[hist_q_df<0]=0
            hist_modes_df = pd.read_csv('data/'+model+'-Modes-csv-CBF-NewOBS/r'+str(member)+'-'+scenario+'_modes_csv_monthly.csv', index_col=['wy', 'year', 'month'])
            hist_modes_df['modesWY']=hist_modes_df.index
            hist_modes_df_co2 = pd.concat((hist_modes_df, (hist_co2)), axis=1)
            hist_modes_df_co2 = hist_modes_df_co2.sort_index(level=0)
            start_wy = np.max([hist_q_df.index.get_level_values(0)[0], hist_modes_df.index.get_level_values(0)[0]])
            hist_df = build_df(station_ids, hist_q_df, 0, 12, 
                           hist_modes_df_co2, start_wy+1, 1, 2013, 1, norm=False)
            hist_df['lat'] = pd.to_numeric(hist_df['lat'])
            hist_df['lon'] = pd.to_numeric(hist_df['lon'])
            hist_df['ele'] = pd.to_numeric(hist_df['ele'])
            hist_dfs.append(hist_df)
    return hist_dfs

IPSL_amip_dfs = load_data('IPSL-CM6A-LR', ensembles=range(1, 11), scenario='amip')
print(len(IPSL_amip_dfs))

predictor = ['PDO_eof', 'AMO_eof', 'PNA_eof', 
                     'NAM_eof', 'NAO_eof', 'SAM_eof']
predictor_high = []
for i in range(1, eof+1):
    for p in predictor:
        predictor_high.append(p+'_'+str(i))
# print(predictor_high)
predictor_high.append('nino34')
predictor_high.append('co2')
aux = []
if lag>0:
    for k in range(1, lag+1):
        predictor_aux = [m+'_lag'+str(k) for m in predictor_high]
        aux = aux+predictor_aux
all_predictor = predictor_high+aux
print(len(all_predictor))
print(len(all_predictor))
ml_predictor = all_predictor.copy()
ml_predictor.append('Q_sim') # for AutoGluon

r2_all_ml = []
r2_all_lasso = []
r2_all_ridge = []
r2_all_linear = []
r2_all_lod = []
for ind, station in enumerate(station_ids):
    _, peak = get_peak_month(station)
    # print(station, peak)
    if peak==1:
        months = [12, peak, peak+1]
    elif peak==12:
        months = [peak-1, peak, 1]
    else:
        months = [peak-1, peak, peak+1]
    mam_dfs_amip_hist = get_seasonal_data(IPSL_amip_dfs, months=months, smooth_mode=mode_smooth)
    test_dfs = []
    for df in mam_dfs_amip_hist:
        df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
        test_dfs.append(df)
    # test_dfs = test_dfs[0] # only reanalysis dataframe. 
    test_dfs = pd.concat(test_dfs)
    station_test_dfs = test_dfs.xs(station, level=1)
    # autoML
    test_x = station_test_dfs[ml_predictor]
    true_y = test_x['Q_sim'].values.reshape(-1, 1)
    pred_y = []
    for seed in range(0, 6):
        model = TabularPredictor(label='Q_sim').load(
            model_path+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed), require_py_version_match=False)
        y_pred = model.predict(test_x)
        pred_y.append(y_pred.values.reshape(-1, 1))
    pred_y = np.concatenate(pred_y, axis=-1)
    pred_y = np.mean(pred_y, axis=-1)
    r2 = r2_score(true_y, pred_y.reshape(-1, 1))
    print(station, ' AutoML: ', r2, ' peak: ', peak)
    r2_all_ml.append(r2)
    # LOD
    pred_y = []
    for seed in range(0, 6):
        with open('/p/lustre2/shiduan/LOD-smooth/model-records-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed), 'rb') as pfile:
            records_all = pickle.load(pfile)
        record = records_all[ind]
        all_prediction, predictors = get_prediction(records=record, station_df=test_x)
        pred_y.append(all_prediction.reshape(-1, 1))
    pred_y = np.concatenate(pred_y, axis=-1)
    pred_y = np.mean(pred_y, axis=-1)
    r2 = r2_score(true_y, pred_y.reshape(-1, 1))
    print(station, ' LOD: ', r2)
    r2_all_lod.append(r2)
    # Lasso
    test_x = station_test_dfs[all_predictor].copy()
    co2_predictors = []
    for predct in all_predictor:
        if 'co2' in predct:
            co2_predictors.append(predct)
    # print(test_x.loc[:, co2_predictors])
    test_x.loc[:, co2_predictors] = test_x.loc[:, co2_predictors].div(300)
    pred_y = []
    for seed in range(0, 6):
        with open('/p/lustre2/shiduan/Lasso-smooth/LS-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed), 'rb') as pfile:
            model = pickle.load(pfile)
        y_pred = model.predict(test_x)
        pred_y.append(y_pred.reshape(-1, 1))
    pred_y = np.concatenate(pred_y, axis=-1)
    pred_y = np.mean(pred_y, axis=-1)
    r2 = r2_score(true_y, pred_y.reshape(-1, 1))
    r2_all_lasso.append(r2)
    print(station, ' Lasso: ', r2)
    # Ridge
    pred_y = []
    for seed in range(0, 6):
        with open('/p/lustre2/shiduan/Ridge-smooth/RD-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed), 'rb') as pfile:
            model = pickle.load(pfile)
        y_pred = model.predict(test_x)
        pred_y.append(y_pred.reshape(-1, 1))
    pred_y = np.concatenate(pred_y, axis=-1)
    pred_y = np.mean(pred_y, axis=-1)
    r2 = r2_score(true_y, pred_y.reshape(-1, 1))
    r2_all_ridge.append(r2)
    print(station, ' Ridge: ', r2)
    # Linear
    pred_y = []
    for seed in range(0, 6):
        with open('/p/lustre2/shiduan/Linear-smooth/LR-'+str(station)+'-EOF-'+str(eof)+'-lag-'+str(lag)+'-seed-'+str(seed), 'rb') as pfile:
            model = pickle.load(pfile)
        y_pred = model.predict(test_x)
        pred_y.append(y_pred.reshape(-1, 1))
    pred_y = np.concatenate(pred_y, axis=-1)
    pred_y = np.mean(pred_y, axis=-1)
    r2 = r2_score(true_y, pred_y.reshape(-1, 1))
    r2_all_linear.append(r2)
    print(station, ' LR: ', r2)
    print()

'''
with open('AutoML/real-data/result-eof-'+str(eof)+'-lag-'+str(lag), 'wb') as pfile:
    pickle.dump(r2_all, pfile)
'''
