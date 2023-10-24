import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sys
import multiprocessing
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(os.getcwd()))
from tools import build_df, load_data, get_peak_month
from iteration import station_iteration, get_seasonal_data
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from iteration import get_prediction
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from ag_linear import CustomLRModel
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
# add linear model
custom_hyperparameters = get_hyperparameter_config('default')
custom_hyperparameters[CustomLRModel] = {}


import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--eof', type=int)
    parser.add_argument('--pre_season', type=int, default=0)
    parser.add_argument('--model_type', 
                        choices=['LR', 'Lasso', 'Ridge', 'AutoML', 'LOD', 'AutoLR', 'PLS'])
    args = vars(parser.parse_args())
    return args

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
station_peaks = [5, 5, 3, 3, 2, 
                 2, 3, 6, 5, 5, 
                 5, 2, 3, 2, 2, 
                 3, 1, 1, 1, 2, 
                 12, 1, 3, 5, 2]
args = get_args()
eof = args['eof']
smooth = args['smooth']
if smooth==0:
    mode_smooth=False
else:
    mode_smooth=True
pre_season = args['pre_season']
if pre_season==0:
    lag3 = False
else:
    lag3 = True
model_type = args['model_type']
hist_co2 = pd.read_csv('historical_co2.csv', index_col=['wy', 'year', 'month'])
hist_co2 = hist_co2/300

IPSL_hist_dfs = load_data('IPSL-CM6A-LR', ensembles=range(1, 11), co2=hist_co2)
print(len(IPSL_hist_dfs))

EC_hist_dfs = load_data('EC-Earth3', ensembles=[13, 15, 16, 11, 25, 24, 23, 22, 21, 1, 4, 5, 6, 7, # 9, 10, 
                                                ], 
                        co2=hist_co2)
print(len(EC_hist_dfs))

ACCESS_hist_dfs = load_data('ACCESS-ESM1-5', ensembles=range(1, 11), co2=hist_co2)
print(len(ACCESS_hist_dfs))

MIROC_hist_dfs = load_data('MIROC6', ensembles=range(1, 11), co2=hist_co2)
print(len(MIROC_hist_dfs))

MPI_hist_dfs = load_data('MPI-ESM1-2-LR', ensembles=range(1, 11), co2=hist_co2)
print(len(MIROC_hist_dfs))

CNRM_hist_dfs = load_data('CNRM-ESM2-1', ensembles=range(1, 11), co2=hist_co2)
print(len(CNRM_hist_dfs))

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

def find_alpha(train_x, train_y, val_x, val_y):
    r2 = -100
    alpha_optim = 0
    if model_type.upper()=='RIDGE':
        alpha_list = [5, 1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
        ml_model = Ridge
    elif model_type.upper()=='LASSO':
        alpha_list = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-5, 5e-5]
        ml_model = Lasso
    for alpha in alpha_list:
        model = ml_model(alpha=alpha, max_iter=10000).fit(train_x, train_y)
        score = model.score(val_x, val_y)
        if score>r2:
            r2 = score
            alpha_optim = alpha
    return alpha_optim, r2

def find_components(train_x, train_y, val_x, val_y):
    r2 = -100
    n_optim = 0
    for n_components in range(1, 12):
        model = PLSRegression(n_components=n_components, max_iter=10000).fit(train_x, train_y)
        score = model.score(val_x, val_y)
        if score>r2:
            r2 = score
            n_optim = n_components
    return n_optim, r2

def run(args):
    test_gcm, eof_modes, random_seed, station_id = args
    print('test: ', test_gcm, ' eof: ', eof_modes)
    r2s_ens = []
    train_all_dfs = []
    val_all_dfs = []
    test_all_dfs = []
    train_y_dfs = []
    val_y_dfs = []
    test_y_dfs = []
    for station in tqdm(station_ids): # get_all_stations. 
        peak = station_peaks[station_ids.index(station)] # if slice station_ids
        if peak==1:
            months = [12, peak, peak+1]
        elif peak==12:
            months = [peak-1, peak, 1]
        else:
            months = [peak-1, peak, peak+1]
        
        print(station, peak)
        # prepare data. 
        if peak>3 and peak<12:
            lag3=True
        else:
            lag3=False
        predictor = ['PDO_eof', 'AMO_eof', 'PNA_eof', 
                     'NAM_eof', 'NAO_eof', 'SAM_eof']
        predictor_high = []
        for i in range(1, eof_modes+1):
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
        aux = ['lat', 'lon', 'ele']
        all_predictor = predictor_high+aux
        if model_type.upper()=='AUTOML' or model_type.upper()=='AUTOLR':
            all_predictor.append('Q_sim')
        print(len(all_predictor))
        mam_dfs_IPSL_hist = get_seasonal_data(IPSL_hist_dfs, months=months, smooth_mode=mode_smooth)
        mam_dfs_ACCESS_hist = get_seasonal_data(ACCESS_hist_dfs, months=months, smooth_mode=mode_smooth)
        mam_dfs_MIROC_hist = get_seasonal_data(MIROC_hist_dfs, months=months, smooth_mode=mode_smooth)
        mam_dfs_MPI_hist = get_seasonal_data(MPI_hist_dfs, months=months, smooth_mode=mode_smooth)
        mam_dfs_CNRM_hist = get_seasonal_data(CNRM_hist_dfs, months=months, smooth_mode=mode_smooth)
        mam_dfs_EC_hist = get_seasonal_data(EC_hist_dfs, months=months, smooth_mode=mode_smooth)
        # Iterate through validation dataset. 
        all_dfs = []
        train_dfs = []
        test_dfs = []
        val_dfs = []
        MPI_dfs = []
        CNRM_dfs = []
        IPSL_dfs = []
        EC_dfs = []
        ACCESS_dfs = []
        MIROC_dfs = []
        for df in mam_dfs_EC_hist[:10]:
            df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
            if 'EC' not in test_gcm:
                all_dfs.append(df)
            else:
                test_dfs.append(df)
            EC_dfs.append(df)
        for df in mam_dfs_IPSL_hist[:10]:
            df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
            if 'IPSL' not in test_gcm:
                all_dfs.append(df)
            else:
                test_dfs.append(df)
            IPSL_dfs.append(df)
        for df in mam_dfs_ACCESS_hist:
            df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
            if 'ACCESS' not in test_gcm:
                all_dfs.append(df)
            else:
                test_dfs.append(df)
            ACCESS_dfs.append(df)
        for df in mam_dfs_MIROC_hist:
            df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
            if 'MIROC' not in test_gcm:
                all_dfs.append(df)
            else:
                test_dfs.append(df)
            MIROC_dfs.append(df)
        for df in mam_dfs_MPI_hist:
            df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
            if 'MPI' not in test_gcm:
                all_dfs.append(df)
            else:
                test_dfs.append(df)
            MPI_dfs.append(df)
        for df in mam_dfs_CNRM_hist:
            df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
            if 'CNRM' not in test_gcm:
                all_dfs.append(df)
            else:
                test_dfs.append(df)
            CNRM_dfs.append(df)

        train_dfs, val_dfs = train_test_split(all_dfs, test_size=.3, shuffle=True, random_state=random_seed)
        train_dfs = pd.concat(train_dfs)
        val_dfs = pd.concat(val_dfs)
        test_dfs = pd.concat(test_dfs)

        station_train_dfs = train_dfs.xs(station, level=1)
        station_test_dfs = test_dfs.xs(station, level=1)
        station_val_dfs = val_dfs.xs(station, level=1)
        
        train_x = station_train_dfs[all_predictor]
        val_x = station_val_dfs[all_predictor]
        # print(val_input.index.duplicated(keep=False))
        val_x = val_x.reset_index(drop=True)
        test_x = station_test_dfs[all_predictor]
        train_all_dfs.append(train_x)
        val_all_dfs.append(val_x)
        test_all_dfs.append(test_x)
        train_y_dfs.append(station_train_dfs['Q_sim'])
        val_y_dfs.append(station_val_dfs['Q_sim'])
        test_y_dfs.append(station_test_dfs['Q_sim'])
    train_all = np.concatenate(train_all_dfs, axis=0)
    val_all = np.concatenate(val_all_dfs, axis=0)
    test_all = np.concatenate(test_all_dfs, axis=0)
    train_y_all = pd.concat(train_y_dfs)
    val_y_all = pd.concat(val_y_dfs)
    test_y_all = pd.concat(test_y_dfs)

    print(train_all.shape)
    if model_type.upper()=='PLS':
        n_optim, score = find_components(train_x=train_all, train_y=train_y_all.values,
                                    val_x=val_all, val_y=val_y_all.values)
        print('Components and score: ', n_optim, score)
        model = PLSRegression(n_components=n_optim).fit(train_all, train_y_all.values)
        y_pred = model.predict(test_all)
        y_pred_train = model.predict(train_all)
        path = '/p/lustre2/shiduan/'+model_type.upper()+'-smooth/'
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        file = '/p/lustre2/shiduan/'+model_type.upper()+'-smooth/LS-'+str(station)+'-EOF-'+str(eof)+'-seed-'+str(random_seed)
        with open(file, 'wb') as pfile:
            pickle.dump(model, pfile)
    if model_type.upper()=='RIDGE' or model_type.upper()=='LASSO':
        alpha_optim, score = find_alpha(train_x=train_all, train_y=train_y_all.values,
                                    val_x=val_all, val_y=val_y_all.values)
        print('Alpha and score: ', alpha_optim, score)
        if model_type.upper()=='RIDGE':
            ml_model = Ridge
        elif model_type.upper()=='LASSO':
            ml_model = Lasso
        model = ml_model(alpha=alpha_optim).fit(train_all, train_y_all.values)
        y_pred = model.predict(test_all)
        y_pred_train = model.predict(train_all)
        path = '/p/lustre2/shiduan/'+model_type.upper()+'-smooth/'
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        file = '/p/lustre2/shiduan/'+model_type.upper()+'-smooth/LS-'+str(station)+'-EOF-'+str(eof)+'-seed-'+str(random_seed)
        with open(file, 'wb') as pfile:
            pickle.dump(model, pfile)
    if model_type.upper()=='AUTOML':
        train_input = station_train_dfs[all_predictor]
        val_input = station_val_dfs[all_predictor]
        val_input = val_input.reset_index(drop=True)
        test_input = station_test_dfs[all_predictor]
        if lag3:
            path = '/p/lustre2/shiduan/AutogluonModels/'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-lag3'
        else:
            path = '/p/lustre2/shiduan/AutogluonModels/'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)
        model = TabularPredictor(label='Q_sim', verbosity=0, 
        path=path).fit(
        train_data=train_input, tuning_data=val_input)
        y_pred = model.predict(test_input).values
        y_pred_train = model.predict(train_input).values
    if model_type.upper()=='AUTOLR':
        train_input = station_train_dfs[all_predictor]
        val_input = station_val_dfs[all_predictor]
        val_input = val_input.reset_index(drop=True)
        test_input = station_test_dfs[all_predictor]
        if lag3:
            path = '/p/lustre2/shiduan/AutogluonModels-LR/'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-lag3'
        else:
            path = '/p/lustre2/shiduan/AutogluonModels-LR/'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)
        model = TabularPredictor(label='Q_sim', verbosity=0, 
        path=path).fit(
        train_data=train_input, tuning_data=val_input, hyperparameters=custom_hyperparameters)
        y_pred = model.predict(test_input).values
        y_pred_train = model.predict(train_input).values
    if model_type.upper()=='LR':
        model = LinearRegression().fit(train_all, train_y_all.values)
        path = '/p/lustre2/shiduan/'+model_type.upper()+'-smooth/'
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        file = '/p/lustre2/shiduan/'+model_type.upper()+'-smooth/LS-'+str(station)+'-EOF-'+str(eof)+'-seed-'+str(random_seed)
        with open(file, 'wb') as pfile:
            pickle.dump(model, pfile)
        y_pred = model.predict(test_all)
        y_pred_train = model.predict(train_all)
    if model_type.upper()=='LOD':
        train_dfs_df = pd.DataFrame(np.concatenate(train_all, train_y_all.values), columns=all_predictor+['Q_sim'])
        test_dfs_df = pd.DataFrame(np.concatenate(test_all, test_y_all.values), columns=all_predictor+['Q_sim'])
        val_dfs_df = pd.DataFrame(np.concatenate(val_all, val_y_all.values), columns=all_predictor+['Q_sim'])
        
        r2, records, target_pred, last_pred_test, train_pred = station_iteration(station_train_dfs=train_dfs_df, all_predictor=all_predictor,
                                    station_test_dfs=test_dfs_df, station_val_dfs=val_dfs_df, norm=False, plot=False)
        file = '/p/lustre2/shiduan/LOD-smooth/'+station+'/model-records-EOF-'+str(eof)+'-seed-'+str(random_seed)
        path = '/p/lustre2/shiduan/LOD-smooth/'+station+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(file+'-seed-'+str(random_seed), 'wb') as pfile:
            pickle.dump(records, pfile)
        y_pred_train = train_pred
        y_pred = last_pred_test
    r2 = r2_score(test_y_all.values.reshape(-1, 1), y_pred.reshape(-1, 1))
    r2s_ens.append(r2)
    print('R2: ', r2)
    if mode_smooth:
        path = '/p/lustre2/shiduan/'+model_type.upper()+'-predictions-smooth/'+str(station)+'/'
    else:
        path = '/p/lustre2/shiduan/'+model_type.upper()+'-predictions/'+str(station)+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    # Save test predictions. 
    file = path+station+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-real.npy'
    np.save(file, 
            test_y_all.values.reshape(-1, 1))
    file = path+station+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-pred.npy'
    np.save(file, 
            y_pred.reshape(-1, 1))
    # Save train predictions
    file = path+station+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-real_train.npy'
    np.save(file, 
            train_y_all.values.reshape(-1, 1))
    file = path+station+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-pred_train.npy'
    np.save(file, 
            y_pred_train.reshape(-1, 1))
    return r2s_ens

station = 0   
test_gcms = ['IPSL', 'EC', 'ACCESS', 'MPI', 'MIROC', 'CNRM']
random_seeds = [0, 1, 2, 3, 4, 5]

pool = multiprocessing.Pool(6)
args_list = [([gcm], eof, seed, station) 
                for gcm, seed in zip(test_gcms, random_seeds)]

results = pool.map(run, args_list)
pool.close()
pool.join()
'''r2s_ens = run(eof_modes=eof, test_gcm=['IPSL'], random_seed=0, station_id=station)
r2s_ens = run(eof_modes=eof, test_gcm=['EC'], random_seed=1, station_id=station)
r2s_ens = run(eof_modes=eof, test_gcm=['ACCESS'], random_seed=2, station_id=station)
r2s_ens = run(eof_modes=eof, test_gcm=['MPI'], random_seed=3, station_id=station)
r2s_ens = run(eof_modes=eof, test_gcm=['MIROC'], random_seed=4, station_id=station)
r2s_ens = run(eof_modes=eof, test_gcm=['CNRM'], random_seed=5, station_id=station)'''

print('Done')
