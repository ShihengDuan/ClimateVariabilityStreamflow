import pandas as pd
import numpy as np
import pickle
import multiprocessing

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sys
import os
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
    parser.add_argument('--model_type', 
                        choices=['LR', 'Lasso', 'Ridge', 'AutoML', 'LOD', 'AutoLR', 'PLS'])
    parser.add_argument('--level', type=int) # how many variables are kept initially. 
    args = vars(parser.parse_args())
    return args

station_ids = ['10336645', '10336660', '11124500', '11141280', '11143000', 
               '11148900', '11151300', '11230500', '11237500', '11264500', 
               '11266500', '11284400', '11381500', '11451100', '11468500', 
               '11473900', '11475560', '11476600', '11478500', '11480390', 
               '11481200', '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
station_peaks = [5, 5, 3, 3, 2, 
                 2, 3, 6, 5, 5, 
                 5, 2, 3, 2, 2, 
                 3, 1, 1, 1, 2, 
                 12, 1, 3, 5, 2]
mods = ['PNA_eof_1', 'PNA_eof_2', 'PNA_eof_3', 'PNA_eof_4', 'PNA_eof_5', 'PNA_eof_6', 
        'PDO_eof_1', 'PDO_eof_2', 'PDO_eof_3', 'PDO_eof_4', 'PDO_eof_5', 'PDO_eof_6', 
        'AMO_eof_1', 'AMO_eof_2', 'AMO_eof_3', 'AMO_eof_4', 'AMO_eof_5', 'AMO_eof_6', 
        'NAM_eof_1', 'NAM_eof_2', 'NAM_eof_3', 'NAM_eof_4', 'NAM_eof_5', 'NAM_eof_6', 
        'SAM_eof_1', 'SAM_eof_2', 'SAM_eof_3', 'SAM_eof_4', 'SAM_eof_5', 'SAM_eof_6', 
        'NAO_eof_1', 'NAO_eof_2', 'NAO_eof_3', 'NAO_eof_4', 'NAO_eof_5', 'NAO_eof_6', 
        'co2', 'nino34']
args = get_args()
smooth = args['smooth']
eof_modes=6
if smooth==0:
    mode_smooth=False
else:
    mode_smooth=True
level = args['level']
print('level: ', level)
model_type = args['model_type']
hist_co2 = pd.read_csv('../historical_co2.csv', index_col=['wy', 'year', 'month'])

hist_co2 = hist_co2/300

IPSL_hist_dfs = load_data('IPSL-CM6A-LR', ensembles=range(1, 11), co2=hist_co2)
print(len(IPSL_hist_dfs))

EC_hist_dfs = load_data('EC-Earth3', ensembles=[13, 15, 16, 11, 25, 24, 23, 22, 21, 1, 4, 5, 6, 7, 9, 10, ], 
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
    n_features = train_x.shape[1]
    for n_components in range(1, n_features+1):
        model = PLSRegression(n_components=n_components, max_iter=10000).fit(train_x, train_y)
        score = model.score(val_x, val_y)
        if score>r2:
            r2 = score
            n_optim = n_components
    return n_optim, r2

# def run(time_lag, test_gcm, eof_modes, random_seed, modes_keep):
def run(args):
    time_lag, test_gcm, eof_modes, random_seed, modes_keep = args
    # print('Time: ', time_lag, ' test: ', test_gcm)
    r2s_ens = []
    station_pred = {}
    station_real = {}
    for ind, station in enumerate(station_ids[:]): # train iteration. 
        # peak = station_peaks[ind]
        peak = station_peaks[station_ids.index(station)] # if slice station_ids
        print(station, peak)
        # prepare data. 
        if peak>3 and peak<12:
            lag3=True
        else:
            lag3=False
        predictor_high = []
        if modes_keep is not None:
            if isinstance(modes_keep, list):
                keep_name = ''
                for p in modes_keep:
                    keep_name = keep_name+p
                print('Keep_name: ', keep_name)
                if lag3:
                    for p in modes_keep:
                        predictor_high.append(p+'_lag3')
                else:
                    for p in modes_keep:
                        predictor_high.append(p)
            else:
                keep_name = modes_keep
                print('Keep_name: ', keep_name)
                if lag3:
                    predictor_high.append(modes_keep+'_lag3')
                else:
                    predictor_high.append(modes_keep)
        aux = []
        if time_lag>0:
            for k in range(1, time_lag+1):
                predictor_aux = [m+'_lag'+str(k) for m in predictor_high]
                aux = aux+predictor_aux
        all_predictor = predictor_high+aux
        aux = []
        if time_lag>0:
            for k in range(1, time_lag+1):
                predictor_aux = [m+'_lag'+str(k) for m in predictor_high]
                aux = aux+predictor_aux
        all_predictor = predictor_high+aux
        if model_type.upper()=='AUTOML' or model_type.upper()=='AUTOLR':
            all_predictor.append('Q_sim')
        print(len(all_predictor), ' predictors: ', all_predictor)
        if peak==1:
            months = [12, peak, peak+1]
        elif peak==12:
            months = [peak-1, peak, 1]
        else:
            months = [peak-1, peak, peak+1]
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
        val_x = val_x.reset_index(drop=True)
        test_x = station_test_dfs[all_predictor]

        if model_type.upper()=='RIDGE' or model_type.upper()=='LASSO':
            alpha_optim, score = find_alpha(train_x=train_x, train_y=station_train_dfs['Q_sim'],
                                            val_x=val_x, val_y=station_val_dfs['Q_sim'])
            print('Alpha and score: ', alpha_optim, score)
            if model_type.upper()=='RIDGE':
                ml_model = Ridge
            elif model_type.upper()=='LASSO':
                ml_model = Lasso
            model = ml_model(alpha=alpha_optim).fit(train_x, station_train_dfs['Q_sim'])
            y_pred = model.predict(test_x)
        
        if model_type.upper()=='PLS':
            n_optim, score = find_components(train_x=train_x, train_y=station_train_dfs['Q_sim'],
                                            val_x=val_x, val_y=station_val_dfs['Q_sim'])
            print('Components and score: ', n_optim, score)
            model = PLSRegression(n_components=n_optim).fit(train_x, station_train_dfs['Q_sim'])
            y_pred = model.predict(test_x)
        
        if model_type.upper()=='AUTOML':
            train_input = station_train_dfs[all_predictor]
            val_input = station_val_dfs[all_predictor]
            val_input = val_input.reset_index(drop=True)
            test_input = station_test_dfs[all_predictor]
            if lag3:
                path = '/p/lustre2/shiduan/AutogluonModels/'+keep_name+'-include'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-lag-'+str(time_lag)+'-seed-'+str(random_seed)+'-lag3'
            else:
                path = '/p/lustre2/shiduan/AutogluonModels/'+keep_name+'-include'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-lag-'+str(time_lag)+'-seed-'+str(random_seed)
            model = TabularPredictor(label='Q_sim', verbosity=0, 
            path=path).fit(
            train_data=train_input, tuning_data=val_input)
            y_pred = model.predict(test_input).values
        if model_type.upper()=='AUTOLR':
            train_input = station_train_dfs[all_predictor]
            val_input = station_val_dfs[all_predictor]
            val_input = val_input.reset_index(drop=True)
            test_input = station_test_dfs[all_predictor]
            if lag3:
                path = '/p/lustre2/shiduan/AutogluonModels-LR/'+keep_name+'-include'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-lag-'+str(time_lag)+'-seed-'+str(random_seed)+'-lag3'+'keep-'+str(keep)
            else:
                path = '/p/lustre2/shiduan/AutogluonModels-LR/'+keep_name+'-include'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-lag-'+str(time_lag)+'-seed-'+str(random_seed)+'keep-'+str(keep)
            model = TabularPredictor(label='Q_sim', verbosity=0, 
            path=path).fit(
            train_data=train_input, tuning_data=val_input, hyperparameters=custom_hyperparameters)
            y_pred = model.predict(test_input).values
        if model_type.upper()=='LR':
            model = LinearRegression().fit(train_x, station_train_dfs['Q_sim'])
            y_pred = model.predict(test_x)
        if model_type.upper()=='LOD':
            r2, records, target_pred, last_pred_test, train_pred = station_iteration(station_train_dfs=station_train_dfs, all_predictor=all_predictor,
                                        station_test_dfs=station_test_dfs, station_val_dfs=station_val_dfs, norm=False, plot=False)
            y_pred = last_pred_test
        
        r2 = r2_score(station_test_dfs['Q_sim'].values.reshape(-1, 1), y_pred.reshape(-1, 1))
        r2s_ens.append(r2)
        # print('R2: ', r2)
        if mode_smooth:
            path = '/p/lustre2/shiduan/'+model_type.upper()+'-predictions-smooth/'+keep_name+'-include'+'/'+str(station)+'/'
        else:
            path = '/p/lustre2/shiduan/'+model_type.upper()+'-predictions/'+keep_name+'-include'+'/'+str(station)+'/'
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if lag3:
            file = path+station+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-real_lag3.npy'
        else:
            file = path+station+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-real.npy'
        np.save(file, 
                station_test_dfs['Q_sim'].values.reshape(-1, 1))
        if lag3:
            file = path+station+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-pred_lag3.npy'
        else:
            file = path+station+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-pred.npy'
        np.save(file, 
                y_pred.reshape(-1, 1))
        station_pred[station] = y_pred
        station_real[station] = station_test_dfs['Q_sim'].values.reshape(-1, 1)
    return r2s_ens, station_pred, station_real

path = '/p/lustre2/shiduan/'
modes_keep = []

if level>0:
    for i in range(level):
        path = '/p/lustre2/shiduan/'+model_type.upper()+'-median-results/level-'+str(i)+'.p'
        with open(path, 'rb') as pfile:
            results = pickle.load(pfile)
            modes_keep.append(results['max_mode'])

level_results = {}
print('Level: ', level)
mod_median_r2s = []
candidate_list = []
for mod in mods:
    if mod not in modes_keep:
        modes_keep.append(mod)
        candidate_list.append(mod)
        preds = []
        reals = []
        time_best = 0
        test_gcms = ['IPSL', 'EC', 'ACCESS', 'MPI', 'MIROC', 'CNRM']
        random_seeds = [0, 1, 2, 3, 4, 5]
        # station_ind_all = np.arange(25)

        pool = multiprocessing.Pool(6)
        args_list = [(time_best, [gcm], eof_modes, seed, modes_keep) 
                        for gcm, seed in zip(test_gcms, random_seeds)]
        
        results = pool.map(run, args_list)
        pool.close()
        pool.join()

        # Unpack the results
        r2s_ens_list, pred_list, real_list = zip(*results)
        
        # Now you have separate lists for each set of results
        r2s_ens0, pred0, real0 = r2s_ens_list[0], pred_list[0], real_list[0]
        r2s_ens1, pred1, real1 = r2s_ens_list[1], pred_list[1], real_list[1]
        r2s_ens1, pred2, real2 = r2s_ens_list[2], pred_list[2], real_list[2]
        r2s_ens1, pred3, real3 = r2s_ens_list[3], pred_list[3], real_list[3]
        r2s_ens1, pred4, real4 = r2s_ens_list[4], pred_list[4], real_list[4]
        r2s_ens1, pred5, real5 = r2s_ens_list[5], pred_list[5], real_list[5]
        r2s = []
        for station in station_ids:
            station_pred0 = pred0[station]
            station_pred1 = pred1[station]
            station_pred2 = pred2[station]
            station_pred3 = pred3[station]
            station_pred4 = pred4[station]
            station_pred5 = pred5[station]
            station_real0 = real0[station]
            station_real1 = real1[station]
            station_real2 = real2[station]
            station_real3 = real3[station]
            station_real4 = real4[station]
            station_real5 = real5[station]
            pred_all = np.concatenate((station_pred0, station_pred1, station_pred2, station_pred3,
                                        station_pred4, station_pred5), axis=0)
            real_all = np.concatenate((station_real0, station_real1, station_real2, station_real3,
                                        station_real4, station_real5), axis=0)
            
            r2 = r2_score(real_all, pred_all) # this is the station performance
            r2s.append(r2)
        mod_median_r2s.append(np.median(r2s))
        level_results[mod] = np.median(r2s)
        print('Overall r2 for ' + mod + ' : ', np.median(r2s))
        modes_keep.remove(mod)
# find the maximum median r2:
max_r2 = np.max(mod_median_r2s)
mod_max = candidate_list[np.where(mod_median_r2s==max_r2)[0][0]]    
print('max: ', mod_max, max_r2)
modes_keep.append(mod_max)
level_results['max_mode'] = mod_max
path = '/p/lustre2/shiduan/'+model_type.upper()+'-median-results/level-'+str(level)+'.p'
with open(path, 'wb') as pfile:
    pickle.dump(level_results, pfile)
    
print('Done')
print(level_results)
