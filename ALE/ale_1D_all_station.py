import os
import inspect
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from alepython.ale import ale_plot, get_ale
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
from tools import build_df
from iteration import get_seasonal_data
import pickle
from matplotlib import pyplot as plt
import argparse
from sklearn.linear_model import LinearRegression

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', type=str)
    parser.add_argument('--model_type', type=str, choices=['automl', 'lr'])
    args = vars(parser.parse_args())
    return args

args = get_args()
variable = args['variable']
model_type = args['model_type']

# Get data
station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]
hist_co2 = pd.read_csv('../historical_co2.csv', index_col=['wy', 'year', 'month'])

hist_co2 = hist_co2/300
def get_centres(x):
    return (x[1:] + x[:-1]) / 2
def load_data(model, ensembles, scenario='hist', start_wy=1951):
    hist_dfs = []
    for member in ensembles:
        path = '../data/'+model+'-Streamflow-csv/r'+str(member)+'-'+scenario+'_q_csv_monthly.csv'
        if os.path.exists(path):
            hist_q_df = pd.read_csv(path, index_col=['wy', 'year', 'month'])
            hist_q_df[hist_q_df<0]=0
            hist_modes_df = pd.read_csv('../data/'+model+'-Modes-csv-CBF-NewOBS/r'+str(member)+'-'+scenario+'_modes_csv_monthly.csv', index_col=['wy', 'year', 'month'])
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


def run_ale(station, variable, peak, lag3):
    station_ids = [station]
    if 'IPSL' in test_gcm:
        IPSL_hist_dfs = load_data('IPSL-CM6A-LR', ensembles=range(1, 12)) # range(1, 34)
        print(len(IPSL_hist_dfs))
    if 'EC' in test_gcm:
        EC_hist_dfs = load_data('EC-Earth3', ensembles=[13, 15, 16, 11, 25, 24, 23, 22, 21, 1, 4, 5, 6, 7, 9, 10, ])
        print(len(EC_hist_dfs))
    if 'ACCESS' in test_gcm:
        ACCESS_hist_dfs = load_data('ACCESS-ESM1-5', ensembles=range(1, 11))
        print(len(ACCESS_hist_dfs))
    if 'MIROC' in test_gcm:
        MIROC_hist_dfs = load_data('MIROC6', ensembles=range(1, 11))
        print(len(MIROC_hist_dfs))
    if 'MPI' in test_gcm:
        MPI_hist_dfs = load_data('MPI-ESM1-2-LR', ensembles=range(1, 11))
        print(len(MPI_hist_dfs))
    if 'CNRM' in test_gcm:
        CNRM_hist_dfs = load_data('CNRM-ESM2-1', ensembles=range(1, 11))
        print(len(CNRM_hist_dfs))
    
    predictor = ['PDO_eof', 'AMO_eof', 'PNA_eof', 
                        'NAM_eof', 'NAO_eof', 'SAM_eof']
    predictor_high = []
    for i in range(1, eof_modes+1):
        for p in predictor:
            predictor_high.append(p+'_'+str(i))
    predictor_high.append('nino34')
    predictor_high.append('co2')
    
    all_predictor = []
    if lag3:
        for p in predictor_high:
            all_predictor.append(p+'_lag3')
    else:
        for p in predictor_high:
            all_predictor.append(p)
    # all_predictor.append('Q_sim')
    print(len(all_predictor))

    for station in station_ids: # train only one station to be faster. 
        print(station, peak)
        if peak==1:
            months = [12, peak, peak+1]
        elif peak==12:
            months = [peak-1, peak, 1]
        else:
            months = [peak-1, peak, peak+1]
        if 'IPSL' in test_gcm:
            mam_dfs_IPSL_hist = get_seasonal_data(IPSL_hist_dfs, months=months, smooth_mode=True)
        if 'ACCESS' in test_gcm:
            mam_dfs_ACCESS_hist = get_seasonal_data(ACCESS_hist_dfs, months=months, smooth_mode=True)
        if 'MIROC' in test_gcm:
            mam_dfs_MIROC_hist = get_seasonal_data(MIROC_hist_dfs, months=months, smooth_mode=True)
        if 'MPI' in test_gcm:
            mam_dfs_MPI_hist = get_seasonal_data(MPI_hist_dfs, months=months, smooth_mode=True)
        if 'CNRM' in test_gcm:
            mam_dfs_CNRM_hist = get_seasonal_data(CNRM_hist_dfs, months=months, smooth_mode=True)
        if 'EC' in test_gcm:
            mam_dfs_EC_hist = get_seasonal_data(EC_hist_dfs, months=months, smooth_mode=True)
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
        if 'EC' in test_gcm:
            for df in mam_dfs_EC_hist[:10]:
                df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
                if 'EC' not in test_gcm:
                    all_dfs.append(df)
                else:
                    test_dfs.append(df)
                EC_dfs.append(df)
        if 'IPSL' in test_gcm:
            for df in mam_dfs_IPSL_hist[:10]:
                df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
                if 'IPSL' not in test_gcm:
                    all_dfs.append(df)
                else:
                    test_dfs.append(df)
                IPSL_dfs.append(df)
        if 'ACCESS' in test_gcm:
            for df in mam_dfs_ACCESS_hist[:]:
                df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
                if 'ACCESS' not in test_gcm:
                    all_dfs.append(df)
                else:
                    test_dfs.append(df)
                ACCESS_dfs.append(df)
        if 'MIROC' in test_gcm:
            for df in mam_dfs_MIROC_hist[:]:
                df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
                if 'MIROC' not in test_gcm:
                    all_dfs.append(df)
                else:
                    test_dfs.append(df)
                MIROC_dfs.append(df)
        if 'MPI' in test_gcm:
            for df in mam_dfs_MPI_hist[:]:
                df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
                if 'MPI' not in test_gcm:
                    all_dfs.append(df)
                else:
                    test_dfs.append(df)
                MPI_dfs.append(df)
        if 'CNRM' in test_gcm:
            for df in mam_dfs_CNRM_hist[:]:
                df['Q_sim'] = (df['Q_sim']-df['Q_sim'].groupby('station_id').mean())/df['Q_sim'].groupby('station_id').std()
                if 'CNRM' not in test_gcm:
                    all_dfs.append(df)
                else:
                    test_dfs.append(df)
                CNRM_dfs.append(df)

        # train_dfs, val_dfs = train_test_split(all_dfs, test_size=.3, shuffle=True, random_state=random_seed)
        # train_dfs = pd.concat(train_dfs)
        # val_dfs = pd.concat(val_dfs)
        test_dfs = pd.concat(test_dfs)

        # station_train_dfs = train_dfs.xs(station, level=1)
        station_test_dfs = test_dfs.xs(station, level=1)
        # station_val_dfs = val_dfs.xs(station, level=1)
        
        # train_input = station_train_dfs[all_predictor]
        # val_input = station_val_dfs[all_predictor]
        # val_input = val_input.reset_index(drop=True)
        test_input = station_test_dfs[all_predictor]
    # print(test_input.columns[9:10])
    print(test_input.columns)
    if lag3:
        column = [variable+'_lag3']
    else:
        column = [variable]
    path = 'ale-figures/ale-1D-EOF'+str(eof_modes)+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    ale, quantiles = get_ale(
        model,
        train_set=test_input,
        features=test_input[column].columns,
        bins=20,
        monte_carlo=False,
        monte_carlo_rep=100,
        monte_carlo_ratio=0.5,
    )
    return ale, quantiles


eof_modes = 6
time_lag = 0
test_gcm=['EC']
random_seed = 1
# variable = 'PNA_eof_5'

fig = plt.figure()
scores = []
slopes = []
for ind, station in enumerate(station_ids):
    peak = station_peaks[ind]    
    if peak>3 and peak<12:
        lag3 = True
    else:
        lag3 = False
    print(station)
    if lag3:
        if model_type.upper()=='AUTOML':
            path = '/p/lustre2/shiduan/AutogluonModels/'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)+'-lag3'
            model = TabularPredictor.load(
                path=path, 
                require_py_version_match=False)
        elif model_type.upper()=='LR':
            file = '/p/lustre2/shiduan/'+model_type.upper()+'-smooth/LS-'+str(station)+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)
            with open(file, 'rb') as pfile:
                model = pickle.load(pfile)
        else:
            print('No Such Model')
    else:
        if model_type.upper()=='AUTOML':
            path = '/p/lustre2/shiduan/AutogluonModels/'+'ag-'+str(station)+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)
            model = TabularPredictor.load(
                path=path, 
                require_py_version_match=False)
        elif model_type.upper()=='LR':
            file = '/p/lustre2/shiduan/'+model_type.upper()+'-smooth/LS-'+str(station)+'-EOF-'+str(eof_modes)+'-seed-'+str(random_seed)
            with open(file, 'rb') as pfile:
                model = pickle.load(pfile)
        else:
            print('No Such Model')
    ale_station = []
    
    for random_seed, test_gcm in zip([0, 1, 2, 3, 4, 5], 
                                     [['IPSL'], ['EC'], ['ACCESS'], ['MPI'], ['MIROC'], ['CNRM']]):
        '''for random_seed, test_gcm in zip([0, ], 
                                     [['IPSL'],]):'''
        ale, quantiles = run_ale(station, variable, lag3=lag3, peak=peak)
        ale_station.append(ale.reshape(-1, 1))
    
    '''for ale in ale_station:
        plt.plot(get_centres(quantiles), ale, color='tab:blue')'''
    ale_station = np.concatenate(ale_station, axis=1)
    ale_station_mean = np.mean(ale_station, axis=1)
    # ale_station_upper = np.percentile(ale_station, axis=1, q=95)
    # ale_station_lower = np.percentile(ale_station, axis=1, q=5)
    # ale_station_lower = np.min(ale_station, axis=1)
    # ale_station_upper = np.max(ale_station, axis=1)
    print(station)
    # plt.plot(get_centres(quantiles), ale_station_mean, label=station, color='black')
    # plt.fill_between(get_centres(quantiles), y1=ale_station_lower, y2=ale_station_upper)
    x = get_centres(quantiles).reshape(-1, 1)
    y = ale_station_mean.reshape(-1, 1)
    lr_model = LinearRegression(fit_intercept=False).fit(x, y)
    score = lr_model.score(x, y)
    slope = lr_model.coef_[0][0]
    scores.append(score)
    slopes.append(slope)
    print(slope.shape)
    print(score.shape)
slopes = np.array(slopes)
scores = np.array(scores)
np.save('slopes-'+variable, slopes)
np.save('scores-'+variable, scores)

ax = fig.add_subplot(121)
ax.boxplot(scores, positions=[0])
ax.set_title('R2', fontsize=12)
ax = fig.add_subplot(122)
ax.boxplot(slopes, positions=[0])
ax.set_title('slope', fontsize=12)
# plt.legend()
plt.savefig('ale-figures/slope-score-'+variable+'.png')


