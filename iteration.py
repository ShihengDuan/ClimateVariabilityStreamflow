from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

def get_seasonal_data(dfs, months=[3, 4, 5], smooth_mode=False):
    season_dfs = []
    for df in dfs:
        season_df = df.loc[pd.IndexSlice[:, :, months], :].sort_index() # select season
        # season_df['Q_sim'][season_df['Q_sim']<0]=0
        if not smooth_mode:
        # get monthly modes:
            season_dis = season_df[['Q_sim', 'station_id']]
            season_dis = season_dis.groupby(['wy', 'station_id']).mean(numeric_only=True) # groupby station and water year
            # season_dis[['Q_sim']] = (season_dis[['Q_sim']] - season_dis[
            #     ['Q_sim']].groupby('station_id').mean())/ season_dis[['Q_sim']].groupby('station_id').std() # calculate std-anomaly
            season_modes = season_df.xs(months[-1], level=2, drop_level=True) # select last month in the season. 
            season_modes = season_modes.set_index('station_id', append=True)
            season_modes.index = season_modes.index.droplevel(1)
            season_modes = season_modes.drop(columns='Q_sim')
            season_modes_dis = season_modes.join(season_dis, on=['wy', 'station_id'])
            season_dfs.append(season_modes_dis)
        else:
            season_dis = season_df.groupby(['wy', 'station_id']).mean(numeric_only=True) # groupby station and water year
            # season_dis[['Q_sim']] = (season_dis[['Q_sim']] - season_dis[
            #     ['Q_sim']].groupby('station_id').mean())/ season_dis[['Q_sim']].groupby('station_id').std() # calculate std-anomaly
            season_dfs.append(season_dis)
    return season_dfs

def calculate_adj_r2(target, pred, p):
    r2 = r2_score(target, pred)
    n = len(target)
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return r2, adj_r2

def station_iteration(all_predictor, station_train_dfs, station_test_dfs, 
                      station_val_dfs, plot=False, verbose=False, norm=False):
    """
    all_predictor: the predictors used in linear iteration. 
    station_train_dfs: dataframe that only contain the training data for one station. 
    station_test_dfs: testing data for one station. 
    station_val_dfs: validation data for one station. 
    plot: iteration plot. 
    verbose: output max_predictor information. 
    
    return
    r2: testing score
    records: information about linear iteration (iteration, linear model). 
    """
    target = station_train_dfs['Q_sim'].values
    if norm:
        target = (target-np.mean(target))/np.std(target)
    last_pred = np.zeros_like(target)
    target_pred = station_test_dfs['Q_sim'].values
    last_pred_test = np.zeros_like(target_pred)
    target_val = station_val_dfs['Q_sim'].values
    if norm:
        target_val = (target_val-np.mean(target_val))/np.std(target_val)
    last_pred_val = np.zeros_like(target_val)
    skip = 0
    records = {}
    iter_n = 0
    prev_adj_r2 = -10
    for i in range(25): # 10 iterations
        max_r = 0
        max_j = -1
        for j, predictor in enumerate(all_predictor[skip:]):
            r, p = pearsonr(target, station_train_dfs[predictor])
            if p<=0.05:
                if np.abs(r)>np.abs(max_r):
                    max_r = r
                    max_j = j
                    # print('Found: ', r, j)
        if max_r==0:
            if verbose:
                print('No More~', iter_n, ' val_r2: ', r2, ' val_adj_r2: ', adj_r2, ' records: ', len(records))
            break
        max_predictor = all_predictor[skip:][max_j]
        if verbose:
            print('max_predictor: ', max_predictor)
        model = LinearRegression().fit(station_train_dfs[max_predictor].values.reshape(-1, 1), target)
        # print('coef: ', model.coef_[0])
        pred = model.predict(station_train_dfs[max_predictor].values.reshape(-1, 1))
        # print('r2-train: ', r2_score(station_train_dfs['Q_sim'].values, pred+last_pred), ' ', )
        target = target - pred
        # print(np.mean(np.square(target)), ' new target')
        last_pred += pred

        pred_val = model.predict(station_val_dfs[max_predictor].values.reshape(-1, 1))
        # print('r2-test: ', r2_score(station_test_dfs['Q_sim'].values, pred_test+last_pred_test), ' \n', )
        r2, adj_r2 = calculate_adj_r2(station_val_dfs['Q_sim'].values, pred_val+last_pred_val, p=i+1)
        if r2<prev_adj_r2:
            if verbose:
                print('Not increase~', iter_n, ' val_r2: ', r2, ' val_adj_r2: ', adj_r2, ' records: ', len(records))
            break
        else:
            prev_adj_r2 = r2
            last_pred_val += pred_val
            records[i] = (model, max_predictor)
            iter_n+=1
    train_pred = last_pred
    target_pred = station_test_dfs['Q_sim'].values[:]
    if norm:
        target_pred = (target_pred-np.mean(target_pred))/np.std(target_pred)
    last_pred_test = np.zeros_like(target_pred)
    if plot:
        fig = plt.figure(figsize=(12, 3))
    for i in range(len(records)):
        model, max_predictor = records[i]
        pred_test = model.predict(station_test_dfs[max_predictor].values.reshape(-1, 1))
        last_pred_test+=pred_test[:]
        if plot:
            plt.plot(last_pred_test[:100])
    if plot:
        plt.plot(target_pred[:100], alpha=0.2, color='black')
        plt.show()
    r2, adj_r2 = calculate_adj_r2(target_pred, last_pred_test, p=i+1)
    if verbose:
        print('test_r2: ', r2, ' test_adj_r2: ', adj_r2, ' \n')
    
    return r2, records, target_pred, last_pred_test, train_pred

def get_prediction(records, station_df):
    predictors = []
    for ind, i in enumerate(records):
        model, max_predictor = records[i] # (model, max_predictor)
        predictors.append(max_predictor)
        pred = model.predict(station_df[max_predictor].values.reshape(-1, 1))
        if ind==0:
            all_prediction = pred
        else:
            all_prediction = all_prediction+pred
    predictors = list(set(predictors))
    return all_prediction, predictors
