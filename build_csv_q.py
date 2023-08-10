import xarray as xa
import pandas as pd

import numpy as np
import pickle

import os
import argparse

def build_csv_q(data, name):
    data_year = np.array(data.date.dt.year)
    data_month = np.array(data.date.dt.month)
    wy = (data_month>=10) + data_year
    data_csv = pd.DataFrame(data={name: data, 'year': data_year, 'month': data_month, 'wy': wy})
    data_csv.set_index(['year', 'month', 'wy'], inplace=True)
    return data_csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--scenario', type=str, default='historical')
    parser.add_argument('--real', type=str)  # realization, e.g., r1 r2 r3
    args = vars(parser.parse_args())
    return args


args = get_args()
model = args['model']
scenario = args['scenario']
member = args['real']
print(model, ' ', scenario, ' ', member)

file = model + '-results/proj-'+member+'-'+scenario+'.p'
if os.path.exists(file):
    with open(file, 'rb') as p:
        historical = pickle.load(p)
    hist_q_csv_list = []
    for i in historical:
        q = historical[i]['1D']['xr']['Q_sim']
        q = q.dropna(dim='date')
        # qs_historical[i] = q
        monthly_q = q.resample(date='1M').mean().sel(time_step=0)
        data_csv = build_csv_q(monthly_q, name=i)
        hist_q_csv_list.append(data_csv)
    hist_q_csv = pd.concat(hist_q_csv_list, axis=1)
    if scenario=='historical':
        scen = 'hist'
    else:
        scen = scenario
    if not os.path.exists('data/'+model+'-Streamflow-csv/'):
        os.makedirs('data/'+model+'-Streamflow-csv/')
    hist_q_csv.to_csv('data/'+model+'-Streamflow-csv/'+str(member)+'-'+scen+'_q_csv_monthly.csv')
    print('Done')
else:
    print('No File')
