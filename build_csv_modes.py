import xarray as xa
import pandas as pd
import numpy as np
import glob
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--scenario', type=str, default='historical')
    parser.add_argument('--real', type=str)  # realization, e.g., r1 r2 r3
    args = vars(parser.parse_args())
    return args

sst_modes = ['PDO', 'AMO']
slp_modes = ['PNA', 'NAM', 'NAO', 'SAM']


def get_modes(member, model, scenario, cf_time=True):
    hist_modes_eof = []
    hist_modes_slope = []
    for m in sst_modes:
        for f in range(1, 6):
            files = glob.glob(sst_path+m+'_ts_EOF'+str(f)+'_monthly_cmip6_'+model+'_'+scenario+'_'+\
                                       member+'_mo_atm_*.nc')
            if len(files)>0:
                file = files[0]
            else:
                print('No File: ', sst_path+m+'_ts_EOF'+str(f)+'_monthly_cmip6_'+model+'_'+scenario+'_'+\
                                       member+'_mo_atm_*.nc')
                # break
            pc_modes = xa.open_dataset(file, use_cftime=cf_time)
            pc_timeseries = pc_modes.pc
            hist_modes_eof.append(pc_timeseries)
            hist_modes_slope.append(pc_modes.slope)
    for m in slp_modes:
        for f in range(1, 6):
            files = glob.glob(slp_path+m+'_psl_EOF'+str(f)+'_monthly_cmip6_'+model+'_'+scenario+'_'+\
                                       member+'_mo_atm_*.nc')
            if len(files)>0:
                file = files[0]
            else:
                print('No File: ', slp_path+m+'_psl_EOF'+str(f)+'_monthly_cmip6_'+model+'_'+scenario+'_'+\
                                       member+'_mo_atm_*.nc')
                # break
            pc_modes = xa.open_dataset(file, use_cftime=cf_time)
            pc_timeseries = pc_modes.pc
            hist_modes_eof.append(pc_timeseries)
            hist_modes_slope.append(pc_modes.slope)
    nino34 = xa.open_dataarray(nino_path+member+'.nc', use_cftime=cf_time)
    return hist_modes_eof, hist_modes_slope, nino34

def build_csv(data, name):
    data_year = np.array(data.time.dt.year)
    data_month = np.array(data.time.dt.month)
    wy = (data_month>=10) + data_year
    data_csv = pd.DataFrame(data={name: data, 'year': data_year, 'month': data_month, 'wy': wy})
    data_csv.set_index(['year', 'month', 'wy'], inplace=True)
    return data_csv

args = get_args()
model = args['model']
scenario = args['scenario']
m = args['real']
print(model, ' ', scenario, ' ', m)

sst_path = 'DuanPMP/'+model+'-ts-'+scenario+'-CBF/'
slp_path = 'DuanPMP-NewOBS/'+model+'-psl-'+scenario+'-CBF/'
nino_path = 'NINO34/nino34_'+model+'_'+scenario+'_'

if model=='CNRM-ESM2-1':
    member = m+'i1p1f2'
else:
    member = m+'i1p1f1'
print(member)
hist_modes_eof, hist_modes_slope, hist_nino = get_modes(member, model=model, scenario=scenario)
modes_names = ['PDO', 'AMO', 'PNA', 'NAM', 'NAO', 'SAM']
all_csv_hist = []
n_mod = 0
f = 1
for i in range(len(hist_modes_eof)):
    name = modes_names[n_mod]+'_eof_'+str(f)
    # print(name)
    data_csv_pca = build_csv(hist_modes_eof[i], name=name)
    f+=1
    if f==6: # EOF1 to EOF5
        f = 1
        n_mod+=1
    all_csv_hist.append(data_csv_pca)
data_csv_nino = build_csv(hist_nino, name='nino34')
all_csv_hist.append(data_csv_nino)
hist_modes_csv = pd.concat(all_csv_hist, axis=1)
if not os.path.exists('data/'+model+'-Modes-csv-CBF-NewOBS/'):
    os.makedirs('data/'+model+'-Modes-csv-CBF-NewOBS/')
if scenario=='historical':
    scen = 'hist'
else:
    scen = scenario
hist_modes_csv.to_csv('data/'+model+'-Modes-csv-CBF-NewOBS/'+str(m)+'-'+scen+'_modes_csv_monthly.csv')
