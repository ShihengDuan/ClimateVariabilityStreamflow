import numpy as np
import pandas as pd
import xarray as xa




def get_WY(data):
    water_year = (data.time.dt.month >= 10) + data.time.dt.year
    data.coords['WY'] = water_year
    year_data = data.groupby('WY').mean()
    return year_data
def get_WY_q(data):
    water_year = (data.date.dt.month >= 10) + data.date.dt.year
    data.coords['WY'] = water_year
    year_data = data.groupby('WY').mean()
    return year_data

def load_modes(path1='ModesPMP/IPSL-CM6A-LR_ts_hist_GCM/',
               path2='ModesPMP/IPSL-CM6A-LR_ts_ssp245_GCM/',
               path3='ModesPMP/IPSL-CM6A-LR_ts_ssp370_GCM/',
               path4='ModesPMP/IPSL-CM6A-LR_ts_ssp585_GCM/',
               path5='ModesPMP/IPSL-CM6A-LR_psl_historical_GCM/',
               path6='ModesPMP/IPSL-CM6A-LR_psl_ssp245_GCM/',
               path7='ModesPMP/IPSL-CM6A-LR_psl_ssp370_GCM/',
               path8='ModesPMP/IPSL-CM6A-LR_psl_ssp585_GCM/'
              ):
    sst_modes = ['PDO', 'NPGO', 'AMO']
    slp_modes = ['PNA', 'NAM', 'NAO', 'SAM', 'NPO']
    hist_modes_eof = []
    ssp245_modes_eof = []
    ssp370_modes_eof = []
    ssp585_modes_eof = []
    hist_modes_cbf = []
    ssp245_modes_cbf = []
    ssp370_modes_cbf = []
    ssp585_modes_cbf = []

    for m in sst_modes:
        # historical
        path = path1
        if m=='NPGO':
            pc_modes = xa.open_dataset(path+m+'_ts_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_1850-2014.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_ts_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_1850-2014.nc')
        pc_modes = pc_modes.pc
        hist_modes_eof.append(pc_modes)
        if m=='NPGO':
            pc_modes = xa.open_dataset(path+m+'_ts_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_1850-2014_cbf.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_ts_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_1850-2014_cbf.nc')
        pc_modes = pc_modes.pc
        hist_modes_cbf.append(pc_modes)
        # ssp245
        path = path2
        if m=='NPGO':
            pc_modes = xa.open_dataset(path+m+'_ts_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_ts_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        pc_modes = pc_modes.pc
        ssp245_modes_eof.append(pc_modes)
        if m=='NPGO':
            pc_modes = xa.open_dataset(path+m+'_ts_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_ts_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        pc_modes = pc_modes.pc
        ssp245_modes_cbf.append(pc_modes)
        # ssp370
        path = path3
        if m=='NPGO':
            pc_modes = xa.open_dataset(path+m+'_ts_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_ts_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        pc_modes = pc_modes.pc
        ssp370_modes_eof.append(pc_modes)
        if m=='NPGO':
            pc_modes = xa.open_dataset(path+m+'_ts_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_ts_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        pc_modes = pc_modes.pc
        ssp370_modes_cbf.append(pc_modes)
        # ssp585
        path = path4
        if m=='NPGO':
            pc_modes = xa.open_dataset(path+m+'_ts_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_ts_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        pc_modes = pc_modes.pc
        ssp585_modes_eof.append(pc_modes)
        if m=='NPGO':
            pc_modes = xa.open_dataset(path+m+'_ts_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_ts_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        pc_modes = pc_modes.pc
        ssp585_modes_cbf.append(pc_modes)

    for m in slp_modes:
        # historical
        path = path5
        if m=='NPO':
            pc_modes = xa.open_dataset(path+m+'_psl_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_1850-2014.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_psl_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_1850-2014.nc')
        pc_modes = pc_modes.pc
        hist_modes_eof.append(pc_modes)
        if m=='NPO':
            pc_modes = xa.open_dataset(path+m+'_psl_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_1850-2014_cbf.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_psl_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_1850-2014_cbf.nc')
        pc_modes = pc_modes.pc
        hist_modes_cbf.append(pc_modes)
        # ssp245
        path = path6
        if m=='NPO':
            pc_modes = xa.open_dataset(path+m+'_psl_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_psl_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        pc_modes = pc_modes.pc
        ssp245_modes_eof.append(pc_modes)
        if m=='NPO':
            pc_modes = xa.open_dataset(path+m+'_psl_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_psl_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        pc_modes = pc_modes.pc
        ssp245_modes_cbf.append(pc_modes)
        # ssp370
        path = path7
        if m=='NPO':
            pc_modes = xa.open_dataset(path+m+'_psl_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_psl_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        pc_modes = pc_modes.pc
        ssp370_modes_eof.append(pc_modes)
        if m=='NPO':
            pc_modes = xa.open_dataset(path+m+'_psl_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_psl_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        pc_modes = pc_modes.pc
        ssp370_modes_cbf.append(pc_modes)
        # ssp585
        path = path8
        if m=='NPO':
            pc_modes = xa.open_dataset(path+m+'_psl_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_psl_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100.nc')
        pc_modes = pc_modes.pc
        ssp585_modes_eof.append(pc_modes)
        if m=='NPO':
            pc_modes = xa.open_dataset(path+m+'_psl_EOF2_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        else:
            pc_modes = xa.open_dataset(path+m+'_psl_EOF1_monthly_cmip5_IPSL-CM6A-LR_historical_r1i1p1_mo_atm_2015-2100_cbf.nc')
        pc_modes = pc_modes.pc
        ssp585_modes_cbf.append(pc_modes)
    return hist_modes_eof, ssp245_modes_eof, ssp370_modes_eof, ssp585_modes_eof, hist_modes_cbf, ssp245_modes_cbf, ssp370_modes_cbf, ssp585_modes_cbf


    