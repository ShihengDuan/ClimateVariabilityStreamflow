import pickle
import numpy as np

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total.
station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]
results = {}
with open('real_timeseries_EOF1.p', 'rb') as pfile:
    time_series_eof1 = pickle.load(pfile)
with open('real_timeseries_EOF2.p', 'rb') as pfile:
    time_series_eof2 = pickle.load(pfile)
with open('real_timeseries_EOF3.p', 'rb') as pfile:
    time_series_eof3 = pickle.load(pfile)
with open('real_timeseries_EOF4.p', 'rb') as pfile:
    time_series_eof4 = pickle.load(pfile)
with open('real_timeseries_EOF5.p', 'rb') as pfile:
    time_series_eof5 = pickle.load(pfile)
with open('real_timeseries_EOF6.p', 'rb') as pfile:
    time_series_eof6 = pickle.load(pfile)
# print(time_series_eof1['11528700'].keys()) # stations, Lasso_pred, AutoML_pred etc.,
time_series = [time_series_eof1, time_series_eof2, time_series_eof3, 
               time_series_eof4, time_series_eof5, time_series_eof6]
for ind, station in enumerate(station_ids):
    result_dic = {}
    # AutoLR
    with open('real_r2_AutoLR_EOF1.p', 'rb') as pfile:
        r2_eof1 = pickle.load(pfile)
    with open('real_r2_AutoLR_EOF2.p', 'rb') as pfile:
        r2_eof2 = pickle.load(pfile)
    with open('real_r2_AutoLR_EOF3.p', 'rb') as pfile:
        r2_eof3 = pickle.load(pfile)
    with open('real_r2_AutoLR_EOF4.p', 'rb') as pfile:
        r2_eof4 = pickle.load(pfile)
    with open('real_r2_AutoLR_EOF5.p', 'rb') as pfile:
        r2_eof5 = pickle.load(pfile)
    with open('real_r2_AutoLR_EOF6.p', 'rb') as pfile:
        r2_eof6 = pickle.load(pfile)
    r2s = [r2_eof1[ind], r2_eof2[ind], r2_eof3[ind], r2_eof4[ind], r2_eof5[ind], r2_eof6[ind]]
    r2_max = np.max(r2s)
    eof_number = np.argmax(r2s) + 1
    result_dic['AutoLR']=(r2_max, eof_number)
    # AutoML
    with open('real_r2_AutoML_EOF1.p', 'rb') as pfile:
        r2_eof1 = pickle.load(pfile)
    with open('real_r2_AutoML_EOF2.p', 'rb') as pfile:
        r2_eof2 = pickle.load(pfile)
    with open('real_r2_AutoML_EOF3.p', 'rb') as pfile:
        r2_eof3 = pickle.load(pfile)
    with open('real_r2_AutoML_EOF4.p', 'rb') as pfile:
        r2_eof4 = pickle.load(pfile)
    with open('real_r2_AutoML_EOF5.p', 'rb') as pfile:
        r2_eof5 = pickle.load(pfile)
    with open('real_r2_AutoML_EOF6.p', 'rb') as pfile:
        r2_eof6 = pickle.load(pfile)
    r2s = [r2_eof1[ind], r2_eof2[ind], r2_eof3[ind], r2_eof4[ind], r2_eof5[ind], r2_eof6[ind]]
    r2_max = np.max(r2s)
    eof_number = np.argmax(r2s) + 1
    result_dic['AutoML']=(r2_max, eof_number, 
                          time_series[eof_number-1][station]['autoML_pred'],
                          time_series[eof_number-1][station]['autoML_true'])
    # LOD
    with open('real_r2_LOD_EOF1.p', 'rb') as pfile:
        r2_eof1 = pickle.load(pfile)
    with open('real_r2_LOD_EOF2.p', 'rb') as pfile:
        r2_eof2 = pickle.load(pfile)
    with open('real_r2_LOD_EOF3.p', 'rb') as pfile:
        r2_eof3 = pickle.load(pfile)
    with open('real_r2_LOD_EOF4.p', 'rb') as pfile:
        r2_eof4 = pickle.load(pfile)
    with open('real_r2_LOD_EOF5.p', 'rb') as pfile:
        r2_eof5 = pickle.load(pfile)
    with open('real_r2_LOD_EOF6.p', 'rb') as pfile:
        r2_eof6 = pickle.load(pfile)
    r2s = [r2_eof1[ind], r2_eof2[ind], r2_eof3[ind], r2_eof4[ind], r2_eof5[ind], r2_eof6[ind]]
    r2_max = np.max(r2s)
    eof_number = np.argmax(r2s) + 1
    result_dic['LOD']=(r2_max, eof_number,
                       time_series[eof_number-1][station]['LOD_pred'],
                       time_series[eof_number-1][station]['LOD_true'])
    # LR
    with open('real_r2_LR_EOF1.p', 'rb') as pfile:
        r2_eof1 = pickle.load(pfile)
    with open('real_r2_LR_EOF2.p', 'rb') as pfile:
        r2_eof2 = pickle.load(pfile)
    with open('real_r2_LR_EOF3.p', 'rb') as pfile:
        r2_eof3 = pickle.load(pfile)
    with open('real_r2_LR_EOF4.p', 'rb') as pfile:
        r2_eof4 = pickle.load(pfile)
    with open('real_r2_LR_EOF5.p', 'rb') as pfile:
        r2_eof5 = pickle.load(pfile)
    with open('real_r2_LR_EOF6.p', 'rb') as pfile:
        r2_eof6 = pickle.load(pfile)
    r2s = [r2_eof1[ind], r2_eof2[ind], r2_eof3[ind], r2_eof4[ind], r2_eof5[ind], r2_eof6[ind]]
    r2_max = np.max(r2s)
    eof_number = np.argmax(r2s) + 1
    result_dic['LR']=(r2_max, eof_number,
                       time_series[eof_number-1][station]['LR_pred'],
                       time_series[eof_number-1][station]['LR_true'])
    # Lasso
    with open('real_r2_Lasso_EOF1.p', 'rb') as pfile:
        r2_eof1 = pickle.load(pfile)
    with open('real_r2_LR_EOF2.p', 'rb') as pfile:
        r2_eof2 = pickle.load(pfile)
    with open('real_r2_LR_EOF3.p', 'rb') as pfile:
        r2_eof3 = pickle.load(pfile)
    with open('real_r2_LR_EOF4.p', 'rb') as pfile:
        r2_eof4 = pickle.load(pfile)
    with open('real_r2_LR_EOF5.p', 'rb') as pfile:
        r2_eof5 = pickle.load(pfile)
    with open('real_r2_LR_EOF6.p', 'rb') as pfile:
        r2_eof6 = pickle.load(pfile)
    r2s = [r2_eof1[ind], r2_eof2[ind], r2_eof3[ind], r2_eof4[ind], r2_eof5[ind], r2_eof6[ind]]
    r2_max = np.max(r2s)
    eof_number = np.argmax(r2s) + 1
    result_dic['Lasso']=(r2_max, eof_number,
                       time_series[eof_number-1][station]['Lasso_pred'],
                       time_series[eof_number-1][station]['Lasso_true'])
    # Ridge
    with open('real_r2_Ridge_EOF1.p', 'rb') as pfile:
        r2_eof1 = pickle.load(pfile)
    with open('real_r2_Ridge_EOF2.p', 'rb') as pfile:
        r2_eof2 = pickle.load(pfile)
    with open('real_r2_Ridge_EOF3.p', 'rb') as pfile:
        r2_eof3 = pickle.load(pfile)
    with open('real_r2_Ridge_EOF4.p', 'rb') as pfile:
        r2_eof4 = pickle.load(pfile)
    with open('real_r2_Ridge_EOF5.p', 'rb') as pfile:
        r2_eof5 = pickle.load(pfile)
    with open('real_r2_Ridge_EOF6.p', 'rb') as pfile:
        r2_eof6 = pickle.load(pfile)
    r2s = [r2_eof1[ind], r2_eof2[ind], r2_eof3[ind], r2_eof4[ind], r2_eof5[ind], r2_eof6[ind]]
    r2_max = np.max(r2s)
    eof_number = np.argmax(r2s) + 1
    result_dic['Ridge']=(r2_max, eof_number,
                       time_series[eof_number-1][station]['Ridge_pred'],
                       time_series[eof_number-1][station]['Ridge_true'])
    results[station]=result_dic # dictionary of dictionary
    # PLS
    with open('real_r2_PLS_EOF1.p', 'rb') as pfile:
        r2_eof1 = pickle.load(pfile)
    with open('real_r2_PLS_EOF2.p', 'rb') as pfile:
        r2_eof2 = pickle.load(pfile)
    with open('real_r2_PLS_EOF3.p', 'rb') as pfile:
        r2_eof3 = pickle.load(pfile)
    with open('real_r2_PLS_EOF4.p', 'rb') as pfile:
        r2_eof4 = pickle.load(pfile)
    with open('real_r2_PLS_EOF5.p', 'rb') as pfile:
        r2_eof5 = pickle.load(pfile)
    with open('real_r2_PLS_EOF6.p', 'rb') as pfile:
        r2_eof6 = pickle.load(pfile)
    r2s = [r2_eof1[ind], r2_eof2[ind], r2_eof3[ind], r2_eof4[ind], r2_eof5[ind], r2_eof6[ind]]
    r2_max = np.max(r2s)
    eof_number = np.argmax(r2s) + 1
    result_dic['PLS']=(r2_max, eof_number,
                       time_series[eof_number-1][station]['PLS_pred'],
                       time_series[eof_number-1][station]['PLS_true'])
    results[station]=result_dic # dictionary of dictionary

print(results)
with open('results.p', 'wb') as pfile:
    pickle.dump(results, pfile)
