import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, r2_score
import os

station_ids = ['10336645', '10336660', '11124500', '11141280', '11143000', '11148900', '11151300', '11176400', '11230500', 
               '11237500', '11264500', '11266500', '11284400', '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', '11482500', '11522500', '11523200', '11528700']
camel_topo = pd.read_csv('/usr/workspace/shiduan/neuralhydrology/data/camels_us/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_topo.txt', delimiter=';')
# camel_topo = pd.read_csv('/Users/duan5/Downloads/camels_topo.txt', delimiter=';')

def get_peak_month(station, real_df):
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

def load_data(model, ensembles, co2:pd.DataFrame, scenario='hist', start_wy=1951, end_wy=2013):
    hist_dfs = []
    for member in ensembles:
        path = '/g/g92/shiduan/MyWorkSpace/hydro/data/'+model+'-Streamflow-csv/r'+str(member)+'-'+scenario+'_q_csv_monthly.csv'
        if os.path.exists(path):
            hist_q_df = pd.read_csv(path, index_col=['wy', 'year', 'month'])
            hist_q_df[hist_q_df<0]=0
            hist_modes_df = pd.read_csv('/g/g92/shiduan/MyWorkSpace/hydro/data/'+model+'-Modes-csv-CBF-NewOBS/r'+str(member)+'-'+scenario+'_modes_csv_monthly.csv', index_col=['wy', 'year', 'month'])
            hist_modes_df['modesWY']=hist_modes_df.index
            hist_modes_df_co2 = pd.concat((hist_modes_df, (co2)), axis=1)
            hist_modes_df_co2 = hist_modes_df_co2.sort_index(level=0)
            start_wy = np.max([hist_q_df.index.get_level_values(0)[0], hist_modes_df.index.get_level_values(0)[0]])
            hist_df = build_df(station_ids, hist_q_df, 0, 12, 
                           hist_modes_df_co2, start_wy+1, 1, end_wy, 1, norm=False)
            hist_df['lat'] = pd.to_numeric(hist_df['lat'])
            hist_df['lon'] = pd.to_numeric(hist_df['lon'])
            hist_df['ele'] = pd.to_numeric(hist_df['ele'])
            hist_dfs.append(hist_df)
    return hist_dfs

def get_year_month(year, month, lag):
    new_month = (month + lag) % 12
    new_year = year + ((month + lag) // 12)
    if new_month == 0:
        new_month = 12
        new_year -= 1
    return new_year, new_month

def build_df(station_ids, dis_df, time_lead, time_window, \
    mode_df, mode_start_year, mode_start_month, mode_end_year, mode_end_month, norm=False):
    """
    Parameters:
    dis_df: 
        discharge data frame of all stations in California. 
    time_lead: 
        the month lead of modes of varibility. 0 is concurrent month. 
    time_window: 
        months that prior to the start time of modes. 0 is concurrent month, 1 means 1 month prior. 
    mode_df: 
        modes of varibility dataframe. 
    mode_start_year: 
        start year of mode.
    mode_start_month:
        start month of mode.
    norm:
        whether normalize streamflow. Default False and should use standard anomalies. 
    """
    # get discharge start and end time based on time lead and mode time. 
    dis_start_year, dis_start_month = get_year_month(mode_start_year, mode_start_month, time_lead)
    dis_end_year, dis_end_month = get_year_month(mode_end_year, mode_end_month, time_lead)
    # Build modes df with different lags from time_window:
    dfs = [mode_df]
    if time_window>=1:
        for i in range(1, time_window+1): # 0 is itself (concurrent)
            modes_df_lag = mode_df.shift(i)
            modes_df_lag.columns = [col+'_lag'+str(i) for col in modes_df_lag.columns]
            dfs.append(modes_df_lag)
        modes_df_new = pd.concat(dfs, axis=1)
    else:
        modes_df_new = mode_df
    # print(modes_df_new.shape, ' modes_df_new shape')
    dis_start_index = (dis_start_year+int(dis_start_month>=10), dis_start_year, dis_start_month)
    dis_end_index = (dis_end_year+int(dis_end_month>=10), dis_end_year, dis_end_month)
    mode_start_index = (mode_start_year+int(mode_start_month>=10), mode_start_year, mode_start_month)
    mode_end_index = (mode_end_year+int(mode_end_month>=10), mode_end_year, mode_end_month)
    # print(dis_start_index, dis_end_index, mode_start_index, mode_end_index)
    dfs = []
    for i, station_id in enumerate(station_ids):
        lat = camel_topo[camel_topo['gauge_id']==int(station_id)]['gauge_lat'].values[0]/50
        lon = camel_topo[camel_topo['gauge_id']==int(station_id)]['gauge_lon'].values[0]/100
        ele = camel_topo[camel_topo['gauge_id']==int(station_id)]['elev_mean'].values[0]/1000
        area = camel_topo[camel_topo['gauge_id']==int(station_id)]['area_gages2'].values[0]
        static = pd.DataFrame(np.array([lat, lon, ele, area, station_id]).reshape(-1, 5), columns=['lat', 'lon', 'ele', 'area', 'station_id'])
        slice_df = dis_df.loc[dis_start_index:dis_end_index, str(station_id)].rename('Q_sim')
        static = pd.concat([static]*len(slice_df))
        static = static.set_index([slice_df.index])
        if norm:
            slice_df = (slice_df-slice_df.mean())/slice_df.std()
        # print(slice_df.shape, ' ', slice_df.index)
        # print(modes_df_new.loc[mode_start_index:mode_end_index].shape)
        mode_slice_df = modes_df_new.loc[mode_start_index:mode_end_index].set_index([slice_df.index]) # 1 on 1 

        df = pd.concat([slice_df, mode_slice_df, static], axis=1)
        dfs.append(df)
    hist_df = pd.concat(dfs)
    return hist_df

def normalize_mi_score(x1, x2, bins=10):
    c_xy, x_edges, y_edges = np.histogram2d(x1, x2, bins=bins)
    c_x, c_x_edges = np.histogram(x1, x_edges, density=True)
    c_y, c_y_edges = np.histogram(x2, y_edges, density=True)
    # Transform to probabilities (i.e., sum to 1). 
    c_x = c_x*np.diff(c_x_edges)
    c_y = c_y*np.diff(c_y_edges)
    # h_true, h_pred = entropy(c_x), entropy(c_y)
    h_true, h_pred = -(c_x*np.log(np.abs(c_x+1e-9))).sum(), -(c_y*np.log(np.abs(c_y+1e-9))).sum()
    mean_entropy = np.min([h_true, h_pred])
    mi = mutual_info_score(None, None, contingency=c_xy)
    n_mi = mi/mean_entropy
    return n_mi, mi


def liang(y1, y2, np_t=1):
    """
    Liang causality
    causal relationship from y2 to y1. Y2 is the reason and Y1 is the result. 
    np_t: Euler forward time step.
    
    """
    dt = 1
    npt = np_t*dt
    # print(y1.shape)
    nm = np.size(y1)
    grad1 = (y1[0+npt:]-y1[0:-npt])/npt # Euler forward/difference time series. 
    grad2 = (y2[0+npt:]-y2[0:-npt])/npt
    y1 = y1[:-np_t] # the common part with the Euler series. 
    y2 = y2[:-np_t]
    N = nm-np_t # length of y1 now. 
    C = np.cov(y1, y2)
    detC = np.linalg.det(C)

    dC = np.ndarray((2, 2))
    dC[0, 0] = np.sum((y1-np.mean(y1))*(grad1-np.mean(grad1)))
    dC[0, 1] = np.sum((y1-np.mean(y1))*(grad2-np.mean(grad2)))
    dC[1, 0] = np.sum((y2-np.mean(y2))*(grad1-np.mean(grad1)))
    dC[1, 1] = np.sum((y2-np.mean(y2))*(grad2-np.mean(grad2)))
    dC /= N-1
    a11 = C[1, 1]*dC[0, 0] - C[0, 1]*dC[1, 0]
    a12 = -C[0, 1]*dC[0, 0] + C[0, 0]*dC[1, 0]
    a11 /= detC
    a12 /= detC
    f1 = np.mean(grad1)-a11*np.mean(y1)-a12*np.mean(y2)
    R1 = grad1-(f1+a11*y1+a12*y2)
    Q1 = np.sum(R1*R1)
    b1 = np.sqrt(Q1*dt/N)
    
    NI = np.ndarray((4, 4))
    NI[0, 0] = N*dt/(b1**2)
    NI[1, 1] = dt/(b1**2)*np.sum(y1*y1)
    NI[2, 2] = dt/(b1**2)*np.sum(y2*y2)
    NI[3, 3] = 3*dt/b1**4*np.sum(R1*R1)-N/(b1**2)
    NI[0, 1] = dt/(b1**2)*np.sum(y1)
    NI[0, 2] = dt/(b1**2)*np.sum(y2)
    NI[0, 3] = 2*dt/(b1**3)*np.sum(R1)
    NI[1, 2] = dt/(b1**2)*np.sum(y1*y2)
    NI[1, 3] = 2*dt/(b1**3)*np.sum(R1*y1)
    NI[2, 3] = 2*dt/(b1**3)*np.sum(R1*y2)

    NI[1, 0] = NI[0, 1]
    NI[2, 0] = NI[0, 3]
    NI[2, 1] = NI[1, 2]
    NI[3, 0] = NI[0, 3]
    NI[3, 1] = NI[1, 3]
    NI[3, 2] = NI[2, 3]

    invNI = np.linalg.pinv(NI)
    var_a12 = invNI[2, 2]
    T21 = C[0, 1]/C[0, 0]*(-C[1, 0]*dC[0, 0]+C[0, 0]*dC[1, 0])/detC
    var_T21 = (C[0, 1]/C[0, 0])**2*var_a12
    dH1_star = a11
    dH1_noise = b1**2/(2*C[0, 0])
    Z = np.abs(T21)+np.abs(dH1_star)+np.abs(dH1_noise)
    tau21 = T21/Z
    dH1_star = dH1_star/Z
    dH1_noise = dH1_noise/Z
    # From Liang's matlab for confidence interval
    # From the standard normal distribution table, 
    # at level alpha=95%, z=1.96
    #                99%, z=2.56
    # 		         90%, z=1.65
    #
    z99 = 2.56
    z95 = 1.96
    z90 = 1.65

    err90 = np.sqrt(var_T21) * z90
    err95 = np.sqrt(var_T21) * z95
    err99 = np.sqrt(var_T21) * z99
    '''
    signif_test_func = {
            'isopersist': signif_isopersist,
            # 'isospec': signif_isospec,
        }
    
    signif_test = 'isopersist'
    qs=[0.005, 0.025, 0.05, 0.95, 0.975, 0.995]
    signif_dict = signif_test_func[signif_test](y1, y2, method='liang', nsim=1000, qs=qs, npt=1)
    T21_noise_qs = signif_dict['T21_noise_qs']
    tau21_noise_qs = signif_dict['tau21_noise_qs']
    '''
    res = {
        'T21': T21,
        'tau21': tau21,
        'Z': Z,
        'dH1_star': dH1_star,
        'dH1_noise': dH1_noise, 
        'err90': err90,
        'err95': err95,
        'err99': err99,
        # 'T21_noise': T21_noise_qs,
        # 'tau21_noise': tau21_noise_qs
    }
    return res
