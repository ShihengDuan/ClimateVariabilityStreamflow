import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import os

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 

scores_LOD = np.load('dataResult/scores_LOD_smooth.npy')
scores_LA = np.load('dataResult/scores_LA_smooth.npy')
scores_RD = np.load('dataResult/scores_RD_smooth.npy')
scores_LR = np.load('dataResult/scores_LR_smooth.npy')
scores_ML_LR = np.load('dataResult/scores_ML_smooth_LR.npy')
scores_ML_ori = np.load('dataResult/scores_ML_smooth.npy')

scores_LOD_lag3 = np.load('dataResult/scores_LOD_smooth_lag3.npy')
scores_LA_lag3 = np.load('dataResult/scores_LA_smooth_lag3.npy')
scores_RD_lag3 = np.load('dataResult/scores_RD_smooth_lag3.npy')
scores_LR_lag3 = np.load('dataResult/scores_LR_smooth_lag3.npy')
scores_ML_LR_lag3 = np.load('dataResult/scores_ML_smooth_LR_lag3.npy')
scores_ML_ori_lag3 = np.load('dataResult/scores_ML_smooth_lag3.npy')
# train
scores_LOD_train = np.load('dataResult/scores_LOD_smooth_train.npy')
scores_LA_train = np.load('dataResult/scores_LA_smooth_train.npy')
scores_RD_train = np.load('dataResult/scores_RD_smooth_train.npy')
scores_LR_train = np.load('dataResult/scores_LR_smooth_train.npy')
# scores_ML_LR_train = np.load('dataResult/scores_ML_smooth_LR_train.npy')
scores_ML_ori_train = np.load('dataResult/scores_ML_smooth_train.npy')

scores_LOD_lag3_train = np.load('dataResult/scores_LOD_smooth_lag3_train.npy')
scores_LA_lag3_train = np.load('dataResult/scores_LA_smooth_lag3_train.npy')
scores_RD_lag3_train = np.load('dataResult/scores_RD_smooth_lag3_train.npy')
scores_LR_lag3_train = np.load('dataResult/scores_LR_smooth_lag3_train.npy')
# scores_ML_LR_lag3_train = np.load('dataResult/scores_ML_smooth_LR_lag3_train.npy')
scores_ML_ori_lag3_train = np.load('dataResult/scores_ML_smooth_lag3_train.npy')

print(scores_ML_ori_lag3.shape)
print(np.median(scores_LR[:, -1]))
LODs = np.zeros((25, 6))
LAs = np.zeros((25, 6))
RDs = np.zeros((25, 6))
LRs = np.zeros((25, 6))
MLLRs = np.zeros((25, 6))
MLs = np.zeros((25, 6))

LODs_train = np.zeros((25, 6))
LAs_train = np.zeros((25, 6))
RDs_train = np.zeros((25, 6))
LRs_train = np.zeros((25, 6))
MLs_train = np.zeros((25, 6))

station_peaks = [5, 5, 3, 3, 2, 2, 3, 6, 5, 5, 5, 2, 3, 2, 2, 3, 1, 1, 1, 2, 12, 1, 3, 5, 2]
for i in range(len(station_peaks)):
    peak = station_peaks[i]
    if peak>3 and peak<12: # summer peaks
        LODs[i, :] = scores_LOD_lag3[i, :]
        LAs[i, :] = scores_LA_lag3[i, :]
        RDs[i, :] = scores_RD_lag3[i, :]
        LRs[i, :] = scores_LR_lag3[i, :]
        MLs[i, :] = scores_ML_ori_lag3[i, :]
        MLLRs[i, :] = scores_ML_LR_lag3[i, :]

        LODs_train[i, :] = scores_LOD_lag3_train[i, :]
        LAs_train[i, :] = scores_LA_lag3_train[i, :]
        RDs_train[i, :] = scores_RD_lag3_train[i, :]
        LRs_train[i, :] = scores_LR_lag3_train[i, :]
        MLs_train[i, :] = scores_ML_ori_lag3_train[i, :]
        
    else:
        LODs[i, :] = scores_LOD[i, :]
        LAs[i, :] = scores_LA[i, :]
        RDs[i, :] = scores_RD[i, :]
        LRs[i, :] = scores_LR[i, :]
        MLs[i, :] = scores_ML_ori[i, :]
        MLLRs[i, :] = scores_ML_LR[i, :]

        LODs_train[i, :] = scores_LOD_train[i, :]
        LAs_train[i, :] = scores_LA_train[i, :]
        RDs_train[i, :] = scores_RD_train[i, :]
        LRs_train[i, :] = scores_LR_train[i, :]
        MLs_train[i, :] = scores_ML_ori_train[i, :]
        
def plot_lines(LRs, RDs, LAs, LODs, MLLRs, file_name='smooth-line.png'):
    fig = plt.figure(figsize=(10, 5))
    # print(np.shape(scores_LA)) # 25, 6. 
    colors=['tab:red', 'tab:blue', 'tab:orange',
            'tab:green', 'tab:purple']
    for i in range(6):
        bplot = plt.boxplot(LRs[:, i], positions=[.2+i*3], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:red')
        bplot['medians'][0].set_color('black')
        bplot = plt.boxplot(RDs[:, i], positions=[.6+i*3], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:green')
        bplot['medians'][0].set_color('black')
        bplot = plt.boxplot(LAs[:, i], positions=[1+i*3], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:blue')
        bplot['medians'][0].set_color('black')
        bplot = plt.boxplot(LODs[:, i], positions=[1.4+i*3], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:purple')
        bplot['medians'][0].set_color('black')
        bplot = plt.boxplot(MLLRs[:, i], positions=[1.8+i*3], patch_artist=True)
        bplot['boxes'][0].set_facecolor('tab:orange')
        bplot['medians'][0].set_color('black')
    plt.xticks(np.arange(1, 17, 3), np.arange(1, 7))
    # [1, 4, 7, 10, 13, 16]
    main_linewidth = 1
    sub_linewidth = .5
    for i in range(5):
        if i==0:
            plt.plot([2+i*3, 3+i*3], np.median(LRs[:, i:i+2], axis=0), 
                    label='Linear', color='tab:red', linestyle='-', 
                    linewidth=main_linewidth)
            plt.plot([2+i*3, 3+i*3], np.median(LAs[:, i:i+2], axis=0), 
                    label='Lasso', color='tab:blue', linestyle='-', 
                    linewidth=main_linewidth)
            plt.plot([2+i*3, 3+i*3], np.median(RDs[:, i:i+2], axis=0), 
                    label='Ridge', color='tab:green', linestyle='-', 
                    linewidth=main_linewidth)
            plt.plot([2+i*3, 3+i*3], np.median(LODs[:, i:i+2], axis=0), 
                    label='LOD', color='tab:purple', linestyle='-', 
                    linewidth=main_linewidth)
            plt.plot([2+i*3, 3+i*3], np.median(MLLRs[:, i:i+2], axis=0), 
                    label='AutoML', color='tab:orange', linestyle='-', 
                    linewidth=main_linewidth)
        else:
            plt.plot([2+i*3, 3+i*3], np.median(LRs[:, i:i+2], axis=0), 
                    color='tab:red', linestyle='-', linewidth=main_linewidth)
            plt.plot([2+i*3, 3+i*3], np.median(LAs[:, i:i+2], axis=0), 
                    color='tab:blue', linestyle='-', linewidth=main_linewidth)
            plt.plot([2+i*3, 3+i*3], np.median(RDs[:, i:i+2], axis=0), 
                    color='tab:green', linestyle='-', linewidth=main_linewidth)
            plt.plot([2+i*3, 3+i*3], np.median(LODs[:, i:i+2], axis=0), 
                    color='tab:purple', linestyle='-', linewidth=main_linewidth)
            plt.plot([2+i*3, 3+i*3], np.median(MLLRs[:, i:i+2], axis=0), 
                    color='tab:orange', linestyle='-', linewidth=main_linewidth)
        plt.plot([.2+i*3, 2+i*3], 
                [np.median(LRs[:, i]), np.median(LRs[:, i])],
                color='tab:red', linestyle='--', linewidth=sub_linewidth)
        plt.plot([.6+i*3, 2+i*3], 
                [np.median(RDs[:, i]), np.median(RDs[:, i])],
                color='tab:green', linestyle='--', linewidth=sub_linewidth)
        plt.plot([1+i*3, 2+i*3], 
                [np.median(LAs[:, i]), np.median(LAs[:, i])],
                color='tab:blue', linestyle='--', linewidth=sub_linewidth)
        plt.plot([1.4+i*3, 2+i*3], 
                [np.median(LODs[:, i]), np.median(LODs[:, i])],
                color='tab:purple', linestyle='--', linewidth=sub_linewidth)
        plt.plot([1.8+i*3, 2+i*3], 
                [np.median(MLLRs[:, i]), np.median(MLLRs[:, i])],
                color='tab:orange', linestyle='--', linewidth=sub_linewidth)
        
        plt.plot([3+i*3, 3.2+i*3], 
                [np.median(LRs[:, i+1]), np.median(LRs[:, i+1])],
                color='tab:red', linestyle='--', linewidth=sub_linewidth)
        plt.plot([3+i*3, 3.6+i*3], 
                [np.median(RDs[:, i+1]), np.median(RDs[:, i+1])],
                color='tab:green', linestyle='--', linewidth=sub_linewidth)
        plt.plot([3+i*3, 4+i*3], 
                [np.median(LAs[:, i+1]), np.median(LAs[:, i+1])],
                color='tab:blue', linestyle='--', linewidth=sub_linewidth)
        plt.plot([3+i*3, 4.4+i*3], 
                [np.median(LODs[:, i+1]), np.median(LODs[:, i+1])],
                color='tab:purple', linestyle='--', linewidth=sub_linewidth)
        plt.plot([3+i*3, 4.8+i*3], 
                [np.median(MLLRs[:, i+1]), np.median(MLLRs[:, i+1])],
                color='tab:orange', linestyle='--', linewidth=sub_linewidth)


    plt.legend(fontsize=12)
    plt.xlabel('Number of EOFs', fontsize=12)
    plt.ylabel('Median R2 Score', fontsize=12)

    plt.savefig(file_name, bbox_inches='tight', dpi=180)
plot_lines(LRs, RDs, LAs, LODs, MLs, file_name='smooth-line.png')
plot_lines(scores_LR, scores_RD, scores_LA, scores_LOD, scores_ML_ori, file_name='smooth-line-concurrent.png')
plot_lines(LRs_train, RDs_train, LAs_train, LODs_train, MLs_train, file_name='smooth-line-train.png')

print(np.median(MLLRs[:, -1]), ' MLLR')
print(np.median(LODs[:, -1]), ' LOD')
print(np.median(LAs[:, -1]), ' LA')
print(np.median(RDs[:, -1]), ' RD')
print(np.median(LRs[:, -1]), ' LR')
print(np.median(MLs[:, -1]), ' ML')

