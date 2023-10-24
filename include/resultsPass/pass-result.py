import pickle
import pandas as pd
import os

station_ids = ['10336645', '10336660', '11124500', '11141280', 
               '11143000', '11148900', '11151300', '11230500', 
               '11237500', '11264500', '11266500', '11284400', 
               '11381500', '11451100', '11468500', '11473900', '11475560', 
               '11476600', '11478500', '11480390', '11481200', 
               '11482500', '11522500', '11523200', '11528700'] # 25 in total. 
lasso_ranking = {}
ridge_ranking = {}
lod_ranking = {}
lr_ranking = {}
automl_ranking = {}
pls_ranking = {}
for station in station_ids:
    print(station)
    with open('LASSO/'+station+'-LASSO.p', 'rb') as pfile:
        lasso_results = pickle.load(pfile)
    with open('RIDGE/'+station+'-RIDGE.p', 'rb') as pfile:
        ridge_results = pickle.load(pfile)
    with open('LOD/'+station+'-LOD.p', 'rb') as pfile:
        lod_results = pickle.load(pfile)
    with open('LOD/'+station+'-LOD.p', 'rb') as pfile:
        lod_results = pickle.load(pfile)
    with open('LR/'+station+'-LR.p', 'rb') as pfile:
        lr_results = pickle.load(pfile)
    with open('PLS/'+station+'-PLS.p', 'rb') as pfile:
        pls_results = pickle.load(pfile)
    # AutoML
    automl_results = []
    for level in range(5):
        file = 'AUTOML/'+station+'-AUTOML-level-'+str(level)+'.p'
        if os.path.exists(file):
            with open(file, 'rb') as pfile:
                result = pickle.load(pfile)
            max_mod = result['max_mod']
            automl_results.append(max_mod)
        else:
            automl_results.append('NAN')
    for i in range(len(ridge_results)):
        print(i)
        print('Lasso: ', lasso_results[i])
        print('Ridge: ', ridge_results[i])
        print('LOD: ', lod_results[i])
        print('LR: ', lr_results[i])
    lasso_ranking[station] = [i[0] for i in lasso_results]
    ridge_ranking[station] = [i[0] for i in ridge_results]
    lod_ranking[station] = [i[0] for i in lod_results]
    lr_ranking[station] = [i[0] for i in lr_results]
    pls_ranking[station] = [i[0] for i in pls_results]
    automl_ranking[station] = automl_results
    print('\n')
lod_results = pd.DataFrame.from_dict(lod_ranking, orient='index', columns=['F1', 'F2', 'F3', 'F4', 'F5'])
ridge_results = pd.DataFrame.from_dict(ridge_ranking, orient='index', columns=['F1', 'F2', 'F3', 'F4', 'F5'])
lasso_results = pd.DataFrame.from_dict(lasso_ranking, orient='index', columns=['F1', 'F2', 'F3', 'F4', 'F5'])
lr_results = pd.DataFrame.from_dict(lr_ranking, orient='index', columns=['F1', 'F2', 'F3', 'F4', 'F5'])
pls_results = pd.DataFrame.from_dict(pls_ranking, orient='index', columns=['F1', 'F2', 'F3', 'F4', 'F5'])
automl_results = pd.DataFrame.from_dict(automl_ranking, orient='index', columns=['F1', 'F2', 'F3', 'F4', 'F5'])
print(lr_results)
dfs = [lod_results, ridge_results, lasso_results, lr_results, pls_results, automl_results]
model_names = ['lod', 'ridge', 'lasso', 'lr', 'pls', 'automl']
results = pd.concat(dfs, keys=model_names, axis=0)

# Create a MultiIndex for the rows
station_names = lod_results.index
multi_index = pd.MultiIndex.from_product([model_names, station_names], names=['model', 'station'])
print(results)
# Set the MultiIndex for the concatenated DataFrame
results.index = multi_index
results = results.swaplevel('station', 'model')
results = results.sort_index(level='model')
# Print the DataFrame
print(results)
results.to_csv('results.csv')