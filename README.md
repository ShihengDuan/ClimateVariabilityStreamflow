```build_csv*.py``` are used to build CSV files from netCDF4 files on Gates/Local machine.    
```smooth-result-build-npy.py``` read predictions and calculate R2 score for different models.   
```smooth-result.py``` read R2 npy file and plot box and line plots.   
```smooth-spatial.py``` plot spatial distribution of concurrent R2.    
```smooth-spatial-lag.py``` plot spatial distribution of lag R2.   
```model-train-smooth.py``` train models with args (pre_season, smooth, eof, lag (default0)).  
```include/``` for reduced models.   

```real-smooth.py``` apply ML models with USGS and reanalysis dataset.  
 
```train_full.py``` train ML models with historical CMIP6 in a cross-validation setting.    
```train_full_ssp.py``` train ML models with historical+ssp CMIP6 in a train-test setting.   

```Figures/``` plot scripts.  
```dataResults``` aggregates data and calculate R2 scores.   
