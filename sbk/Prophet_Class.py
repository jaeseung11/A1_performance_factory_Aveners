import datetime as dt
import itertools
import numpy as np
import pandas as pd
import os
import holidays
import matplotlib.pyplot as plt
import seaborn as sns
import holidays


from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from plotnine import *



class Prophet_Class:
    
    def __init__(self, data: pd.DataFrame, target: str, param_grid: dict):
        
        
        self.df = pd.DataFrame()
        
        self.df['y'] = target
        
        self.df['ds'] = data.index
        
        
        self.param_grid = param_grid
        

    
    
    def metrics(self, metrix: str, horizon: str):
        
    # horizon 사용시 e.g.'365 days'
    # metrix = mse rmse mae mdape smape coverage 
    
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        evaluation_metrix = []  
    
        for params in all_params:

            model = Prophet(**params)
            model.fit(df)  

            df_cv = cross_validation(model, horizon=horizon, parallel="processes")

            df_p = performance_metrics(df_cv)

            evaluation_metrix.append(df_p[metrix].values[0])
            print(evaluation_metrix)

        tuning_results = pd.DataFrame(all_params)
        tuning_results[metrix] = evaluation_metrix

 
        tuning_results = tuning_results.sort_values(by=[metrix])


        self.final_params = tuning_results.iloc[0, :-1].to_dict()
    
    
    
    
    def final(periods: int, freq: str):
        
        m = Prophet(**final_params).fit(df)
        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast = m.predict(future)

    
        return final_params, forecast.loc[df.shape[0]:, 'yhat'].values
        
        
    