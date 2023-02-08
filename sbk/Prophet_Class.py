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



class Prophet_model:

    def __init__(self, df: pd.DataFrame, param_grid: dict, matrics: str, periods=30, horizon='30 days'): 
        self.df = df
        self.df['y'] = df['y']
        self.df['ds'] = df['ds']
        self.param_grid = param_grid
        self.matrics = matrics
        self.periods = periods
        self.horizon = horizon

        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        evaluation_metrix = []  # Store the RMSEs for each params here
        # 하이퍼 파라미터만 조정
        
        holiday = pd.DataFrame([])
        for date, name in sorted(holidays.KR(years=[2018,2019,2020,2021]).items()):
            # years만 조정
            holiday = holiday.append(pd.DataFrame({'ds': date, 'holiday' : "KR-Holidays"}, index=[0]), ignore_index=True)
        holiday['ds'] = pd.to_datetime(holiday['ds'], format='%Y-%m-%d', errors='ignore')
        

        for params in all_params:
            model = Prophet(**params)
            model.fit(df)  # Fit model with given params
            df_cv = cross_validation(model, horizon=self.horizon, parallel="processes")
            df_p = performance_metrics(df_cv)
            evaluation_metrix.append(df_p[self.matrics].values[0])

        tuning_results = pd.DataFrame(all_params)
        tuning_results[self.matrics] = evaluation_metrix

        # Sorted by rmes values
        tuning_results = tuning_results.sort_values(by=[self.matrics])

        # Pick the optimized(having minimum rmse value) hyper-parameter combination
        self.final_params = tuning_results.iloc[0, :-1].to_dict()

        # Train the final model with optimized params
        self.m = Prophet(**self.final_params, holidays=holiday).fit(df)
        future = self.m.make_future_dataframe(periods=self.periods, freq='D')
        self.forecast = self.m.predict(future)

    def get_results(self):
        return self.final_params, self.forecast.loc[self.df.shape[0]:, 'yhat'].values
        # 마지막 리턴값만 조정

        
    