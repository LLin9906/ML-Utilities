import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

class LinRegDiagnostics:
    def __init__(self, X, y):
        self.y = y
        self.X = X
    
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, X_new):
        self._X = X_new
        self.ols_res = sm.OLS(self.y, X_new).fit()
        self.ols_influence = OLSInfluence(self.ols_res)
        
    def resid_vs_fitted(self, ax=None, figsize=(8, 6)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(self.ols_res.predict(self._X), self.ols_influence.resid_studentized_external)
        
        return ax
    
    def qqplot(self, ax=None, figsize=(8, 6)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        sm.qqplot(data=self.ols_influence.resid_studentized_external, ax=ax, line='s')
        
        return ax
        
    def vif_series(self, sorted=True):
        vif_list = [variance_inflation_factor(self._X, i) for i in range(self._X.shape[1])]
        vif_series = pd.Series(vif_list, index=self._X.columns)
        self._vif_series = vif_series
        return vif_series.sort_values(ascending=False) if sorted else vif_series
    
    def get_X_y_truncated(self, values: pd.Series, abs_flag: bool = False,
                          max_val=None, n_drop=None):
        if abs_flag:
            values = values.abs()
        
        if max_val is not None:
            X_new = self._X[values <= max_val].sort_index()
        
        if n_drop is not None:
            values = values.sort_values(ascending=False)
            X_new = self._X.loc[values.index[n_drop:]].sort_index()
            
        y_new = self.y.loc[X_new.index]
        
        return X_new, y_new
        
    def plot_abnomral(self, values, abs_flag: bool = False,
                      max_val=None, n_points=None, ax=None, figsize=(8, 6)):
        if abs_flag:
            values = np.abs(values)
        values = pd.Series(values, index=self.y.index)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        y_fitted = self.ols_res.predict(self._X)
        
        if max_val is not None:
            normal_points_idx = values.index[values <= max_val]
            abnormal_points_idx = values.index[values > max_val]
        
        if n_points is not None:
            values = values.sort_values(ascending=False)
            normal_points_idx = values.index[n_points:]
            abnormal_points_idx = values.index[:n_points]
        
        ax.scatter(self.y.loc[normal_points_idx], y_fitted.loc[normal_points_idx],
                   label='normal points')
        ax.scatter(self.y.loc[abnormal_points_idx], y_fitted.loc[abnormal_points_idx],
                   label='abnormal points')
        ax.legend()
        
        return ax