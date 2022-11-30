import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class Visualizer:
    def __init__(self, X: pd.DataFrame) -> None:
        self.X = X
        
    def visualize_num_attr(self, nrows=None, ncols=None, figsize=None, 
                           kind='histplot', X=None, sort_features=True,
                           save_fig=False, filename='Num_attr.png'):
        """ Visualize all numerical attributes in a DataFrame using histogram or boxplot.
        """
        if X is None:
            X = self.X
        num_X = self.X.select_dtypes(include=np.number)
        features = num_X.columns
        if sort_features:
            features = features.sort_values()
        n_feature = num_X.shape[1]
        
        nrows, ncols = self._get_num_row_col(nrows, ncols, n_feature)
        
        if figsize is None:
            figsize = (ncols * 4, ncols * 4)
        
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, facecolor=(1, 1, 1))
        axs_flat = axs.flatten()
        
        if kind == 'histplot':
            plotting_func = sns.histplot
        elif kind == 'boxplot':
            plotting_func = sns.boxplot
        else:
            raise AttributeError("Parameter kind must be 'histplot' or 'boxplot'!")
        
        for i in range(n_feature):
            ax = axs_flat[i]
            plotting_func(data=num_X, x=features[i], ax=ax)
        
        if save_fig:
            fig.savefig(filename)
        return axs
    
    def visualize_cat_attr(self, nrows=None, ncols=None, figsize=None,
                           X=None, sort_features=True,
                           save_fig=False, filename='Cat_attr'):
        """ Visualize all categorical attributes in a DataFrame using histogram.
        """
        if X is None:
            X = self.X
        cat_X = X.select_dtypes(include=object)
        features = cat_X.columns
        if sort_features:
            features = features.sort_values()
        n_feature = cat_X.shape[1]
        
        nrows, ncols = self._get_num_row_col(nrows, ncols, n_feature)
        
        if figsize is None:
            figsize = (ncols * 4, ncols * 4)
            
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, facecolor=(1, 1, 1))
        axs_flat = axs.flatten()
        
        for i in range(n_feature):
            ax = axs_flat[i]
            sns.histplot(data=cat_X, x=features[i], ax=ax)
        
        if save_fig:
            fig.savefig(filename)
        
        return axs
        
    def corr_heatmap(self, yname, n_features=10, figsize=None, X=None,
                     save_fig=False, filename='Corr_heatmap'):
        """ Get the top n_features features that are most correlated with y,
            and then plot a heatmap of correlation matrix.
        """
        if X is None:
            X = self.X
        if figsize is None:
            figsize = (n_features, n_features)
        corr_mat = X.corr().sort_values(by=yname, ascending=False)
        max_corr_name = corr_mat.index[:n_features+1]
        corr_mat = corr_mat.reindex(max_corr_name, axis=1).head(n_features+1)
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=(1, 1, 1))
        sns.heatmap(corr_mat, annot=True, ax=ax)
        
        if save_fig:
            fig.savefig(filename)
        return ax, corr_mat
    
    def plot_features(self, features, yname, nrows=None, ncols=None, kind='stripplot', 
                      figsize=None, X=None, save_fig=False, filename='Features_plot',
                      **kwargs):
        """ Make plots for each feature in the input features.
        """
        if X is None:
            X = self.X
        n_feature = len(features)
        nrows, ncols = self._get_num_row_col(nrows, ncols, n_feature)
        if figsize is None:
            figsize = (ncols * 4, nrows * 4)
        
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, facecolor=(1, 1, 1))
        axs_flat = axs.flatten()
        
        if kind == 'auto':
            for i, feature in enumerate(features):
                ax = axs_flat[i]
                if X[feature].nunique() > 5:
                    sns.stripplot(data=X, x=feature, y=yname, ax=ax, **kwargs)
                else:
                    sns.violinplot(data=X, x=feature, y=yname, ax=ax, **kwargs)
        else:
            if kind == 'scatterplot':
                plotting_func = sns.scatterplot
            elif kind == 'boxplot':
                plotting_func = sns.boxplot
            elif kind == 'boxenplot':
                plotting_func = sns.boxenplot
            elif kind == 'violinplot':
                plotting_func = sns.violinplot
            elif kind == 'stripplot':
                plotting_func = sns.stripplot
            else:
                raise AttributeError(f"Parameter kind={kind} is invalid!")
        
            for i, feature in enumerate(features):
                ax = axs_flat[i]
                plotting_func(data=X, x=feature, y=yname, ax=ax, **kwargs)
        
        if save_fig:
            fig.savefig(filename)
        
        return axs
    
    def _get_num_row_col(cls, nrows, ncols, n_feature):
        if nrows is None and ncols is None:
            ncols = 5
            nrows = n_feature // ncols + 1
        elif nrows is None:
            nrows = -(n_feature // -ncols)
        elif ncols is None:
            ncols = -(n_feature // -nrows)

        return nrows, ncols
    