import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector

cat2num_code = {'a': {'b': 1}}
y_train = pd.Series(range(5))

class Cat2NumTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.replace(cat2num_code)
        cats = list(cat2num_code.keys())
        X[cats] = X[cats].astype(np.float64)
        return X
    

num_attrs_no_info = []
cat_attrs_no_info = []
class NoInfoFeatureDeletor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.drop(num_attrs_no_info + cat_attrs_no_info, axis=1)
        return X
    

nan_features = []
class NanFeatureDeletor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        X = X.drop(nan_features, axis=1)
        return X
    
    
class CatAttrEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        encoder = OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist',
                                min_frequency=0.02, sparse=False)
        self.transformer = ColumnTransformer(
            [('Num', 'passthrough', make_column_selector(dtype_include=np.number)),
             ('Cat', encoder, make_column_selector(dtype_include=object))],
        )
        self.transformer.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame):
        return pd.DataFrame(self.transformer.transform(X),
                            index=y_train.index,
                            columns=self.transformer.get_feature_names_out())
        

class MissingValImputer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5, weights='distance')
        self.imputer.fit(self.scaler.fit_transform(X))
        return self
    
    def transform(self, X: pd.DataFrame):
        return pd.DataFrame(self.scaler.inverse_transform(
                                self.imputer.transform(self.scaler.transform(X))),
                            index=y_train.index, 
                            columns=X.columns)