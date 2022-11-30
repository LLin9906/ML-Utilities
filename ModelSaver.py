from collections import namedtuple

import numpy as np
import pandas as pd
import joblib

Model = namedtuple('Model', ['model', 'model_name', 'cv_score', 'test_score'])
class ModelSaver:
    
    def __init__(self, filename='models.joblib'):
        self.models = []
        self.filename = filename
        
    def save_model(self, model, model_name, cv_score=None, test_score=None, print_res=True):
        self.models.append(Model(model, model_name, cv_score, test_score))
        joblib.dump(self, self.filename)
        
        if print_res:
            print(f"**** {model_name} ****")
            print(f"cross validation score: {cv_score}")
            print(f"test score: {test_score}")
    
    def get_model_summary(self):
        model_info = [model[1:] for model in self.models]
        return pd.DataFrame(model_info, columns=['model', 'cv_score', 'test_score']).set_index('model')