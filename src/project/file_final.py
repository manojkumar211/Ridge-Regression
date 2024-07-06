from joblib import dump
from joblib import load
import pandas as pd
import pickle
from models import Ridge_regression


with open('file_ridge.pickle','wb') as f:
    pickle.dump(Ridge_regression.ridge_model,f) # type: ignore

